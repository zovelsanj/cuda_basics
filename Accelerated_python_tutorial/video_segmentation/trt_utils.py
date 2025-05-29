# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorrt as trt
import torch
import os
import numpy as np
import cvcuda
from torchvision.models import segmentation as segmentation_models
import nvtx


class Segmentation:
    def __init__(self, seg_class_name, batch_size, image_size):
        # For TensorRT, the process is the following:
        # We check if there already exists a TensorRT engine generated
        # previously. If not, we check if there exists an ONNX model generated
        # previously. If not, we will generate both of the one by one
        # and then use those.
        # The underlying pytorch model that we use in case of TensorRT
        # inference is the FCN model from torchvision. It is only used during
        # the conversion process and not during the inference.
        onnx_file_path = "model.%d.%d.%d.onnx" % (
            batch_size,
            image_size[1],
            image_size[0],
        )
        trt_engine_file_path = "model.%d.%d.%d.trtmodel" % (
            batch_size,
            image_size[1],
            image_size[0],
        )

        torch_model = segmentation_models.fcn_resnet101
        weights = segmentation_models.FCN_ResNet101_Weights.DEFAULT

        try:
            self.class_index = weights.meta["categories"].index(seg_class_name)
        except ValueError:
            raise ValueError(
                "Requested segmentation class '%s' is not supported by the "
                "DeepLabV3 model. All supported class names are: %s"
                % (seg_class_name, ", ".join(weights.meta["categories"]))
            )

        # Check if we have a previously generated model.
        if not os.path.isfile(trt_engine_file_path):
            if not os.path.isfile(onnx_file_path):
                # First we use PyTorch to create a segmentation model.
                with torch.no_grad():
                    fcn_base = torch_model(weights=weights)

                    class FCN_Softmax(torch.nn.Module):
                        def __init__(self, fcn):
                            super(FCN_Softmax, self).__init__()
                            self.fcn = fcn

                        def forward(self, x):
                            infer_output = self.fcn(x)["out"]
                            return torch.nn.functional.softmax(infer_output, dim=1)

                    fcn_base.eval()
                    pyt_model = FCN_Softmax(fcn_base)
                    pyt_model.cuda()
                    pyt_model.eval()

                    # Allocate a dummy input to help generate an ONNX model.
                    dummy_x_in = torch.randn(
                        batch_size,
                        3,
                        image_size[1],
                        image_size[0],
                        requires_grad=False,
                    ).cuda()

                    # Generate an ONNX model using the PyTorch's onnx export.
                    torch.onnx.export(
                        pyt_model,
                        args=dummy_x_in,
                        f=onnx_file_path,
                        export_params=True,
                        opset_version=15,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={
                            "input": {0: "batch_size"},
                            "output": {0: "batch_size"},
                        },
                    )

                    # Remove the tensors and model after this.
                    del pyt_model
                    del dummy_x_in
                    torch.cuda.empty_cache()

            # Now that we have an ONNX model, we will continue generating a
            # serialized TensorRT engine from it.
            convert_onnx_to_tensorrt(
                onnx_file_path,
                trt_engine_file_path,
                max_batch_size=batch_size,
                max_workspace_size=1,
            )

        # Once the TensorRT engine generation is all done, we load it.
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(trt_engine_file_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            trt_model = runtime.deserialize_cuda_engine(f.read())

        # Create execution context.
        self.model = trt_model.create_execution_context()

        # Allocate the output bindings.
        self.output_tensors, self.output_idx = setup_tensort_bindings(
            trt_model, batch_size
        )

    def __call__(self, tensor):
        nvtx.push_range("inference")

        input_bindings = [tensor.cuda().__cuda_array_interface__["data"][0]]
        output_bindings = []
        for t in self.output_tensors:
            output_bindings.append(t.data_ptr())
        io_bindings = input_bindings + output_bindings

        # Must call this before inference
        binding_i = self.model.engine.get_binding_index("input")
        assert self.model.set_binding_shape(binding_i, tensor.shape)

        self.model.execute_async_v2(
            bindings=io_bindings, stream_handle=cvcuda.Stream.current.handle
        )

        segmented = self.output_tensors[self.output_idx]

        nvtx.pop_range()
        return segmented


def convert_onnx_to_tensorrt(
    onnx_file_path, trt_engine_file_path, max_batch_size, max_workspace_size=5
):
    """
    Converts an ONNX engine to a serialized TensorRT engine.
    :param onnx_file_path: Full path to an existing ONNX file.
    :param trt_engine_file_path: Full path to save the generated TensorRT Engine file.
    :param max_batch_size: The maximum batch size to use in the TensorRT engine.
    :param max_workspace_size: The maximum GPU memory that TensorRT can use (in GB.)
    :return: True if engine was generated. False otherwise.
    """
    # print("Using TensorRT version: %s" % trt.__version__)
    trt_logger = trt.Logger(trt.Logger.ERROR)

    with trt.Builder(trt_logger) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, trt_logger) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024 * max_workspace_size
        )  # Sets workspace size in GB.
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Parse model file
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                raise ValueError("Failed to parse the ONNX engine.")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        profile = builder.create_optimization_profile()
        dynamic_inputs = False
        for input_tensor in inputs:
            if input_tensor.shape[0] == -1:
                dynamic_inputs = True
                min_shape = [1] + list(input_tensor.shape[1:])
                opt_shape = [max_batch_size] + list(input_tensor.shape[1:])
                max_shape = [max_batch_size] + list(input_tensor.shape[1:])
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        if dynamic_inputs:
            config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)

        if not engine:
            raise ValueError("Failed to generate the TensorRT engine.")

        with open(trt_engine_file_path, "wb") as f:
            f.write(engine)


def setup_tensort_bindings(trt_model, batch_size):
    """
    Setups the I/O bindings for a TensorRT engine for the first time.
    :param trt_model: Full path to the generated TensorRT Engine file.
    :return: A list of output tensors and the index of the first output.
    """

    # For TensorRT, we need to allocate the output data buffers.
    # The input data buffers are already allocated by us.
    output_binding_idx = 0
    output_idx = 0
    output_tensors = []

    # Loop over all the I/O bindings.
    for b_idx in range(trt_model.num_io_tensors):
        # Get various properties associated with the bindings.
        b_name = trt_model.get_tensor_name(b_idx)
        b_shape = tuple(trt_model.get_tensor_shape(b_name))
        b_dtype = np.dtype(trt.nptype(trt_model.get_tensor_dtype(b_name))).name

        # Append to the appropriate list.
        if trt_model.get_tensor_mode(b_name) == trt.TensorIOMode.OUTPUT:
            # First allocate on device output buffers, using PyTorch.
            output = torch.zeros(
                size=(batch_size, b_shape[1], b_shape[2], b_shape[3]),
                dtype=getattr(torch, b_dtype),
                device="cuda",
            )

            # Since we know the name of our output layer, we will check against
            # it and grab its binding index.
            if b_name == "output":
                output_idx = output_binding_idx

            output_binding_idx += 1
            output_tensors.append(output)

    return output_tensors, output_idx
