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

import cvcuda
import torch
import numpy as np
import nvtx


class Preprocessing:
    def __init__(self):
        self.mean_tensor = torch.Tensor([0.485, 0.456, 0.406])
        self.mean_tensor = self.mean_tensor.reshape(1, 1, 1, 3).cuda()
        self.mean_tensor = cvcuda.as_tensor(self.mean_tensor, "NHWC")

        self.stddev_tensor = torch.Tensor([0.229, 0.224, 0.225])
        self.stddev_tensor = self.stddev_tensor.reshape(1, 1, 1, 3).cuda()
        self.stddev_tensor = cvcuda.as_tensor(self.stddev_tensor, "NHWC")

    def __call__(self, frame_nhwc, out_size):
        nvtx.push_range("preprocess")

        if isinstance(frame_nhwc, torch.Tensor):
            frame_nhwc = cvcuda.as_tensor(frame_nhwc, "NHWC")
            has_copy = False
        elif not isinstance(frame_nhwc, cvcuda.Tensor):
            has_copy = True
            nvtx.push_range("copy CPU->GPU")
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(frame_nhwc).to(device="cuda", non_blocking=True), "NHWC"
            )
            nvtx.pop_range()
            nvtx.push_range("processing")

        resized = cvcuda.resize(
            frame_nhwc,
            (
                frame_nhwc.shape[0],
                out_size[1],
                out_size[0],
                frame_nhwc.shape[3],
            ),
            cvcuda.Interp.LINEAR,
        )

        normalized = cvcuda.convertto(resized, np.float32, scale=1 / 255)

        normalized = cvcuda.normalize(
            normalized,
            base=self.mean_tensor,
            scale=self.stddev_tensor,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )

        normalized = cvcuda.reformat(normalized, "NCHW")

        if has_copy:
            nvtx.pop_range()  # processing

        nvtx.pop_range()

        return (
            torch.as_tensor(frame_nhwc.cuda(), device="cuda"),
            resized,
            torch.as_tensor(normalized.cuda(), device="cuda"),
        )


class Postprocessing:
    def __init__(self, output_layout, gpu_output):
        self.gpu_output = gpu_output
        if output_layout != "NCHW" and output_layout != "NHWC":
            raise RuntimeError(
                "Unknown post-processing output layout: %s" % output_layout
            )

        self.output_layout = output_layout

    def __call__(self, probabilities, frame_nhwc, resized_tensor, class_index):
        nvtx.push_range("postprocess")

        if not self.gpu_output:
            nvtx.push_range("processing")

        actual_batch_size = resized_tensor.shape[0]

        if isinstance(frame_nhwc, cvcuda.Tensor):
            probabilities = torch.as_tensor(frame_nhwc.cuda(), device="cuda")

        class_probs = probabilities[:actual_batch_size, class_index, :, :]
        class_probs = torch.unsqueeze(class_probs, dim=-1)
        class_probs *= 255
        class_probs = class_probs.type(torch.uint8)

        cvcuda_class_masks = cvcuda.as_tensor(class_probs.cuda(), "NHWC")

        cvcuda_class_masks_upscaled = cvcuda.resize(
            cvcuda_class_masks,
            (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 1),
            cvcuda.Interp.NEAREST,
        )

        cvcuda_blurred_input_imgs = cvcuda.gaussian(
            resized_tensor, kernel_size=(15, 15), sigma=(5, 5)
        )
        cvcuda_blurred_input_imgs = cvcuda.resize(
            cvcuda_blurred_input_imgs,
            (frame_nhwc.shape[0], frame_nhwc.shape[1], frame_nhwc.shape[2], 3),
            cvcuda.Interp.LINEAR,
        )

        cvcuda_frame_nhwc = cvcuda.as_tensor(frame_nhwc.cuda(), "NHWC")

        cvcuda_image_tensor_nhwc_gray = cvcuda.cvtcolor(
            cvcuda_frame_nhwc, cvcuda.ColorConversion.BGR2GRAY
        )

        cvcuda_jb_masks = cvcuda.joint_bilateral_filter(
            cvcuda_class_masks_upscaled,
            cvcuda_image_tensor_nhwc_gray,
            diameter=5,
            sigma_color=50,
            sigma_space=1,
        )

        # Create an overlay image. We do this by selectively blurring out pixels
        # in the input image where the class mask prediction was absent (i.e. False)
        # We already have all the things required for this: The input images,
        # the blurred version of the input images and the upscale version
        # of the mask.
        cvcuda_composite_imgs_nhwc = cvcuda.composite(
            cvcuda_frame_nhwc,
            cvcuda_blurred_input_imgs,
            cvcuda_jb_masks,
            3,
        )

        if self.output_layout == "NCHW":
            cvcuda_composite_imgs_out = cvcuda.reformat(
                cvcuda_composite_imgs_nhwc, "NCHW"
            )
        else:
            assert self.output_layout == "NHWC"
            cvcuda_composite_imgs_out = cvcuda_composite_imgs_nhwc

        if self.gpu_output:
            cvcuda_composite_imgs_out = torch.as_tensor(
                cvcuda_composite_imgs_out.cuda(), device="cuda"
            )
        else:
            nvtx.pop_range()  # processing
            nvtx.push_range("copy GPU->CPU")
            cvcuda_composite_imgs_out = (
                torch.as_tensor(cvcuda_composite_imgs_out.cuda()).cpu().numpy()
            )
            nvtx.pop_range()

        nvtx.pop_range()  # postprocess

        return cvcuda_composite_imgs_out
