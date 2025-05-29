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

import sys
import pycuda.driver as cuda
import cvcuda
import torch
import nvtx

# Select codec backend ---------------------------------
from opencv_utils import BatchEncoder, BatchDecoder

# Select pre-/post- processing backend -----------------
from cvcuda_utils import Preprocessing, Postprocessing

# Select inference backend -----------------------------
from torch_utils import Segmentation

nvtx.push_range("total")

# Define some pipeline parameters ----------------------
inference_size = (224, 224)
batch_size = 4
args = sys.argv[1:]
if len(args) == 1:
    batch_size = int(args[0])
    print("Using batch size", batch_size)

# Define the objects that handle the pipeline stages ---

# Use the cuda device #0
cuda_dev = cuda.Device(0)
cuda_ctx = cuda_dev.retain_primary_context()
cuda_ctx.push()

# Define the cuda stream we'll use.
cvcuda_stream = cvcuda.Stream()
torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)

# Now define the object that will handle pre-processing
preprocess = Preprocessing()

# Encoder/decoder
decoder = BatchDecoder(
    fname="pexels-ilimdar-avgezer-7081456.mp4", batch_size=batch_size
)
if "DecodeThread" in globals():
    decoder = DecodeThread(decoder)  # noqa: F821

encoder = BatchEncoder(fname="segmented.mp4", fps=decoder.fps)
if "EncodeThread" in globals():
    encoder = EncodeThread(encoder)  # noqa: F821

# Define the oject that will handle post-processing
postprocess = Postprocessing(
    output_layout=encoder.input_layout, gpu_output=encoder.gpu_input
)

# Cat segmentation
inference = Segmentation("cat", batch_size, inference_size)

# Define and execute the processing pipeline ------------
nvtx.push_range("pipeline")

# Fire up encoder/decoder
decoder.start()
encoder.start()

# Loop through all input frames
idx_batch = 0
while True:
    with cvcuda_stream, torch.cuda.stream(torch_stream), nvtx.annotate(
        "batch_%d" % idx_batch
    ):
        # Stage 1: decode
        batch = decoder()
        if batch is None:
            break  # No more frames to decode
        assert idx_batch == batch.idx

        # Stage 2: pre-processing
        orig_tensor, resized_tensor, normalized_tensor = preprocess(
            batch.frame, out_size=inference_size
        )

        # Stage 3: inference
        probabilities = inference(normalized_tensor)

        # Stage 4: post-processing
        blurred_frame = postprocess(
            probabilities,
            orig_tensor,
            resized_tensor,
            inference.class_index,
        )

        # Stage 5: encode
        batch.frame = blurred_frame
        encoder(batch)

        idx_batch = idx_batch + 1

print(idx_batch, "batches processed")

# Make sure encoder finishes any outstanding work
encoder.join()

nvtx.pop_range() # pipeline

cuda_ctx.pop()

nvtx.pop_range() # total
