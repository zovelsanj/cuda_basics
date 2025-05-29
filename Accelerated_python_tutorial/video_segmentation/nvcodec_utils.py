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

import torch
import cvcuda
import pycuda.driver as cuda
import PyNvCodec as nvc
from nvcodec_utils_old import nvencoder, nvdecoder
import nvtx


class BatchDecoder:
    def __init__(self, fname, batch_size):
        self.batch_size = batch_size
        self.fname = fname
        self.total_decoded = 0

        nvDemux = nvc.PyFFmpegDemuxer(fname)

        self.frame_size = (nvDemux.Width(), nvDemux.Height())
        self.fps = nvDemux.Framerate()
        self.total_frames = nvDemux.Numframes()

        self.next_idx_batch = 0

        self._decoder = None

    def __call__(self):
        if self.total_decoded == self.total_frames:
            return None

        if self._decoder is None:
            dev_id = torch.cuda.current_device()
            cuda_ctx = cuda.Context.get_device().retain_primary_context()

            cvcuda_stream = cvcuda.Stream.current
            torch_stream = torch.cuda.ExternalStream(cvcuda.Stream.current.handle)

            self._decoder = nvdecoder(
                self.fname, dev_id, cuda_ctx, torch_stream, cvcuda_stream
            )

        nvtx.push_range("decode") # % self.next_idx_batch)

        if self.total_decoded + self.batch_size > self.total_frames:
            actual_batch_size = self.total_frames - self.total_decoded
        else:
            actual_batch_size = self.batch_size

        frame_list = [
            self._decoder.decode_to_tensor() for x in range(actual_batch_size)
        ]

        image_tensor_nchw = torch.stack(frame_list)

        # Convert to NHWC by permuting the axis.
        # Must call contiguous if the tensors are coming from VPF.
        image_tensor_nhwc = image_tensor_nchw.permute(
            0, 2, 3, 1
        ).contiguous()  # from NCHW to NHWC.

        self.total_decoded = self.total_decoded + len(frame_list)

        class Batch:
            pass

        batch = Batch()
        batch.frame = image_tensor_nhwc
        batch.idx = self.next_idx_batch

        self.next_idx_batch = self.next_idx_batch + 1

        nvtx.pop_range()

        return batch

    def start(self):
        pass

    def join(self):
        pass


class BatchEncoder:
    def __init__(self, fname, fps):
        self._encoder = None
        self.fps = fps
        self.fname = fname
        self.input_layout = "NCHW"
        self.gpu_input = True

    def __call__(self, batch):
        frame = batch.frame

        if self._encoder is None:
            # Get the current cuda device and context
            dev_id = torch.cuda.current_device()
            cuda_ctx = cuda.Context.get_device().retain_primary_context()

            self._encoder = nvencoder(
                dev_id,
                frame.shape[3],
                frame.shape[2],
                self.fps,
                self.fname,
                cuda_ctx,
                cvcuda.Stream.current,
            )

        nvtx.push_range("encode") #_%d" % batch.idx)

        for img_idx in range(frame.shape[0]):
            img = frame[img_idx]
            self._encoder.encode_from_tensor(img)

        nvtx.pop_range()

    def start(self):
        pass

    def join(self):
        self._encoder.flush()
        pass
