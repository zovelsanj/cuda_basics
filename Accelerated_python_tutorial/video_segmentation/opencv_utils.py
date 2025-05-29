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

import cv2
import torch
import cvcuda
import numpy as np
import nvtx


class BatchDecoder:
    def __init__(self, fname, batch_size):
        self.batch_size = batch_size
        self.total_decoded = 0

        self._decoder = cv2.VideoCapture(fname)
        if not self._decoder.isOpened():
            raise ValueError("Can not open video file for reading: %s" % fname)

        self.frame_size = (
            int(self._decoder.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._decoder.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = self._decoder.get(cv2.CAP_PROP_FPS)

        self.total_frames = int(self._decoder.get(cv2.CAP_PROP_FRAME_COUNT))

        self.next_idx_batch = 0

    def __call__(self):
        if self.total_decoded == self.total_frames:
            return None

        nvtx.push_range("decode") #_%d" % self.next_idx_batch)

        if self.total_decoded + self.batch_size > self.total_frames:
            actual_batch_size = self.total_frames - self.total_decoded
        else:
            actual_batch_size = self.batch_size

        frame_list = [self._decoder.read()[-1] for x in range(actual_batch_size)]
        image_tensor_nhwc = np.stack(frame_list, axis=0)
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


class BatchEncoder:
    def __init__(self, fname, fps):
        self.fname = fname
        self.fps = fps
        self._encoder = None
        self.input_layout = "NHWC"
        self.gpu_input = False

    def __call__(self, batch):
        frame = batch.frame

        if self._encoder is None:
            self._encoder = cv2.VideoWriter(
                self.fname,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (frame.shape[2], frame.shape[1]),
            )
            if not self._encoder.isOpened():
                raise ValueError("Can not open video file for writing: %s" % self.fname)

        nvtx.push_range("encode") #_%d" % batch.idx)

        for img_idx in range(frame.shape[0]):
            img = frame[img_idx]
            self._encoder.write(img)

        nvtx.pop_range()

    def start(self):
        pass

    def join(self):
        pass


class Preprocessing:
    def __call__(self, input_nhwc, out_size):
        nvtx.push_range("preprocess")

        if isinstance(input_nhwc, torch.Tensor):
            nvtx.push_range("copy GPU->CPU")
            input_nhwc = input_nhwc.cpu().numpy()
            nvtx.pop_range()
        elif isinstance(input_nhwc, cvcuda.Tensor):
            nvtx.push_range("copy GPU->CPU")
            input_nhwc = torch.as_tensor(input_nhwc.cuda(), device="cuda").cpu().numpy()
            nvtx.pop_range()

        nvtx.push_range("processing")

        downscaled_batch = []
        normalized_batch = []
        # For all images in the batch
        for img in input_nhwc:
            # Step 1: downscale input
            downscaled = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)

            # Step 2: normalize
            normalized = np.float32(downscaled)
            normalized = normalized / 255.0

            mean_arr = np.array([0.485, 0.456, 0.406])
            stddev_arr = np.array([0.229, 0.224, 0.225])
            normalized -= mean_arr
            normalized /= stddev_arr

            downscaled_batch.append(downscaled)
            normalized_batch.append(normalized)

        # Step 3: reformat to NCHW tensor
        downscaled_batch_nhwc = np.stack(downscaled_batch, axis=0)

        normalized_batch_nhwc = np.stack(normalized_batch, axis=0)
        normalized_batch_nchw = np.moveaxis(normalized_batch_nhwc, -1, 1)

        nvtx.pop_range()  # processing

        nvtx.push_range("copy CPU->GPU")
        normalized_batch_nchw_torch = torch.from_numpy(normalized_batch_nchw).to(
            device="cuda", non_blocking=True
        )
        nvtx.pop_range()

        nvtx.pop_range()  # preprocess

        return input_nhwc, downscaled_batch_nhwc, normalized_batch_nchw_torch


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

        actual_batch_size = len(resized_tensor)

        input_image_size = (frame_nhwc.shape[2], frame_nhwc.shape[1])

        # Class mask computation
        nvtx.push_range("copy GPU->CPU")
        probabilities = probabilities.cpu().numpy()
        nvtx.pop_range()

        nvtx.push_range("processing")

        class_probs = probabilities[:actual_batch_size, class_index, :, :]

        # Up-scaling class masks.
        class_probs_upscaled = [
            cv2.resize(
                img,
                input_image_size,
                interpolation=cv2.INTER_NEAREST,
            )
            for img in class_probs
        ]

        class_probs_upscaled = np.stack(class_probs_upscaled, axis=0)
        class_probs_upscaled = np.expand_dims(
            class_probs_upscaled, axis=-1
        )  # Makes it NHWC

        # Repeat in last dimension to make the mask 3 channel
        class_probs_upscaled = np.repeat(class_probs_upscaled, repeats=3, axis=-1)

        class_probs_upscaled_jb = [
            cv2.ximgproc.jointBilateralFilter(
                img.astype(np.float32), mask, d=5, sigmaColor=50, sigmaSpace=1
            )
            for img, mask in zip(frame_nhwc, class_probs_upscaled)
        ]
        class_probs_upscaled_jb = np.stack(class_probs_upscaled_jb, axis=0)

        # Blur low resolution input images
        blurred_inputs = [
            cv2.GaussianBlur(img, ksize=(15, 15), sigmaX=5, sigmaY=5)
            for img in resized_tensor
        ]
        # Up-scale blurred input images.
        blurred_inputs = [
            cv2.resize(
                x,
                input_image_size,
                interpolation=cv2.INTER_LINEAR,
            )
            for x in blurred_inputs
        ]
        blurred_inputs_nhwc = np.stack(blurred_inputs, axis=0)

        composite_imgs = (
            blurred_inputs_nhwc * (1 - class_probs_upscaled_jb)
            + class_probs_upscaled_jb * frame_nhwc
        )

        composite_imgs_nhwc = composite_imgs.astype(np.uint8)

        if self.output_layout == "NCHW":
            composite_imgs_out = np.ascontiguousarray(
                np.moveaxis(composite_imgs_nhwc, -1, 1)
            )
        else:
            assert self.output_layout == "NHWC"
            composite_imgs_out = composite_imgs_nhwc

        nvtx.pop_range()  # processing

        if self.gpu_output:
            nvtx.push_range("copy CPU->GPU")
            composite_imgs_out = torch.as_tensor(composite_imgs_out).to(
                device="cuda", non_blocking=True
            )
            nvtx.pop_range()

        nvtx.pop_range()  # postprocess

        return composite_imgs_out
