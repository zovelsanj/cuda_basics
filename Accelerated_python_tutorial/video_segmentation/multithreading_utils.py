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

import threading
import torch
import queue
import cvcuda
import pycuda.driver as cuda


class EventPool:
    def __init__(self):
        self._queue = queue.SimpleQueue()

    def pop(self):
        try:
            event = self._queue.get_nowait()
        except queue.Empty:
            event = torch.cuda.Event()
            self._queue.put(event)

        return event

    def push(self, event):
        self._queue.put(event)


_event_pool = EventPool()


class DecodeThread(threading.Thread):
    def __init__(self, decoder, stream=None):
        super().__init__()

        # store the current cuda context, it'll be used in the
        # decoder thread
        self.cuda_ctx = cuda.Context.get_device().retain_primary_context()

        self.frame_size = decoder.frame_size
        self.fps = decoder.fps

        self._decoder = decoder
        self._queue = queue.Queue(10)

        if stream is None:
            self.cvcuda_stream = cvcuda.Stream()
        else:
            self.cvcuda_stream = stream
        self.torch_stream = torch.cuda.ExternalStream(self.cvcuda_stream.handle)

    def run(self):
        # The cuda context we're using must be correctly set in this thread
        self.cuda_ctx.push()
        try:
            with self.cvcuda_stream, torch.cuda.stream(self.torch_stream):
                idx_batch = 0
                while True:
                    batch = self._decoder()
                    if batch is None:
                        self._queue.put(None)
                        break
                    assert idx_batch == batch.idx

                    batch.event = _event_pool.pop()
                    batch.event.record(self.torch_stream)

                    self._queue.put(batch)
                    idx_batch = idx_batch + 1
        finally:
            self.cuda_ctx.pop()

    def __call__(self):
        batch = self._queue.get()
        if batch is not None:
            batch.event.wait(torch.cuda.current_stream())
        return batch


class EncodeThread(threading.Thread):
    def __init__(self, encoder, stream=None):
        super().__init__()
        self._encoder = encoder
        self._queue = queue.Queue(10)
        self._must_terminate = False
        self.input_layout = encoder.input_layout
        self.gpu_input = encoder.gpu_input

        # store the current cuda context, it'll be used in the
        # decoder thread
        self.cuda_ctx = cuda.Context.get_device().retain_primary_context()

        if stream is None:
            self.cvcuda_stream = cvcuda.Stream()
        else:
            self.cvcuda_stream = stream
        self.torch_stream = torch.cuda.ExternalStream(self.cvcuda_stream.handle)
        self.exception = None

    def run(self):
        # The cuda context we're using must be correctly set in this thread
        self.cuda_ctx.push()
        try:
            with self.cvcuda_stream, torch.cuda.stream(self.torch_stream):
                while not self._queue.empty() or not self._must_terminate:
                    batch = self._queue.get()
                    try:
                        batch.event.wait(self.torch_stream)
                        self._encoder(batch)
                    finally:
                        self._queue.task_done()
                        _event_pool.push(batch.event)
        except Exception as e:
            self.exception = e
        finally:
            self.cuda_ctx.pop()

    def __call__(self, batch):
        # thread exited with an exception?
        if self.exception is not None:
            raise self.exception  # forward it to the current thread

        self._queue.put(batch)

    def join(self):
        self._must_terminate = True
        if self.exception is None:
            self._queue.join()
        self._encoder.join()
        self.cvcuda_stream.sync()
        threading.Thread.join(self)
