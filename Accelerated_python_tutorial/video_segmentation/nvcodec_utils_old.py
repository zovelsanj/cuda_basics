# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
nvcodec_utils

This file hosts various Video Processing Framework(VPF) related utilities.
"""


import av
import torch
import numpy as np
from fractions import Fraction
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import nvtx


class nvdecoder:
    def __init__(self, enc_file, device_id, ctx, stream_pyt, stream, use_nvtx=False):
        """
        Create instance of HW-accelerated video decoder.
        :param gpu_id: id of video card which will be used for decoding & processing.
        :param enc_file: path to encoded video file.
        """
        # Demuxer is instantiated only to collect required information about
        # certain video file properties.
        # self.device_id = gpu_id
        self.device_id = device_id
        self.ctx = ctx
        self.cuda_stream_pyt = stream_pyt
        self.stream = stream
        self.use_nvtx = use_nvtx
        nvDemux = nvc.PyFFmpegDemuxer(enc_file)
        self.w, self.h = nvDemux.Width(), nvDemux.Height()
        self.fps = nvDemux.Framerate()
        self.total_frames = nvDemux.Numframes()

        # Determine color space and color range for accurate conversion to RGB.
        self.cspace = nvDemux.ColorSpace()
        self.crange = nvDemux.ColorRange()
        if nvc.ColorSpace.UNSPEC == self.cspace:
            self.cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == self.crange:
            self.crange = nvc.ColorRange.JPEG
        self.cc_ctx = nvc.ColorspaceConversionContext(self.cspace, self.crange)

        # In case sample aspect ratio isn't 1:1 we will re-scale the decoded
        # frame to maintain uniform 1:1 ratio across the pipeline.
        sar = 8.0 / 9.0
        self.fixed_h = self.h
        self.fixed_w = int(self.w * sar)

        self.pix_fmt = nvDemux.Format()
        is_yuv420 = (
            nvc.PixelFormat.YUV420 == self.pix_fmt
            or nvc.PixelFormat.NV12 == self.pix_fmt
        )
        is_yuv444 = nvc.PixelFormat.YUV444 == self.pix_fmt

        codec = nvDemux.Codec()
        is_hevc = nvc.CudaVideoCodec.HEVC == codec

        # YUV420 or YUV444 sampling formats are supported by Nvdec
        self.is_hw_dec = is_yuv420 or is_yuv444

        # But YUV444 HW decode is supported for HEVC only
        if self.is_hw_dec and is_yuv444 and not is_hevc:
            self.is_hw_dec = False

        if self.is_hw_dec:
            # Nvdec supports NV12 (resampled YUV420) and YUV444 formats
            if self.ctx:
                self.nvDec = nvc.PyNvDecoder(
                    input=enc_file, context=self.ctx.handle, stream=self.stream.handle
                )
            else:
                self.nvDec = nvc.PyNvDecoder(
                    input=enc_file,
                    gpu_id=self.device_id,
                )
        else:
            assert False
            # # No HW decoding acceleration, fall back to CPU back-end.
            # self.to_gpu = nvc.PyFrameUploader(
            #     self.w, self.h, self.pix_fmt, self.device_id
            # )

        if is_yuv420:
            # YUV420 videos will be decoded by Nvdec to NV12 which is the
            # same thing but resampled (U and V planes interleaved).
            if self.ctx:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    nvc.PixelFormat.NV12,
                    nvc.PixelFormat.RGB,
                    self.ctx.handle,
                    self.stream.handle,
                )
            else:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    nvc.PixelFormat.NV12,
                    nvc.PixelFormat.RGB,
                    self.device_id,
                )
        elif is_yuv444:
            if self.ctx:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    self.pix_fmt,
                    nvc.PixelFormat.RGB,
                    self.ctx.handle,
                    self.stream.handle,
                )
            else:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    self.pix_fmt,
                    nvc.PixelFormat.RGB,
                    self.device_id,
                )
        else:
            if self.ctx:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    self.pix_fmt,
                    nvc.PixelFormat.RGB_PLANAR,
                    self.ctx.handle,
                    self.stream.handle,
                )
                self.cc_conv = nvc.PySurfaceColorconversion(
                    self.w,
                    self.h,
                    nvc.PixelFormat.YUV422,
                    self.ctx.handle,
                    self.stream.handle,
                )
            else:
                self.to_rgb = nvc.PySurfaceConverter(
                    self.w,
                    self.h,
                    self.pix_fmt,
                    nvc.PixelFormat.RGB_PLANAR,
                    self.device_id,
                )
                self.cc_conv = nvc.PySurfaceColorconversion(
                    self.w,
                    self.h,
                    nvc.PixelFormat.YUV422,
                    self.device_id,
                )

        if self.ctx:
            self.to_pln = nvc.PySurfaceConverter(
                self.w,
                self.h,
                nvc.PixelFormat.RGB,
                nvc.PixelFormat.RGB_PLANAR,
                self.ctx.handle,
                self.stream.handle,
            )
        else:
            self.to_pln = nvc.PySurfaceConverter(
                self.w,
                self.h,
                nvc.PixelFormat.RGB,
                nvc.PixelFormat.RGB_PLANAR,
                self.device_id,
            )

        if self.h != self.fixed_h:
            if self.ctx:
                self.to_sar = nvc.PySurfaceResizer(
                    self.fixed_w,
                    self.fixed_h,
                    nvc.PixelFormat.RGB_PLANAR,
                    self.ctx.handle,
                    self.stream.handle,
                )
            else:
                self.to_sar = nvc.PySurfaceResizer(
                    self.fixed_w,
                    self.fixed_h,
                    nvc.PixelFormat.RGB_PLANAR,
                    self.device_id,
                )
        else:
            self.to_sar = None

    def decode_hw(self, seek_ctx=None) -> nvc.Surface:
        """
        Decode single video frame with Nvdec, convert it to planar RGB.
        """
        # Decode with HW decoder
        if self.use_nvtx:
            nvtx.push_range("DecodeSingleSurface")
        if seek_ctx is None:
            dec_surface = self.nvDec.DecodeSingleSurface()
        else:
            dec_surface = self.nvDec.DecodeSingleSurface(seek_ctx)
        if not dec_surface or dec_surface.Empty():
            raise RuntimeError("Can not decode frame.")
        if self.use_nvtx:
            nvtx.pop_range()

        # Convert to packed RGB
        if self.use_nvtx:
            nvtx.push_range("to_rgb")
        rgb_int = self.to_rgb.Execute(dec_surface, self.cc_ctx)
        if not rgb_int or rgb_int.Empty():
            raise RuntimeError("Can not convert nv12 -> rgb.")
        if self.use_nvtx:
            nvtx.pop_range()

        # Convert to planar RGB
        if self.use_nvtx:
            nvtx.push_range("to_pln")
        rgb_pln = self.to_pln.Execute(rgb_int, self.cc_ctx)
        if not rgb_pln or rgb_pln.Empty():
            raise RuntimeError("Can not convert rgb -> rgb planar.")
        if self.use_nvtx:
            nvtx.pop_range()

        # Resize if necessary to maintain 1:1 SAR
        if self.use_nvtx:
            nvtx.push_range("to_sar")
        if self.to_sar:
            rgb_pln = self.to_sar.Execute(rgb_pln)
        if self.use_nvtx:
            nvtx.pop_range()

        return rgb_pln

    def decode_to_tensor(self, *args, **kwargs) -> torch.Tensor:
        """
        Decode single video frame, convert it to torch.cuda.FloatTensor.
        Image will be planar RGB normalized to range [0.0; 1.0].
        """
        if self.use_nvtx:
            nvtx.push_range("decode_to_tensor")
        dec_surface = None

        if self.is_hw_dec:
            dec_surface = self.decode_hw(*args, **kwargs)
        else:
            assert False

        if not dec_surface or dec_surface.Empty():
            raise RuntimeError("Can not convert rgb -> rgb planar.")

        if self.use_nvtx:
            nvtx.push_range("makefromDevicePtrUint8")
        surf_plane = dec_surface.PlanePtr()

        def __decode():
            img_tensor = pnvc.makefromDevicePtrUint8(
                surf_plane.GpuMem(),
                surf_plane.Width(),
                surf_plane.Height(),
                surf_plane.Pitch(),
                surf_plane.ElemSize(),
            )
            if self.use_nvtx:
                nvtx.pop_range()

            if img_tensor is None:
                raise RuntimeError("Can not export to tensor.")

            if self.use_nvtx:
                nvtx.push_range("resize_")

            # Convert from 3*H, W to 3, H, W
            new_t = img_tensor.view(3, int(surf_plane.Height() / 3), surf_plane.Width())

            if self.use_nvtx:
                nvtx.pop_range()
                nvtx.pop_range()

            return new_t

        if self.cuda_stream_pyt:
            with torch.cuda.stream(self.cuda_stream_pyt):
                return __decode()
        else:
            return __decode()


class cconverter:
    """
    Colorspace conversion chain.
    """

    def __init__(
        self,
        width: int,
        height: int,
        gpu_id: int,
        ctx,
        stream,
        use_nvtx: bool = False,
    ):
        self.device_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []
        self.use_nvtx = use_nvtx
        self.ctx = ctx
        self.stream = stream

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(
            nvc.PySurfaceConverter(
                width=self.w,
                height=self.h,
                src_format=src_fmt,
                dst_format=dst_fmt,
                context=self.ctx.handle,
                stream=self.stream.handle,
            )
            if self.ctx
            else nvc.PySurfaceConverter(
                width=self.w,
                height=self.h,
                src_format=src_fmt,
                dst_format=dst_fmt,
                gpu_id=self.device_id,
            )
        )

    def Execute(self, src_surface: nvc.Surface, cc) -> nvc.Surface:
        surf = src_surface
        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf


class nvencoder:
    def __init__(
        self,
        gpu_id: int,
        width: int,
        height: int,
        fps: float,
        enc_file: str,
        ctx,
        stream,
        use_nvtx: bool = False,
    ) -> None:
        """
        Create instance of HW-accelerated video encoder.
        :param gpu_id: id of video card which will be used for encoding & processing.
        :param width: encoded frame width.
        :param height: encoded frame height.
        :param enc_file: path to encoded video file.
        :param options: dictionary with encoder initialization options.
        """
        self.device_id = gpu_id
        self.ctx = ctx
        self.stream = stream
        self.use_nvtx = use_nvtx
        self.to_nv12 = cconverter(
            width, height, gpu_id=self.device_id, ctx=self.ctx, stream=self.stream
        )
        self.to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        self.to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        self.cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG
        )
        fps = round(Fraction(fps), 6)

        opts = {
            "preset": "P5",
            "tuning_info": "high_quality",
            "codec": "h264",
            "fps": str(fps),
            "s": str(width) + "x" + str(height),
            "bitrate": "10M",
        }

        self.gpu_id = gpu_id
        self.fps = fps
        self.enc_file = enc_file

        if self.ctx:
            self.nvEnc = nvc.PyNvEncoder(opts, self.ctx.handle, self.stream.handle)
        else:
            self.nvEnc = nvc.PyNvEncoder(opts, self.device_id)

        caps = self.nvEnc.Capabilities()
        print(caps)

        self.pts_time = 0
        self.delta_t = 1  # Increment the packets' timestamp by this much.
        self.encoded_frame = np.ndarray(shape=(0), dtype=np.uint8)
        self.container = av.open(enc_file, "w")
        self.avstream = self.container.add_stream("h264", rate=fps)
        self.avstream.width = width
        self.avstream.height = height
        self.avstream.time_base = 1 / Fraction(fps)  # 1/fps would be our scale.
        self.surface = None
        self.surf_plane = None

    def width(self) -> int:
        """
        Get video frame width.
        """
        return self.nvEnc.Width()

    def height(self) -> int:
        """
        Get video frame height.
        """
        return self.nvEnc.Height()

    def tensor_to_surface(self, img_tensor: torch.tensor) -> nvc.Surface:
        """
        Converts cuda float tensor to planar rgb surface.
        """
        if len(img_tensor.shape) != 3 and img_tensor.shape[0] != 3:
            raise RuntimeError("Shape of the tensor must be (3, height, width)")

        _, tensor_h, tensor_w = img_tensor.shape
        assert tensor_w == self.width() and tensor_h == self.height()

        if not self.surface:
            if self.ctx:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.RGB_PLANAR,
                    width=tensor_w,
                    height=tensor_h,
                    context=self.ctx.handle,
                )
            else:
                self.surface = nvc.Surface.Make(
                    format=nvc.PixelFormat.RGB_PLANAR,
                    width=tensor_w,
                    height=tensor_h,
                    gpu_id=self.device_id,
                )
            self.surf_plane = self.surface.PlanePtr()

        pnvc.TensorToDptr(
            img_tensor,
            self.surf_plane.GpuMem(),
            self.surf_plane.Width(),
            self.surf_plane.Height(),
            self.surf_plane.Pitch(),
            self.surf_plane.ElemSize(),
        )

        return self.surface

    def encode_from_tensor(self, tensor: torch.Tensor):
        """
        Encode single video frame from torch.cuda.FloatTensor.
        Tensor must have planar RGB format and be normalized to range [0.0; 1.0].
        Shape of the tensor must be (3, height, width).
        """
        assert tensor.dim() == 3
        assert self.device_id == tensor.device.index

        if self.use_nvtx:
            nvtx.push_range("tensor_to_surface")
        surface_rgb = self.tensor_to_surface(tensor)
        if self.use_nvtx:
            nvtx.pop_range()
            nvtx.push_range("to_nv12")

        dst_surface = self.to_nv12.Execute(surface_rgb, self.cc_ctx)
        if dst_surface.Empty():
            raise RuntimeError("Can not convert to yuv444.")
        if self.use_nvtx:
            nvtx.pop_range()

        if self.use_nvtx:
            nvtx.push_range("encoding_single_surface")
        success = self.nvEnc.EncodeSingleSurface(dst_surface, self.encoded_frame)
        if self.use_nvtx:
            nvtx.pop_range()
            nvtx.push_range("av_mux")

        if success:
            self.write_frame(
                self.encoded_frame,
                self.pts_time,
                self.fps,
                self.avstream,
                self.container,
            )
            self.pts_time += self.delta_t

        if self.use_nvtx:
            nvtx.pop_range()

    def write_frame(self, encoded_frame, pts_time, fps, stream, container):
        encoded_bytes = bytearray(encoded_frame)
        pkt = av.packet.Packet(encoded_bytes)
        pkt.pts = pts_time
        pkt.dts = pts_time
        pkt.stream = stream
        pkt.time_base = 1 / Fraction(fps)
        container.mux(pkt)

    def flush(self):
        if self.use_nvtx:
            nvtx.push_range("flush")
        packets = np.ndarray(shape=(0), dtype=np.uint8)
        success = self.nvEnc.Flush(packets)

        if success:
            self.write_frame(
                self.encoded_frame,
                self.pts_time,
                self.fps,
                self.avstream,
                self.container,
            )
            self.pts_time += self.delta_t

        if self.use_nvtx:
            nvtx.pop_range()
