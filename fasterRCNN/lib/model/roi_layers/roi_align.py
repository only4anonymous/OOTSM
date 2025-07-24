# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torchvision.ops import RoIAlign
from fasterRCNN.lib.model import _C

import pdb
import torch.distributed as dist

def check_cuda_memory(device='cuda'):
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    print(f"[{device}] 当前已分配内存: {allocated / 1024**2:.2f} MB")
    print(f"[{device}] 当前保留内存: {reserved / 1024**2:.2f} MB")
    print(f"[{device}] 最大已分配内存: {max_allocated / 1024**2:.2f} MB")
    print(f"[{device}] 最大保留内存: {max_reserved / 1024**2:.2f} MB")

class _ROIAlign(Function):
	@staticmethod
	def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
		ctx.save_for_backward(roi)
		ctx.output_size = _pair(output_size)
		ctx.spatial_scale = spatial_scale
		ctx.sampling_ratio = sampling_ratio
		ctx.input_shape = input.size()
		#在local rank 1上打印一些信息
		# if dist.get_rank() == 1:
		# 	print("################")
		# 	check_cuda_memory(device='cuda:1')
		# 	print("################")
		# 检查 RoIs 是否在图像范围内
		# image_width = input.size(3)
		# image_height = input.size(2)
		# roi[:, 1] = torch.clamp(roi[:, 1], min=0, max=image_width - 1)  # 左坐标
		# roi[:, 3] = torch.clamp(roi[:, 3], min=0, max=image_width - 1)  # 右坐标
		# roi[:, 2] = torch.clamp(roi[:, 2], min=0, max=image_height - 1)  # 上坐标
		# roi[:, 4] = torch.clamp(roi[:, 4], min=0, max=image_height - 1)  # 下坐标
		# assert (roi[:, 1:] >= 0).all(), "RoIs 有负值，可能越界"
		# assert (roi[:, 3] < image_width).all(), "RoIs 超出图像宽度"
		# assert (roi[:, 4] < image_height).all(), "RoIs 超出图像高度"
		roi_align_layer = RoIAlign(output_size=output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
		output = roi_align_layer(input, roi)
		# output = _C.roi_align_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio)
		return output
	
	@staticmethod
	@once_differentiable
	def backward(ctx, grad_output):
		rois, = ctx.saved_tensors
		output_size = ctx.output_size
		spatial_scale = ctx.spatial_scale
		sampling_ratio = ctx.sampling_ratio
		bs, ch, h, w = ctx.input_shape
		grad_input = _C.roi_align_backward(
			grad_output,
			rois,
			spatial_scale,
			output_size[0],
			output_size[1],
			bs,
			ch,
			h,
			w,
			sampling_ratio,
		)
		return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
	def __init__(self, output_size, spatial_scale, sampling_ratio):
		super(ROIAlign, self).__init__()
		self.output_size = output_size
		self.spatial_scale = spatial_scale
		self.sampling_ratio = sampling_ratio
	
	def forward(self, input, rois):
		return roi_align(
			input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
		)
	
	def __repr__(self):
		tmpstr = self.__class__.__name__ + "("
		tmpstr += "output_size=" + str(self.output_size)
		tmpstr += ", spatial_scale=" + str(self.spatial_scale)
		tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
		tmpstr += ")"
		return tmpstr
