import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util
from .hFunction import HybridFunction


class Hybrid_model(nn.Module):
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(Hybrid_model, self).__init__()
        self.threshold = threshold
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True
        self.sp_x = None
        self.sp_y = None

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask = util.cal_feat_mask(mask_global, layer_to_last, threshold)
        self.mask = mask.squeeze() 
        return self.mask

    def forward(self, input):
        flip_feature = input[0,512:].unsqueeze(0)
        input= input[0,:512].unsqueeze(0)
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(0,0,1).data
            self.flag, self.nonmask_point_idx, self.flatten_offsets ,self.mask_point_idx= util.cal_mask_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, self.stride, self.mask_thred)
            self.cal_fixed_flag = False

        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            self.sp_x, self.sp_y = util.cal_sps_for_Advanced_Indexing(self.h, self.w)

        return HybridFunction.apply(input,flip_feature, self.mask, self.shift_sz, self.stride, self.triple_weight, self.flag, self.nonmask_point_idx, self.mask_point_idx, self.flatten_offsets, self.sp_x, self.sp_y)

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + 'threshold: ' + str(self.threshold) \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
