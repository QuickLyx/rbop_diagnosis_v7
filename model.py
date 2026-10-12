# -*- coding: utf-8 -*-
"""只读复用 v6 网络与损失，不改变网络或默认损失实现。"""
from _v6 import load_v6

_v6model = load_v6("model")
MSRSABiLSTM = _v6model.MSRSABiLSTM
SMCBlock = _v6model.SMCBlock
ConvFrontEnd = _v6model.ConvFrontEnd
ResidualShrinkageUnit = _v6model.ResidualShrinkageUnit
AdditiveAttention = _v6model.AdditiveAttention
gaussian_kernel = _v6model.gaussian_kernel
mmd_loss = _v6model.mmd_loss
classwise_mmd_loss = _v6model.classwise_mmd_loss

__all__ = ["MSRSABiLSTM", "SMCBlock", "ConvFrontEnd", "ResidualShrinkageUnit",
           "AdditiveAttention", "gaussian_kernel", "mmd_loss", "classwise_mmd_loss"]
