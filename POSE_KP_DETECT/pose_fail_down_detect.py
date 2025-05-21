import os
import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from rknn.api import RKNN
import onnx

import onnxruntime as ort


def load_rknn_model_3588(model_path, target_platform="rk3588"):
    rknn = RKNN(verbose=True)
    print(f"--> loading RKNN mode: {model_path}")
    
    ret = rknn.load_rknn(model_path)
    if ret !=0 :
        raise RuntimeError(f"Failed to load RKNN model: {model_path}")
    ret = rknn.init_runtime(target=target_platform)
    if ret !=0 :
        raise RuntimeError("Failed to init RKNN runtime")
    
    print("RKNN model loaded and initialized. ")
    return rknn


if __name__ == '__main__':
    VIDEO_PATH = "FALL_DOWN_DETECT/ori.mp4"
    POSE_FILE_PATH = "/home/orangepi/Documents/pose_tracking_on_rknn/POSE_KP_DETECT/bbox_values.txt"
    ONNX_PATH = "FALL_DOWN_DETECT/alg_st-attention_model_fall_down24x17x3_240426.onnx"
    RKNN_PATH = "FALL_DOWN_DETECT/alg_st-attention_fall_down_3588.rknn"
    output_video_path="FALL_DOWN_DETECT/output.mp4"