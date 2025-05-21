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


VIDEO_PATH = "FALL_DOWN_DETECT/ori.mp4"
# VIDEO_PATH = "/home/10177178@zte.intra/Desktop/摔倒数据集/fall_recognition_20210816_354.mp4"
# POSE_FILE_PATH = "/home/10177178@zte.intra/文档/多模态相关/4-任务/跌倒检测/用官方的onnx人形检测模型/人形检测/bbox_values.txt"
POSE_FILE_PATH = "/home/orangepi/Documents/pose_tracking_on_rknn/POSE_KP_DETECT/bbox_values.txt"
# POSE_FILE_PATH="/home/10177178@zte.intra/下载/image0.txt"     #关键点检测生成的labels
ONNX_PATH = "FALL_DOWN_DETECT/alg_st-attention_model_fall_down24x17x3_240426.onnx"
RKNN_PATH = "FALL_DOWN_DETECT/alg_st-attention_fall_down_3588.rknn"
output_video_path="FALL_DOWN_DETECT/output.mp4"


skip_frames = 2
infer_frames = 24

# 检查视频是否成功打开
video = cv.VideoCapture(VIDEO_PATH)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频属性
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv.CAP_PROP_FPS))

# 创建输出视频文件
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Create RKNN object
rknn = load_rknn_model_3588(model_path=RKNN_PATH)

kpts_frames = []
with open(POSE_FILE_PATH, "r") as file:
    for i, line in enumerate(file, start=1):
        data = line.strip()  # 去除首尾的空白字符（包括换行符）
        print(f"第{i}行是{data}")

#for i in range(len(pose_list)):
        ret, frame = video.read()

        if i % skip_frames == 0:
            data_line = data.strip()
            print(f"data_line:{data_line}")
            box = data_line.split(' ')[1:5]
            print(f"box:{box}")
            box = [float(box[i]) for i in range(len(box))]
            kpts = data_line.split(' ')[5:]
            kpts = [float(kpts[i]) for i in range(len(kpts))]
            box_x_min = box[0] - box[2] / 2
            box_y_min = box[1] - box[3] / 2
            scale = max(box[2], box[3])
            kpts = np.array(kpts,dtype=np.float32)
            kpts = kpts.reshape(17, 3)

            # 归一化
            kpts[:,0] = (kpts[:,0] - box_x_min) / scale + (0.5 - box[2] / scale / 2)
            kpts[:,1] = (kpts[:,1] - box_y_min) / scale

            if len(kpts_frames) < infer_frames:
                kpts_frames.append(kpts)
            else:
                kpts_frames[0:-1] = kpts_frames[1:]
                kpts_frames[-1] = kpts

            if len(kpts_frames) == infer_frames:
                input_data = np.array(kpts_frames)
                input_data = np.expand_dims(input_data, axis=0)
                input_data_dict = {
                    'input_tensor': input_data
                }
                outputs_name = ['cls_score']

                # 运行模型推理
                # outputs = session.run(outputs_name, input_data_dict)

                input_data = np.transpose(input_data, (0, 2, 3, 1))
                outputs = rknn.inference(inputs=[input_data.astype(np.float32)])

                print('outputs')
                print(outputs)
                for output in outputs:
                    exp_output = np.exp(output)
                    probabilities = exp_output / np.sum(exp_output, axis=1)

                    # 求出预测的类别
                    predicted_class = np.argmax(probabilities, axis=1)
                    if predicted_class == 0:
                        show_txt = "normal:"+str(probabilities[0,predicted_class])
                        color = (255, 255, 255)
                    else:
                        show_txt = "fall:"+str(probabilities[0,predicted_class])
                        color = (0, 0, 255)  # 红色表示跌倒警报
                    print(show_txt)

                    cv.putText(frame, show_txt, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
                    # 写入输出视频
                    out.write(frame)

# 释放视频对象
video.release()
out.release()
