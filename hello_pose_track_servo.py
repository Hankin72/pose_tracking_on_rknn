# hello_pose.py
import cv2
from ultralytics import YOLO
import time
import numpy as np
from ServoPidUtils import update_servo_position, reset_servo_position

rknn_model = YOLO("./yolo11n-pose_ncnn_model")

TARGET_ID = 1  # 目前 测试 跟踪 ID=1的pose

video_path = "./videos/ori.mp4"
# video_path = "./videos/faildown/fall_recognition_20210816_1210.mp4"

def run_usb_cam(camera_index=0):
    # cap = cv2.VideoCapture(video_path) 
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("无法打开USB摄像头, 请检查video index")
    

    # --- 获取帧尺寸和FPS 用于保存 --- 
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    # --- 初始化 VideoWriter ---
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可用 'XVID'
    # out = cv2.VideoWriter('output_inference_02.mp4', fourcc, fps_in, (width, height))
    center_x = width / 2
    center_y =  height / 2
    
    frame_index = 0
    frame_interval = 5
    last_result = None
    
    # --- 新增: FPS 计算变量 ---
    prev_time   = time.time()
    frame_count = 0
    fps         = 0.0
    font        = cv2.FONT_HERSHEY_SIMPLEX
    
    reset_servo_position()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧。正在退出...")
            break

        frame = cv2.flip(frame, 1)
        frame_index += 1
        
        
        if frame_index % frame_interval == 0:
            results = rknn_model.track(
                source=frame, 
                tracker='./bytetrack.yaml', 
                persist=True, 
                conf=0.2
                )
            # results = rknn_model.track(
            #     source=frame, 
            #     tracker='./botsort_cus.yaml', 
            #     persist=True, 
            #     conf=0.2, iou=0.3
            #    )botsort_cus.yaml

            last_result = results[0] if results else None
            
            # print("---------------->res:\n", res)
            # botsort_cus.yaml
        if last_result:
            annotated_frame = last_result.plot()
        else:
            annotated_frame = frame
        
        cv2.circle(annotated_frame, (int(center_x), int(center_y)), 6, (0,0,255), -1) # red
            
        # 绘制 + 处理舵机跟踪
        if last_result and last_result.keypoints is not None and last_result.boxes.id is not None:
            boxes = last_result.boxes
            ids = boxes.id.cpu().numpy().astype(int)
            keypoints = last_result.keypoints.data.cpu().numpy()
            
            for i, track_id in enumerate(ids):
                if track_id == TARGET_ID:
                    kp = keypoints[i]
                    
                    face_x, face_y = kp[0][0], kp[0][1]
                    
                    # 判断左右肩膀是否有效
                    # ls = kp[5]
                    # rs = kp[6]
                    # valid_ls = ls[0] > 0 and ls[1] > 0
                    # valid_rs = rs[0] > 0 and rs[1] > 0
                    
                    # if valid_ls and valid_rs:
                    #     face_x = (ls[0] + rs[0]) / 2
                    #     face_y = (ls[1] + rs[1]) / 2
                    # else:
                    #     nose = kp[0]
                    #     face_x, face_y = nose[0], nose[1]
                        
                    
                    update_servo_position(face_x, face_y, center_x, center_y)
                    
                    cv2.circle(annotated_frame, (int(face_x), int(face_y)), 6, (0, 255, 255), -1)
                    
                    cv2.line(annotated_frame, (int(center_x), int(center_y)), (int(face_x), int(face_y)), (0, 255, 0), 2)
                    
        
        # output_img = annotated_frame.copy().astype(np.uint8)
        
        
        # --- 新增: 统计 FPS ---
        frame_count += 1
        curr_time = time.time()
        elapsed   = curr_time - prev_time
        if elapsed >= 1.0:               # 每秒刷新一次 FPS
            fps = frame_count / elapsed
            frame_count = 0
            prev_time   = curr_time

        # --- 新增: 在左上角绘制 FPS ---
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            font,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        # --- 写入视频文件 ---
        # out.write(output_img)

        cv2.imshow('usb CAM detect', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    # out.release()  # ✨释放视频写入资源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_usb_cam()
    print()


