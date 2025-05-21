# hello_pose.py
import cv2
from ultralytics import YOLO
import time
import numpy as np

# rknn_model = YOLO("./yolo11n_rknn_model", task="detect")
# rknn_model = YOLO("./yolo11n-pose.pt")
# rknn_model = YOLO("./yolo11n-pose_ncnn_model")

rknn_model = YOLO("./yolo11n-pose_rknn_model")
# rknn_model = YOLO("./yolo11n_rknn_model")
# rknn_model = YOLO("./yolo11n-cls_rknn_model")

video_path = "./videos/ori.mp4"
# video_path = 'videos/faildown/fall_recognition_20210816_354.mp4'

def run_usb_cam(camera_index=0):
    cap = cv2.VideoCapture(video_path) 
    # cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("无法打开USB摄像头, 请检查video index")
    

    # --- 获取帧尺寸和FPS 用于保存 ---
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    # --- 初始化 VideoWriter ---
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可用 'XVID'
    # out = cv2.VideoWriter('output_inference_02.mp4', fourcc, fps_in, (width, height))
    
    
    frame_index = 0
    frame_interval = 5
    last_result = None
    
    # --- 新增: FPS 计算变量 ---
    prev_time   = time.time()
    frame_count = 0
    fps         = 0.0
    font        = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧。正在退出...")
            break

        frame = cv2.flip(frame, 1)
        frame_index += 1
        
        
        if frame_index % frame_interval == 0:
            results = rknn_model.predict(frame)
            # results = rknn_model.track(source=frame, tracker='./botsort_cus.yaml', conf=0.4, persist=True)
            
            # results = rknn_model.track(source=frame, tracker='./botsort_cus.yaml')
               
            # results = rknn_model.track(source=frame, tracker='./botsort_cus.yaml', persist=True)
            
            # results = rknn_model.track(source=frame, tracker='./bytetrack.yaml', persist=True, conf=0.2, iou=0.5)


            last_result = results[0]
            
            # print("---------------->res:\n", res)
            
        if last_result:
            annotated_frame = last_result.plot()
        else:
            annotated_frame = frame
        
        output_img = annotated_frame.copy().astype(np.uint8)
        
        
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
            output_img,
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

        cv2.imshow('usb CAM detect', output_img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    # out.release()  # ✨释放视频写入资源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_usb_cam()
    print()


