import cv2
from ultralytics import YOLO
import time

CURR_PATH = "/home/orangepi/Documents/pose_tracking_on_rknn/yolo"

# model = YOLO(CURR_PATH + "/models/yolov8n-pose.onnx")  # load a pretrained model (recommended for training)

# model = YOLO("yolo11n.pt")

# model.export(format='rknn', name='rk3588')

"""_summary_

WARNING ⚠️ Unable to automatically guess model task,
assuming 'task=detect'. 
Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.
"""

# rknn_model = YOLO("./yolo11n_rknn_model", task="detect")
rknn_model = YOLO("./yolo11n-pose_rknn_model")
# rknn_model = YOLO("./yolov8n-pose_rknn_model")


video_path = "./videos/ori.mp4"
# video_path = "./videos/faildown/fall_recognition_20210816_1210.mp4"


def run_usb_cam(camera_index=0):
    # cap = cv2.VideoCapture(video_path) 
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("无法打开USB摄像头, 请检查video index")
    
    
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
        
        # results = model.predict(frame)
        results = rknn_model.track(frame, tracker='bytetrack.yaml')
        res = results[0]
        
        annotated_frame = res.plot()
        
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

        cv2.imshow('usb CAM detect', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_usb_cam()
    print()


