import cv2
from ultralytics import YOLO

CURR_PATH = "/home/orangepi/Documents/pose_tracking_on_rknn/yolo"

# model = YOLO(CURR_PATH + "/models/yolov8n-pose.onnx")  # load a pretrained model (recommended for training)

# model = YOLO("yolov8n-pose.pt")
# # model.export(format='rknn', name='rk3588')
# model.export(format='rknn', name='rk3588')

rknn_model = YOLO("./yolov8n-pose_rknn_model")



def run_usb_cam(camera_index=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("无法打开USB摄像头, 请检查video index")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧。正在退出...")
            break

        frame = cv2.flip(frame, 1)
        
        # results = model.predict(frame)
        results = rknn_model(frame)
        res = results[0]
        
        annotated_frame = res.plot()

        cv2.imshow('usb CAM detect', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_usb_cam(0)
    print()


