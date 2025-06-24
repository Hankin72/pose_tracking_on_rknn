from yolov8_pose import Yolov8PoseRKNN
import cv2 
import time 

pose_model = Yolov8PoseRKNN(model_path="./yolov8n-pose_int8.rknn", target="rk3588", verbose=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("无法打开USB摄像头, 请检查video index")

# --- 新增: FPS 计算变量 ---
pTime = 0
cTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧。正在退出...")
        break

    frame = cv2.flip(frame, 1)
    
    start_time = time.time()
    
    boxes, drawed_image = pose_model.infer(frame, need_draw=True)
    
    inference_time = (time.time() - start_time) * 1000
    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    
    pTime =  cTime
    
    print(f"Found {len(boxes)} person, Inference time: {inference_time:.4f}ms, FPS: {fps:.2f}")
    
    cv2.putText(drawed_image, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('usb CAM detect', drawed_image)

    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pose_model.release()