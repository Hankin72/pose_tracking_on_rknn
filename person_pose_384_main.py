from POSE_KP_DETECT.pose_detect_rknn import PoseRKNNModel, draw_pose
import cv2 
import os 
import time 


if __name__ == '__main__':
    # video_path = 'videos/faildown/fall_recognition_20210816_354.mp4'
    video_path = 'videos/faildown/fall_recognition_20210816_1210.mp4'
    # video_path = 'videos/faildown/fall_recognition_20210816_1210.mp4'
    
    model = PoseRKNNModel(model_path="POSE_KP_DETECT/person_pose640x384_3588.rknn")
    
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        
    fall_display_counter = 0
    fall_display_duration = 50
    last_kpts, last_box = None, None
    
    frame_count = 0

    # --- 新增: FPS 计算变量 ---
    pTime = 0
    cTime = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame = cv2.flip(frame, 1)
 
        boxes, img_letterbox, ori_image = model.predict(frame)
        annotated_frame = ori_image.copy()
        


        if len(boxes) > 0:
            annotated_frame = draw_pose(boxes, img_letterbox, annotated_frame)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime =  cTime
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)


        cv2.imshow('usb CAM detect', annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
