import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    
    pTime =  cTime
    
    cv2.putText(frame, f'FPS: {int(fps)}', (40, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('BlazePose', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# out.release()  # ✨释放视频写入资源
cv2.destroyAllWindows()
