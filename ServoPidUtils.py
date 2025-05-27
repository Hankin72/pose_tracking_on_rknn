# ServoPidUtils.py

from pcb.RobotController import RobotController
import time

PAN_CHANNEL, TILT_CHANNEL = 2, 1 # 水平方向舵机, # 垂直方向舵机
pan_angle, tilt_angle = 90, 60  # 水平方向舵机初始化角度, # 垂直方向舵机初始化角度


# 误差死区，防止舵机微小抖动
DEAD_ZONE = 50

servoCar = RobotController()

# PID 控制器参数
dt = 0.1
class PID:
    def __init__(self, kp, ki, kd, output_limit=5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # 限制输出最大变化幅度
        output = max(-self.output_limit, min(self.output_limit, output))
        return output
    
    
pid_pan = PID(0.015, 0.01, 0.0008)
pid_tilt = PID(0.025, 0.01, 0.0083)

def angle_to_pulse(angle):
    return int(600 + (angle / 180.0) * (2400 - 600))

def reset_servo_position():
    global pan_angle, tilt_angle
    pan_angle, tilt_angle = 90, 60
    servoCar.set_servo(PAN_CHANNEL, pan_angle)
    servoCar.set_servo(TILT_CHANNEL, tilt_angle)
    
    pid_pan.prev_error  = 0
    pid_pan.integral    = 0
    pid_tilt.prev_error = 0 
    pid_tilt.integral   = 0
    time.sleep(0.2)  # 舵机回中后，等待1秒再开始 PID 控制                    if frame_index > 10:


def update_servo_position(face_x, face_y, center_x, center_y):
    global pan_angle, tilt_angle
    dx = face_x - center_x
    dy = face_y - center_y

    if abs(dx) < DEAD_ZONE: dx = 0
    if abs(dy) < DEAD_ZONE: dy = 0

    pan_angle += pid_pan.update(dx)
    tilt_angle += pid_tilt.update(dy)

    pan_angle = max(0, min(180, pan_angle))
    tilt_angle = max(0, min(180, tilt_angle))

    servoCar.set_servo(PAN_CHANNEL, pan_angle)
    servoCar.set_servo(TILT_CHANNEL, tilt_angle)
    time.sleep(0.1)
    

if __name__ == "__main__":
    reset_servo_position()