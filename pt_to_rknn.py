
from ultralytics import YOLO


# please make sure to use x86 Linux machine when exporting to RKNN
# model = YOLO("yolo11n-pose.pt")

model = YOLO("yolo11n-seg.pt")

# model = YOLO("yolo11n.pt")
# model.export(format='rknn',name='rk3588', int8=Tru

model.export(format='rknn', name='rk3588')


# Export the model to NCNN format
# model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'