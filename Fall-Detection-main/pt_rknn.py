
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# model = YOLO("yolo11n.pt")
# model.export(format='rknn',name='rk3588', int8=Tru

model.export(format='rknn', name='rk3588')


# Export the model to NCNN format
# model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'