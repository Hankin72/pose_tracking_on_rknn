from rknn.api import RKNN
import cv2
import numpy as np

def load_rknn_model(model_path):
    rknn = RKNN(verbose=True)
    rknn.config(target_platform="rk3588")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load onnx model failed!')
        exit(ret)
    print('done')

    # build onnx
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    # ret = rknn.init_runtime(target=args.target, device_id=args.device_id)
    ret = rknn.init_runtime()

    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    return rknn


def convert_onnx_2_rknn_person_pose():
    rknn = load_rknn_model('person_pose640x384.onnx')

    # 导出 RKNN 模型
    print('Exporting RKNN model...')
    ret = rknn.export_rknn('person_pose640x384_3588.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('Conversion successful! The RKNN model is saved as person_pose640x384_3588.rknn.')


if __name__ == "__main__":
    print()
    convert_onnx_2_rknn_person_pose()
