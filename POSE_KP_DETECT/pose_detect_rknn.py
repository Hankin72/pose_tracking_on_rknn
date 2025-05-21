# pose_detect_rknn.py

import os
import cv2
import numpy as np
from rknn.api import RKNN


# -------------------- 模型加载 --------------------
def load_rknn_model_3588(model_path, target_platform="rk3588"):
    rknn = RKNN(verbose=True)
    print(f"--> loading RKNN mode: {model_path}")
    
    ret = rknn.load_rknn(model_path)
    if ret !=0 :
        raise RuntimeError(f"Failed to load RKNN model: {model_path}")
    ret = rknn.init_runtime(target=target_platform)
    if ret !=0 :
        raise RuntimeError("Failed to init RKNN runtime")
    
    print("RKNN model loaded and initialized. ")
    return rknn

# -------------------- 图像预处理 -------------------- 调整图像大小和两边灰条填充
def letterbox(im, new_shape=(384, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加灰条
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def pre_process(img):
    # 归一化 调整通道为（1，3，640，640）
    img = img / 255.
    # img = np.transpose(img, (2, 0, 1))
    data = np.expand_dims(img, axis=0)
    return data


def preprocess_image(ori_image, target_size=(640, 384)):
    img = letterbox(ori_image)
    data = pre_process(img)

    return data, img, ori_image


# -------------------- 坐标变换工具 --------------------
def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(a):
    ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
    b = np.copy(a)
    # y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    # y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    b[:, 2] = a[:, 2] - a[:, 0]  # w
    b[:, 3] = a[:, 3] - a[:, 1]  # h
    return b

# nms算法
# dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
def nms(dets, iou_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)

def write_bbox_to_txt(bbox, file_path, image):
    H, W, channels = image.shape
    with open(file_path, 'a') as f:
        for row in bbox:
            # 将每行数据转换为字符串，用空格分隔
            row_str = ' '.join(map(str, row))  # 每个数据转换为字符串，并用空格分隔

            # 将字符串按空格分割成列表
            elements = row_str.split()
            # 取第2-5个元素
            second_to_fifth = []
            origin = elements[0:4]
            # 处理
            second_to_fifth.append((float(origin[0]) + float(origin[2])) / (2 * W))
            second_to_fifth.append((float(origin[1]) + float(origin[3])) / (2*H))
            second_to_fifth.append((float(origin[2]) - float(origin[0])) / W)
            second_to_fifth.append((float(origin[3]) - float(origin[1])) / H)
            # 取第7个元素到最后，按每组三个元素处理
            rest_elements = elements[5:]
            # 每三元素为一组，按要求进行处理
            processed_rest = []
            for i in range(0, len(rest_elements), 3):
                group = rest_elements[i:i + 3]
                X = (float(group[0]) / W)
                Y = float(group[1]) / H
                group[0] = 0 if float(group[2]) < 0.5 else X  # 第1个元素除以W
                group[1] = 0 if float(group[2]) < 0.5 else Y  # 第2个元素除以H
                processed_rest.extend(group)
            # 合并第2-5个元素处理后的结果和第7个开始的元素处理后的结果
            row_str = second_to_fifth + processed_rest
            row_float = [float(x) for x in row_str]

            # f.write(f"{row_float}\n")  # 写入每行数据到文件
            row_str2 = ' '.join(map(str, row_float))
            row_str2 = '0 ' + row_str2
            f.write(f"{row_str2}\n")
            
def scale_boxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''
    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    num_kpts = boxes.shape[1] // 3   # 56 // 3 = 18
    for kid in range(2,num_kpts+1):
        boxes[:, kid * 3-1] = (boxes[:, kid * 3-1] - pad[0]) / gain
        boxes[:, kid * 3 ]  = (boxes[:, kid * 3 ] -  pad[1]) / gain
    # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x      #左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  #右下
    boxes[:, 3] = bottom_right_y
    
# 调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
					[230, 230, 0], [255, 153, 255], [153, 204, 255],
					[255, 102, 255], [255, 51, 255], [102, 178, 255],
					[51, 153, 255], [255, 153, 153], [255, 102, 102],
					[255, 51, 51], [153, 255, 153], [102, 255, 102],
					[51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
					[255, 255, 255]])
# 17个关键点连接顺序
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
			[7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
			[1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# 骨架颜色
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# 关键点颜色
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 =17
    # 画点
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:   # 关键点的置信度必须大于 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 6, (int(r), int(g), int(b)), -1)
    # 画骨架
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1 >0.5 and conf2 >0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=1)
            
# 画出关键点
def show_rect_image(bboxs, img, ori_image):
    # 坐标从左上点，右下点 到 左上点，宽，高.
    bboxs = np.array(bboxs)
    bboxs = xyxy2xywh(bboxs)

    # oriimg = cv2.imread('bus.jpg')
    # 坐标点还原到原图
    bboxs = scale_boxes(img.shape, bboxs, ori_image.shape)

    # 画框 画点 画骨架
    for box in bboxs:
        # 依次为 检测框（左上点，右下点）、置信度、17个关键点
        det_bbox, det_scores, kpts = box[0:4], box[4], box[5:]
        # 画框
        cv2.rectangle(ori_image, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])),
                      (0, 0, 255), 1)
        # 人体检测置信度
        if int(det_bbox[1]) < 30:
            cv2.putText(ori_image, "conf:{:.2f}".format(det_scores),
                        (int(det_bbox[0]) + 5, int(det_bbox[1]) + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
        else:
            cv2.putText(ori_image, "conf:{:.2f}".format(det_scores),
                        (int(det_bbox[0]) + 5, int(det_bbox[1]) - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
        # 画点 连线
        plot_skeleton_kpts(ori_image, kpts)
    return ori_image  # 返回处理后图像


class PoseResult:
    def __init__(self, box, img, ori):
        self.box = box
        self.img = img
        self.ori = ori

    def plot(self):
        return show_rect_image([self.box], self.img, self.ori)
    
# -------------------- 核心推理封装 --------------------
class PoseRKNNModel:
    def __init__(self, model_path):
        self.rknn = load_rknn_model_3588(model_path)

    def predict(self, frame, conf_thresh=0.1, iou_thresh=0.4):
        if isinstance(frame,str):
            frame = cv2.imread(frame)
            if frame is None:
                raise ValueError(f"Image file not found or invalid: {frame}")
        
        data, img_letterbox, ori_image = preprocess_image(frame)
    
        frame_resized = cv2.resize(frame, (640, 384))
        
        input_tensor = np.expand_dims(frame_resized / 255.0, axis=0).astype(np.float32)

        pred = self.rknn.inference(inputs=[input_tensor])[0]
        
        pred = np.squeeze(pred, axis=0)
        pred = np.transpose(pred, (1, 0))
        pred = pred[pred[:, 4] > conf_thresh]

        results = []
        if len(pred) > 0:
            bboxs = xywh2xyxy(pred)
            bboxs = nms(bboxs, iou_thresh)
            
            for box in bboxs:
                results.append(PoseResult(box=box, img=img_letterbox, ori=ori_image))
        return results


if __name__ == '__main__':
    rknn_model_path = "POSE_KP_DETECT/person_pose640x384_3588.rknn"

    video_path = 'videos/faildown/fall_recognition_20210816_354.mp4'
    # video_path = 'POSE_KP_DETECT/ori.mp4'
    
    model = PoseRKNNModel(model_path=rknn_model_path)
    
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        results = model.predict(frame)
    
        # print(results[0])
        
        annotated_frame = results[0].plot()
        
        cv2.imshow('usb CAM detect', annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    # out.release()  # ✨释放视频写入资源
    cv2.destroyAllWindows()
