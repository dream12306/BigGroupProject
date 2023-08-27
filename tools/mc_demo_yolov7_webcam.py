import argparse
import time
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import copyOfWebcam

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT

sys.path.insert(0, './yolov7')
sys.path.append('.')

boxes_tlwhs = []
boxes_ids = []
tp_ids = []
unconfirmed_tp_ids = []
unconfirmed_tp_ids_state = []
offset_x = 0
offset_y = 0
big_state = 1
zoom_factor = 1
size = (640, 480)
mode = 1  # everything:1,humanonly:0


def magnify(boxes_ids, boxes_tlwhs, tp_ids, Label, img):
    global zoom_factor
    global offset_x
    global offset_y
    global big_state
    if len(tp_ids) > 0:
        tp_id = tp_ids[0]
    else:
        return Label,img

    if len(tp_ids) == 1 and big_state and tp_id in boxes_ids:
        # 调整画面位置使目标对象始终在画面中央
        # 假设只追踪第一个目标对象
        index = boxes_ids.index(tp_id)
        tp_tlwh = boxes_tlwhs[index]
        x1, y1, w, h = tp_tlwh
        x2 = x1 + w
        y2 = y1 + h
        # 画布大小
        canvas_width = size[0]
        canvas_height = size[1]

        if y1 < 0:
            y1 = 0
            y2 = min(canvas_height, y2)
        if y2 > canvas_height:
            y2 = canvas_height
            y1 = max(0, y1)
        if x1 < 0:
            x1 = 0
            x2 = min(canvas_width, x2)
        if x2 > canvas_width:
            x2 = canvas_width
            x1 = max(0, x1)

        # 要放大的区域的中心坐标
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        # 画布的中心坐标
        canvas_center_x = canvas_width / 2
        canvas_center_y = canvas_height / 2
        # 要放大的区域的尺寸
        target_width = x2 - x1
        target_height = y2 - y1

        # 放大倍数
        zoom_factor = min(canvas_width / target_width, canvas_height / target_height)

        # 计算放大后的区域大小
        all_width = int(canvas_width * zoom_factor)
        all_height = int(canvas_height * zoom_factor)
        zoomed_width = int(target_width * zoom_factor)
        zoomed_height = int(target_height * zoom_factor)

        # 计算放大后的区域左上角坐标
        zoomed_x1 = int(canvas_center_x - zoomed_width / 2)
        zoomed_y1 = int(canvas_center_y - zoomed_height / 2)

        # 计算鼠标横纵坐标偏置
        offset_x = int(target_center_x * zoom_factor - canvas_width / 2)
        offset_y = int(target_center_y * zoom_factor - canvas_height / 2)

        # 裁剪图像，获取要放大的区域
        zoomed_img = img[int(y1):int(y2), int(x1):int(x2)]
        # 将整张图片放大，再从中截取目标区域
        img = cv2.resize(img, (all_width, all_height))

        image_y1 = int(target_center_y * zoom_factor - canvas_height / 2)
        image_y2 = int(target_center_y * zoom_factor + canvas_height / 2)
        image_x1 = int(target_center_x * zoom_factor - canvas_width / 2)
        image_x2 = int(target_center_x * zoom_factor + canvas_width / 2)
        # 越限坐标处理
        if image_y1 < 0:
            image_y1 = 0
            image_y2 = canvas_height
        if image_y2 > all_height:
            image_y2 = all_height
            image_y1 = all_height - canvas_height
        if image_x1 < 0:
            image_x1 = 0
            image_x2 = canvas_width
        if image_x2 > all_width:
            image_x2 = all_width
            image_x1 = all_width - canvas_width
        # 截取目标区域
        img = img[image_y1:image_y2, image_x1:image_x2]


    else:
        # 放大倍数
        zoom_factor = 1
        # 鼠标横纵坐标偏置
        offset_x = 0
        offset_y = 0

    return Label, img


def Mouse_Coordinate_Conversion(x,y):
    global big_state
    global zoom_factor
    global offset_x
    global offset_y
    x = (x + offset_x * big_state) / zoom_factor
    y = (y + offset_y * big_state) / zoom_factor
    return x, y


def update(opt, model, cap, device, half, tracker, imgsz, stride, count, names, colors, window, Label):
    pass


def on_click_left(event):
    pass


def on_click_right(event):
    pass


def track_all():
    pass


# 取消跟踪
def cancel_tracking():
    pass


# 取消放大
def cancel_zoom():
    pass


# 放大模式
def zoom():
    pass


def track_human():
    pass


def track_everything():
    pass


def detect():
    print('++++++++++++++++++')
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    opt.agnostic_nms = True
    # 初始化
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 加载模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    print(opt.with_reid)

    # 获取类名与颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # 创建追踪器
    tracker = BoTSORT(opt, frame_rate=60.0)

    # 开始运行
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    print('+++++++++--------------')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    window = tk.Tk()  # 创建Tkinter窗口
    Label = tk.Label(window)  # 创建画布
    Label.pack()
    button0 = tk.Button(window, text="Select All", command=track_all)  # 创建一个按钮，识别所有类别
    button0.pack()

    button1 = tk.Button(window, text="Cancel Tracking", command=cancel_tracking)  # 创建一个按钮，取消追踪
    button1.pack()

    button2 = tk.Button(window, text="Cancel Zoom Mode", command=cancel_zoom)  # 创建一个按钮，取消放大模式
    button2.pack()

    button3 = tk.Button(window, text="Zoom Mode", command=zoom)  # 创建一个按钮，开启放大模式
    button3.pack()

    button4 = tk.Button(window, text="Track Human Only", command=track_human)  # 创建一个按钮，只识别人类
    button4.pack()

    button5 = tk.Button(window, text="Track Everything", command=track_everything)  # 创建一个按钮，跟踪所有对象
    button5.pack()

    Label.bind("<Button-1>", on_click_left)  # 绑定鼠标左键点击事件到画布,并传递边界框
    Label.bind("<Button-3>", on_click_right)  # 绑定鼠标右键点击事件到画布,并传递边界框

    count = 0
    print(type(Label))
    print('update')
    update(opt,model,cap,device,half,tracker,imgsz,stride,count,names,colors,window,Label)
    window.mainloop()
    cap.release()  # 释放相机资源
    cv2.destroyAllWindows()  # 关闭所有窗口

    if cap:
        cap.release()
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='pretrained/yolov7x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')
    parser.add_argument('--save-results', default=False, action='store_true', help='save results')
    parser.add_argument('--mode', default="webcam", action='store_true', help='mode can be webcam')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=400, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=True, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str,
                        help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            print('-------------------------')
            detect()
