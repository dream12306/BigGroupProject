# README——基于yolov7和BOTSORT的人体识别与追踪

## 小组信息

### 成员信息

#### 刘向前U202112570（组长），朱龙天 U202112587，陈雨奇U202112563，刘源浩U202112545。

### 组内分工

1. 刘向前：主程序detect()运行逻辑的设计，代码所需要的各个函数的设计，程序调试。
2. 刘源浩：主循环图像更新函数update()的实现。
3. 朱龙天：图像放大显示函数magnify()以及鼠标坐标转换函数的实现。
4. 陈雨奇： 选定追踪目标的事件处理函数，与程序运行状态相关的函数的实现。

## 基础环境

### python=3.9.17

### 使用的第三方库：

```
numpy==1.22.3
opencv-python==4.8.0.76
loguru==0.7.0
scikit-image==0.21.0
scikit-learn==1.3.0
tqdm=4.66.1
torch==2.0.0+cu118
torchaudio==2.0.1+cu118
torchvision==0.15.1+cu118
Pillow==9.3.0
thop==0.1.1.post2209072238
ninja==1.11.1
tabulate==0.9.0
tensorboard==2.14.0
tensorboard-data-server==0.7.1
lap==0.4.0
motmetrics==1.4.0
filterpy==1.4.5
h5py==3.9.0
matplotlib==3.7.2
scipy==1.11.2
prettytable==3.8.0
easydict==1.10
pyyaml==6.0.1
yacs==0.1.8
termcolor==2.3.0
gdown==4.7.1
onnx==1.8.1
onnxruntime==1.8.0
onnx-simplifier==0.3.5
faiss-gpu==1.72
```

## 运行方式

### 摄像头实时识别：

在“BoT-SORT-main/”文件夹内，在命令行窗口运行：

```
python tools/mc_demo_yolov7_camera.py
```

### 视频人体识别

在“BoT-SORT-main/”文件夹内，在命令行窗口运行：

```
python tools/mc_demo_yolov7.py --weights pretrained/yolov7x.pt --source 视频路径 --fuse-score --agnostic-nms --with-reid
```

如要识别其他类别对象，可再添加--class 0 1 等整数

## 程序相关
### 主要文件：tools/mc_demo_yolov7_camera.py

### 使用的函数：

```
# 自定义函数
1. magnify(boxes_ids, boxes_tlwhs, tp_ids, Label, img): 这个函数用于放大图像中的目标对象。
2. Mouse_Coordinate_Conversion(x, y): 这个函数用于将鼠标点击的坐标转换为放大后图像中的坐标。
3. update(opt, model, cap, device, half, tracker, imgsz, stride, count, names, colors, window, Label): 这个函数用于更新目标检测和跟踪的结果。它首先读取摄像头的图像，然后使用目标检测模型进行推理，得到检测结果。接着，它使用目标跟踪算法对检测结果进行跟踪，得到跟踪结果。最后，它根据跟踪结果，绘制边界框和标签，并将结果显示在窗口中。
4. on_click_left(event): 这个函数用于处理鼠标左键点击事件。当用户点击图像中的目标对象时，它将该目标对象添加到跟踪列表中。
5. on_click_right(event): 这个函数用于处理鼠标右键点击事件。当用户点击图像中的目标对象时，它将该目标对象从跟踪列表中移除。
6. track_all(): 这个函数用于将所有目标对象添加到跟踪列表中。
7. cancel_tracking(): 这个函数用于清空跟踪列表，取消所有目标对象的跟踪。
8. cancel_zoom(): 这个函数用于取消图像的放大模式。
9. zoom(): 这个函数用于开启图像的放大模式。
10. track_human(): 这个函数用于设置只跟踪人类目标对象。
11. track_everything(): 这个函数用于设置跟踪所有类型的目标对象。
12. detect(save_img=False): 这个函数是程序的主函数，用于初始化模型和设备，创建窗口和标签，启动摄像头，以及启动程序的主循环。


# 非自定义函数
1. set_logging(): 设置日志记录，用于记录程序运行过程中的信息。
2. select_device(opt.device): 选择运行设备，可以是CPU或者GPU。
3. attempt_load(weights, map_location=device): 尝试加载模型权重。
4. check_img_size(imgsz, s=stride): 检查图像尺寸，确保图像尺寸可以被模型的步长整除。
5. TracedModel(model, device, opt.img_size): 如果启用了模型追踪，将模型转换为追踪模型。
6. BoTSORT(opt, frame_rate=60.0): 创建一个BoTSORT跟踪器。
8. window.mainloop(): 启动Tkinter窗口的主循环。
9. cap.read(): 从摄像头读取一帧图像。
10. letterbox(img=new_shape=imgsz,stride=stride): 对图像进行尺寸调整和填充，以适应模型的输入尺寸。
11. torch.from_numpy(img).to(device): 将numpy数组转换为PyTorch张量，并将其移动到指定的设备上（CPU或GPU）。
12. model(img, augment=opt.augment): 使用模型对图像进行推理。
13.non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms): 对推理结果进行非最大抑制处理，去除重叠的边界框。
14. tracker.update(detections, im0): 使用跟踪器对检测结果进行跟踪。
15. plot_one_box(tlbr, im0, label=label, color=(0,255,0), line_thickness=2): 在图像上绘制一个边界框。
16. magnify(boxes_ids,boxes_tlwhs,tp_ids,Label,im0): 根据目标对象的位置和状态，对图像进行放大显示。
17. cv2.cvtColor(im0, cv2.COLOR_BGR2RGBA): 将图像从BGR颜色空间转换为RGBA颜色空间。
18. Image.fromarray(im0): 将numpy数组转换为PIL图像。
19. ImageTk.PhotoImage(image=img): 将PIL图像转换为Tkinter PhotoImage。

```

### 重要函数

#### detect()

```
# 无参数
# 运行逻辑：
1. 导入所需的库和模块。
2. 设置日志记录，用于记录程序运行过程中的信息。
3. 选择运行设备，可以是CPU或者GPU。
4. 加载模型权重。
5. 检查图像尺寸，确保图像尺寸可以被模型的步长整除。
6. 如果启用了模型追踪，将模型转换为追踪模型。
7. 创建一个BoTSORT跟踪器。
8. 打开摄像头，设置摄像头的参数。
9. 创建一个Tkinter窗口和标签，用于显示图像。
10. 创建一些按钮，用于用户交互。
11. 绑定鼠标点击事件到画布。
12. 调用update函数，首次更新目标检测和跟踪的结果。
13. 启动Tkinter窗口的主循环，等待用户的操作和事件。
14. 用户退出或无图像来源时，释放摄像头资源，关闭所有窗口。

```

#### **update**(*opt*,*model*,*cap*,*device*,*half*,*tracker*,*imgsz*,*stride*,*count*,*names*,*colors*,*window*,*Label*)

```
# 参数：
opt（程序选项），
model（模型），
cap（视频捕获对象），
device（设备），
half（是否使用半精度），
tracker（目标跟踪器），
imgsz（图像尺寸），
stride（模型的步长），
count（计数器），
names（类别名称），
colors（类别颜色），
window（Tkinter窗口对象），
Label（Tkinter标签对象）

# 运行逻辑：
1. 对一些全局变量进行声明
2. 对计数器count进行自增操作，并检查是否达到了每30帧打印一次的条件。
3. 读取视频帧，并检查是否成功读取。
4. 如果成功读取到视频帧，进行一系列的图像处理操作，包括尺寸调整、颜色转换和数据类型转换。
5. 使用模型对图像进行推理，得到预测结果。
6. 根据模式的不同，设置目标类别。
7. 对预测结果进行非最大抑制（NMS）处理，过滤掉重叠的边界框。
8. 处理检测结果，包括跟踪器的更新、目标的筛选和结果的保存。
9. 根据目标的状态和类别，绘制边界框和标签，并根据目标是否在tp_ids列表中，设置不同的颜色。
10. 处理跟踪器中未确认的目标，根据一定的条件将其添加到tp_ids列表中或更新其状态。
11. 更新全局变量boxes_tlwhs和boxes_ids，以便下一次循环使用。
12. 调用magnify函数对图像进行放大处理，并将结果显示在Tkinter窗口中。
13. 将图像转换为Tkinter可用的格式，并更新窗口中的图像。

```

## 视频识别测试结果见BoT-SORT-main/runs/detect/exp/palace_result.mp4
