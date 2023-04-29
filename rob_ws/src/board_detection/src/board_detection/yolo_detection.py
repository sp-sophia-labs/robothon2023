import torch
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel

class YoloDetection():
    # This code is not clean but is functional, input ndarray image and path to your weights and returns image with bounding box, xyxy coordinated of the box corners, xywh of the box, and label of detected classe
    def __init__(self, weights='yolov7.pt', device='', trace = False, img_size=640):
        self.weights = weights
        print("using weights of model : ", self.weights)
        self.trace = trace
        self.img_size = img_size
        self.device = select_device(device)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        
    def detect_in_stream(
            self,
            source,
            classes=None,
            conf_thres=0.25,
            iou_thres=0.45,
            augment=True,
            agnostic_nms=False,
            ):
        # Initialize
        set_logging()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA
        imgsz = self.img_size

        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        if isinstance(source, np.ndarray):
            # Padded resize
            img0 = source
            img = letterbox(source, self.img_size, stride=stride)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, img0)
        
        s = ''
        xyxy_list = []
        xywh_list = []
        label_list = []
        # Process detections
        for _ , det in enumerate(pred):  # detections per image
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                detection = True
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                    
                    xyxy_list.append([x.cpu().numpy() for x in xyxy])
                    xywh_list.append(xywh)
                    label_list.append(label)

                return detection, img0, xyxy_list, xywh_list, label_list
            
            else:
                detection = False
                return detection, img0, xyxy_list, xywh_list, label_list

