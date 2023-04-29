from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np



class YoloMulticlasses():
    def __init__(self, weights='weights/best_multiclasses_model.pt', device='', trace = True, img_size=640):
        self.weights = weights
        print("using weights of model : ", self.weights)
        self.trace = trace
        self.img_size = img_size
        self.device = select_device(device)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)


    # out1 = cv2.VideoWriter('./labtest1_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (640,480))
    def detect(self,img,classes = None, conf_thres = 0.20, iou_thres = 0.45, augment = True, agnostic_nms = False):
        trace = True
        device = ''
        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, 640)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='best_multiclasses_model', n=2)  # initialize
            modelc.load_state_dict(torch.load(self.weights, map_location=device)['model']).to(device).eval()

        # Get names and colors
        
        # names = model.module.names if hasattr(model, 'module') else model.names
        names = ['Blue button', 'door handle', 'led screen', 'red button', 'red connector', 'slider'] 
        
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        # Letterbox
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                                    
                    if names[c] == 'Blue button':
                        print("Blue button detected with a cofiance of :", f'{conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_blue = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_blue = (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_blue), int(y_center_blue)), 1, (0, 255, 0), 1) # center of blue button
                        print("Center of blue button",x_center_blue, y_center_blue)
                        print("\n")
                        
                    if names[c] == 'door handle':
                        print("door handle detected with a cofiance of :", f'{conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_door = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_door= (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_door), int(y_center_door)), 1, (0, 255, 0), 1) # center of door handle
                        print("Center of door handle",x_center_door, y_center_door)                        
                        print("\n") 
                    
                    if names[c] == 'led screen':
                        print("led screen detected with a cofiance of :", f'{conf:.2f}') 
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_screen = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_screen  = (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_screen ), int(y_center_screen )), 1, (0, 255, 0), 1) # center of led screen
                        print("Center of led screen",x_center_screen , y_center_screen)                         
                        print("\n") 
                            
                    if names[c] == 'red button':
                        print("red button detected with a cofiance of :", f'{conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_red_button = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_red_button  = (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_red_button ), int(y_center_red_button )), 1, (0, 255, 0), 1) # center of red button
                        print("Center of red button",x_center_red_button , y_center_red_button)                         
                        print("\n") 
                        
                    if names[c] == 'red connector':
                        print("red connector detected with a cofiance of :", f'{conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_red_connector = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_red_connector  = (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_red_connector ), int(y_center_red_connector)), 1, (0, 255, 0), 1) # center of red connector
                        print("Center of red connector",x_center_red_connector , y_center_red_connector)                         
                        print("\n") 
                        
                    if names[c] == 'slider':
                        print("slider detected with a cofiance of :", f'{conf:.2f}')   
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             
                        x_center_slider = (int(xyxy[0]) + int(xyxy[2]))/2
                        y_center_slider  = (int(xyxy[1]) + int(xyxy[3]))/2
                        cv2.circle(im0, (int(x_center_slider ), int(y_center_slider)), 1, (0, 255, 0), 1) # center of slider
                        print("Center of slider",x_center_slider , y_center_slider)                         
                        print("\n") 
 
                        
            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
                        # Stream results
                    cv2.imshow("Recognition result", im0)
                    cv2.waitKey(0)
                results = ((x_center_blue, y_center_blue),(x_center_screen , y_center_screen),(x_center_red_button , y_center_red_button),(x_center_red_connector , y_center_red_connector),(x_center_slider , y_center_slider),(x_center_door, y_center_door))
                # results = ((x_center_blue, y_center_blue),(x_center_screen , y_center_screen),(x_center_red_button , y_center_red_button))

                return results     
