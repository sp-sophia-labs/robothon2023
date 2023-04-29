#!/usr/bin/env python
import cv2
import os
from statistics import mean
import rospy

from image_processing.image_processing import ImageProcessing
from board_detection.yolo_detection import YoloDetection


class DetectMulticlass(ImageProcessing):
    """DetectMulticlass"""
    def __init__(self):
        super(DetectMulticlass, self).__init__()
        self.dirname_path = os.path.dirname(os.path.abspath(__file__))
        self.YD = YoloDetection(weights= self.dirname_path + '/weights/best_multiclasses_model.pt', device='', img_size=640)

    def capture_points(self, show=False):


        img_cpt = 0
        img_cpt_lim = 4
        blue_x_list = []
        blue_y_list = []
        screen_x_list = []
        screen_y_list = []
        connector_x_list = []
        connector_y_list = []
        last_image = None

        while img_cpt<img_cpt_lim:

            rospy.logerr("Detection iteration : %s", img_cpt+1)

            raw_image = self.read_image()
            detection, image, xyxy, xywh, labels = self.YD.detect_in_stream(        
                                                                source=raw_image,
                                                                conf_thres=0.2,
                                                                iou_thres=0.45,
                                                            )
            contains_label = False
            found_1 = -1
            found_3 = -1
            found_5 = -1
            cpt = 0
            for label in labels:
                if label[0] == '1':
                    found_1 = cpt
                if label[0] == '3':
                    found_3 = cpt
                if label[0] == '5':
                    found_5 = cpt
                cpt += 1

            contains_label = found_1 >= 0 and found_5 >= 0 and found_3 >= 0
            if(not detection or not contains_label):
               continue

            img_cpt+=1
            if not img_cpt<img_cpt_lim:
                last_image = image

            blue_top_left_corner = (int(xyxy[found_1][0].item()), int(xyxy[found_1][1].item()))
            blue_bottom_right_corner = (int(xyxy[found_1][2].item()), int(xyxy[found_1][3].item()))
            blue_x_list.append(int((blue_top_left_corner[0] + blue_bottom_right_corner[0]) / 2))
            blue_y_list.append(int((blue_top_left_corner[1] + blue_bottom_right_corner[1]) / 2))

            connector_top_left_corner = (int(xyxy[found_5][0].item()), int(xyxy[found_5][1].item()))
            connector_bottom_right_corner = (int(xyxy[found_5][2].item()), int(xyxy[found_5][3].item()))
            connector_x_list.append(int((connector_top_left_corner[0] + connector_bottom_right_corner[0]) / 2))
            connector_y_list.append(int((connector_top_left_corner[1] + connector_bottom_right_corner[1]) / 2))

            screen_top_left_corner = (int(xyxy[found_3][0].item()), int(xyxy[found_3][1].item()))
            screen_bottom_right_corner = (int(xyxy[found_3][2].item()), int(xyxy[found_3][3].item()))
            screen_x_list.append(int((screen_top_left_corner[0] + screen_bottom_right_corner[0]) / 2))
            screen_y_list.append(int((screen_top_left_corner[1] + screen_bottom_right_corner[1]) / 2))

        blue_center_point = (int(mean(blue_x_list)), int(mean(blue_y_list)))
        screen_center_point = (int(mean(screen_x_list)), int(mean(screen_y_list)))
        connector_center_point = (int(mean(connector_x_list)), int(mean(connector_y_list)))
        cv2.circle(last_image, blue_center_point, 1, (255, 0, 0), 1)
        cv2.circle(last_image, screen_center_point, 1, (255, 0, 0), 1)
        cv2.circle(last_image, connector_center_point, 1, (255, 0, 0), 1)
        cv2.line(last_image, blue_center_point, screen_center_point, (255, 0, 0), 1)

        blue_screen_x_diff = (screen_center_point[0] - blue_center_point[0])
        blue_screen_y_diff = (screen_center_point[1] - blue_center_point[1])
        if blue_screen_x_diff == 0:
            print('vertical')
            fake_point_x = connector_center_point[0]
            fake_point_y = screen_center_point[1]
        elif blue_screen_y_diff == 0:
            print('horizontal')
            fake_point_y = connector_center_point[1]
            fake_point_x = screen_center_point[0]
        else:
            print('normal')
            blue_screen_tangent = blue_screen_y_diff / blue_screen_x_diff
            connector_origin = connector_center_point[1] - connector_center_point[0] * blue_screen_tangent
            orthogonal_tangent = -1 / blue_screen_tangent
            screen_origin = screen_center_point[1] - orthogonal_tangent * screen_center_point[0]
            fake_point_x = (screen_origin - connector_origin) / (blue_screen_tangent - orthogonal_tangent)
            fake_point_y = (orthogonal_tangent * fake_point_x + screen_origin)
        
        fake_point = (int(fake_point_x), int(fake_point_y))
        cv2.circle(last_image, fake_point, 1, (0, 255, 0), 1)
        cv2.line(last_image, connector_center_point, fake_point, (0, 255, 0), 1)
        cv2.line(last_image, screen_center_point, fake_point, (0, 0, 255), 1)
        if show:
            cv2.imshow("screen detection", last_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return ("", connector_center_point, fake_point)
            

