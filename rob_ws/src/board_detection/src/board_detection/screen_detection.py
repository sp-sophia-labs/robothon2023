#!/usr/bin/env python
import cv2
import os
import numpy
import rospy

from collections import OrderedDict
from image_processing.image_processing import ImageProcessing
from board_detection.yolo_detection import YoloDetection


class DetectScreen(ImageProcessing):
    """DetectScreen"""
    def __init__(self):
        super(DetectScreen, self).__init__()
        self.dirname_path = os.path.dirname(os.path.abspath(__file__))
        self.YD = YoloDetection(weights=self.dirname_path + '/weights/multiclasses_6.pt', device='', img_size=640)

    # Detect the led screen and return the crop of the bounding box
    def crop_screen(self, show=False):
        while True:
            raw_image = None
            try:
                raw_image = self.read_image()
                cv2.imwrite(self.dirname_path + '/images/image_' + str(rospy.Time().now().to_sec()) + '.jpg', raw_image)
            except:
                rospy.logerr('camera is not working properly')
                continue
            image = raw_image.copy()
            detection, image, xyxy, xywh, labels = self.YD.detect_in_stream(        
                                                                source = image,
                                                                conf_thres=0.2,
                                                                iou_thres=0.45,
                                                            )

            found_3 = -1
            cpt = 0
            for label in labels:
                if label[0] == '3':
                    found_3 = cpt
                cpt += 1

            if(not detection and found_3 >= 0):
                continue

            xyxy_3 = xyxy[found_3]
            top_left_corner = (int(xyxy_3[0].item()), int(xyxy_3[1].item()))
            bottom_right_corner = (int(xyxy_3[2].item()), int(xyxy_3[3].item()))
            try:
                crop = image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
            except:
                continue

            if show:
                cv2.imshow("crop", image)
                cv2.waitKey(0) 
                cv2.destroyAllWindows() 

            return crop

    # Adjust Brightness and contrast of the image
    # Brightness from 0 to 510
    # Contrast from 0 to 255
    def apply_brightness_contrast(self, raw_image, brightness=255, contrast=127, show=False):

        image = raw_image.copy()

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

        if contrast != 0:
            f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        if show:
            cv2.imshow("before treatment", raw_image)
            cv2.imshow("after treatment", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    # Detect the black bar that represent the slider and crop around it
    # Return slider size and cropped image
    def crop_slider(self, raw_image, show=False):

        image = raw_image.copy()
        frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        black_threshold = cv2.inRange(frame_HSV, (5, 0, 0), (150, 255, 140))
        black_opening = cv2.morphologyEx(black_threshold, cv2.MORPH_OPEN, kernel, iterations=3)
        
        if show:
            cv2.imshow("black threshold", black_opening)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        try:
            black_pixels = numpy.nonzero(black_opening)
            x_black_pixels = black_pixels[0]
            y_black_pixels = black_pixels[1]
        except:
            return image, 0

        #TODO optimize with numpy
        x_min = 1280
        x_max = 0
        for x in x_black_pixels:
            if x < x_min:
                x_min = x
                continue
            if x > x_max:
                x_max = x

        y_min = 1280
        y_max = 0
        for x in y_black_pixels:
            if x < y_min:
                y_min = x
                continue
            if x > y_max:
                y_max = x
    
        x_dist = abs(x_max - x_min)
        y_dist = abs(y_max - y_min)

        try: 
            x_min_revised = x_min-x_dist
            if(x_min_revised < 0):
                x_min_revised = 0
            image = image[x_min_revised:x_max+5, y_min:y_max]
        except:
            return image, 0

        # if show:
        #     rospy.logerr("%s", x_dist)
        #     cv2.imshow("slider crop", image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        #TODO arbitrary measures to change in case we change the resolution
        if (25 > x_dist > 5) and (130 > y_dist > 70):
            return image, y_dist

        return image, 0


    def detect_triangle(self, raw_image, show=False):

        # Crop the bright yellow pixels (triangles)
        frame_HSV = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        yellow_threshold = cv2.inRange(frame_HSV, (6, 0, 100), (120, 190, 255))
        yellow_opening = cv2.morphologyEx(yellow_threshold, cv2.MORPH_OPEN, kernel, iterations=3)

        if show:
            cv2.imshow("threshold", yellow_opening) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

        # Find the maximum from the selected pixels
        try:
            y_pixels = numpy.nonzero(yellow_opening)[0]
            x_pixels = numpy.nonzero(yellow_opening)[1]
        except:
            return []
        
        maximums = {}
        for index, x in enumerate(x_pixels):
            y = y_pixels[index]
            if x in list(maximums):
                if y > maximums[x]:
                    maximums[x] = y
            else:
                maximums[x] = y
        maximums = OrderedDict(sorted(maximums.items()))

        # Detect the local maximums (points of the triangles)
        local_max = []
        length = len(list(maximums))
        for index, x in enumerate(list(maximums)):
            if index==0 or index==length-1:
                continue

            try:
                y_before = maximums[x-1]
                y_after = maximums[x+1]
            except KeyError:
                continue

            if (maximums[x] >= y_before and maximums[x] > y_after) or (maximums[x] > y_before and maximums[x] >= y_after):
                local_max.append(x)

        # Path the non-pointy triangles (double pixel points)
        if(len(local_max) > 1):
            revised_local_max = []
            base_x = -1

            for index, x in enumerate(local_max):

                if base_x < 0:
                    if index == len(local_max) - 1:
                        revised_local_max.append(x)
                        break
                    if x != local_max[index + 1] -1:
                        revised_local_max.append(x)
                    else:
                        base_x = x

                else:
                    if index == len(local_max) - 1:
                        revised_local_max.append((base_x + x) / 2)
                        break
                    if x != local_max[index + 1] - 1:
                        revised_local_max.append((base_x + x) / 2)
                        base_x = -1

            local_max = revised_local_max

        return local_max
