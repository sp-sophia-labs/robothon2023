from image_processing.image_processing import ImageProcessing
from board_detection.real_time_detect import YoloMulticlasses
from board_detection.real_time_detect import *
import cv2 
import math
import os


class MyDetection(ImageProcessing):
    def __init__(self):
        super(MyDetection, self).__init__()
        self.dirname_path = os.path.dirname(os.path.abspath(__file__))
        self.YoloMulticlasses = YoloMulticlasses(weights=self.dirname_path + 'weights/best_multiclasses_model.pt', device='', img_size=640)
        
    def capture_points(self):
        #im0 = self.read_image() 
        im0 = cv2.imread('box3.jpg')   
        result = self.YoloMulticlasses.detect(im0, classes = None, conf_thres = 0.20, iou_thres = 0.45, augment = True, agnostic_nms = False)
        blue_button   = result[0][0], result[0][1] # position of blue button in pixels
        led_screen    = result[1][0], result[1][1] # position of led screen in pixels
        red_button    = result[2][0], result[2][1] # position of red button in pixels
        red_connector = result[3][0], result[3][1] # position of red connector in pixels
        slider        = result[4][0], result[4][1] # position of slider in pixels
        door_handle   = result[5][0], result[5][1] # position of door handle in pixels       
        m = ( red_button[1] - blue_button[1] ) / ( red_button[0] - blue_button[0] )
        p = blue_button[1] - m * blue_button[0]
        print("Equations of a Straight Line (blueButton_redButton) : y = {} x + {}".format(m,p), "\n") # Straight line through blue button and red button
        verif_1 = m*led_screen[0] + p
        verif_2 = led_screen[1]
        print("alignment condition", abs(verif_1 - verif_2), "\n")
        if abs(verif_1 - verif_2) <= 2: # alignment condition
            print("aligned points, constitution of the frame with the blue and red button", "\n")
            return (blue_button, red_button)        
        else :
            pose_blue_button = self.pose_from_pixels(blue_button) # position of blue button in camera frame  
            pose_red_button = self.pose_from_pixels(red_button) # position of red button in camera frame               
            pose_led_screen = self.pose_from_pixels(led_screen) # position of led screen in camera frame              
            pose_red_connector = self.pose_from_pixels(red_connector) # position of red connector in camera frame              
            pose_slider = self.pose_from_pixels(slider) # position of slider in camera frame               
            pose_door_handle = self.pose_from_pixels(door_handle) # position of door handle in camera frame       
            dist_blue_connector = math.sqrt((pose_blue_button.pose.position.x-pose_red_connector.pose.position.x)**2+(pose_blue_button.pose.position.y-pose_red_connector.pose.position.y)**2) # distance between the blue button and the red connector in camera frame
            print("calculated distance between blue button and connector :", dist_blue_connector, "\n")    
            dist_red_button_connector = math.sqrt((pose_red_button.pose.position.x-pose_red_connector.pose.position.x)**2+(pose_red_button.pose.position.y-pose_red_connector.pose.position.y)**2) # distance between the red button and the red connector in camera frame
            print("calculated distance between red button and connector :", dist_red_button_connector, "\n")    
            dist_screen_connector = math.sqrt((pose_led_screen.pose.position.x-pose_red_connector.pose.position.x)**2+(pose_led_screen.pose.position.y-pose_red_connector.pose.position.y)**2) # distance between the led screen and the red connector in camera frame
            print("calculated distance between led screen and connector :", dist_screen_connector, "\n")    
            dist_slider_connector = math.sqrt((pose_slider.pose.position.x-pose_red_connector.pose.position.x)**2+(pose_slider.pose.position.y-pose_red_connector.pose.position.y)**2) # distance between the slider and the red connector in camera frame
            print("calculated distance between slider and connector :", dist_slider_connector, "\n")    
            dist_door_handle_connector = math.sqrt((pose_door_handle.pose.position.x-pose_red_connector.pose.position.x)**2+(pose_door_handle.pose.position.y-pose_red_connector.pose.position.y)**2) # distance between the door handle and the red connector in camera frame
            print("calculated distance between door handle and connector :", dist_door_handle_connector, "\n")    
            real_dist_blue_connector = 0.063 # real distance between blue button and red connector
            real_dist_red_button_connector = 0.060 # real distance between red button and red connector
            real_dist_screen_connector = 0.067 # real distance between led screen and red connector
            real_dist_slider_connector = 0.0725 # real distance between slider and red connector
            real_dist_door_handle_connector = 0.088 # real distance between door handle and red connector             
            inac_dist_blue_connector = abs(real_dist_blue_connector - dist_blue_connector) # inaccuracy between real distance and calculated distance of blue button and red connector
            print("inaccuracy between blue button and connector = {} mm".format(inac_dist_blue_connector*1000), "\n")
            inac_dis_red_button_connector = abs(real_dist_red_button_connector - dist_red_button_connector) # inaccuracy between real distance and calculated distance of red button and red connector
            print("inaccuracy between red button and connector = {} mm".format(inac_dis_red_button_connector*1000), "\n")
            inac_dist_screen_connector = abs(real_dist_screen_connector - dist_screen_connector) # inaccuracy between real distance and calculated distance of led screen and red connector 
            print("inaccuracy between led screen and connector = {} mm".format(inac_dist_screen_connector*1000), "\n")
            inac_dis_slider_connector = abs(real_dist_slider_connector - dist_slider_connector) # inaccuracy between real distance and calculated distance of slider and red connector
            print("inaccuracy between slider and connector = {} mm".format(inac_dis_slider_connector*1000), "\n")
            inac_dist_door_handle_connector = abs(real_dist_door_handle_connector - dist_door_handle_connector) # inaccuracy between real distance and calculated distance of door handle and red connector 
            print("inaccuracy between led screen and connector = {} mm".format(inac_dist_door_handle_connector*1000), "\n")
            list = [inac_dist_blue_connector, inac_dis_red_button_connector, inac_dist_screen_connector, inac_dis_slider_connector, inac_dist_door_handle_connector] 
            print(list, "\n")
            first_best_accurate_distance = min(list) # looking for the first accurate distance
            print("first best accurate distance :", first_best_accurate_distance, "\n")
            index = list.index(first_best_accurate_distance)        
            del list[index]
            if first_best_accurate_distance == inac_dist_blue_connector : # recover the best coordinates for the first point
                print(" The first most accurate point is the blue button ", "\n")
                first_point = blue_button               
            if first_best_accurate_distance == inac_dis_red_button_connector :
                print(" The first most accurate point is the red button ", "\n")
                first_point = red_button                
            if first_best_accurate_distance == inac_dist_screen_connector :
                print(" The first most accurate point is the led screen", "\n")
                first_point = led_screen           
            if first_best_accurate_distance == inac_dis_slider_connector :
                print(" The first most accurate point is the slider", "\n")
                first_point = slider               
            if first_best_accurate_distance == inac_dist_door_handle_connector :
                print(" The first most accurate point is the door handle", "\n")
                first_point = door_handle    
            new_list = list
            print(new_list, "\n")
            second_best_accurate_distance = min(new_list) # looking for the second accurate distance
            print("second best accurate distance :", second_best_accurate_distance, "\n")
            new_index = new_list.index(second_best_accurate_distance)
            del new_list[new_index]          
            if second_best_accurate_distance == inac_dist_blue_connector :
                print("The second most accurate point is the blue button ")
                second_point = blue_button    
            if second_best_accurate_distance == inac_dis_red_button_connector :
                print("The second most accurate point is the red button")
                second_point = red_button                
            if second_best_accurate_distance == inac_dist_door_handle_connector :
                print("The second most accurate point is the led screen")
                second_point = led_screen
            if second_best_accurate_distance == inac_dis_slider_connector :
                print("The second most accurate point is the slider")
                second_point = slider                  
            if second_best_accurate_distance == inac_dist_door_handle_connector :
                print("The second most accurate point is the door handle")
                second_point = door_handle                       
        return (first_point, second_point)

# if __name__ == '__main__':
    
#     MyDetection = MyDetection() 
#     x, y = MyDetection.capture_points()
#     print("coordinates of the returned points", x, y)



    