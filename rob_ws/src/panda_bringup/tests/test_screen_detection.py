from board_detection.screen_detection import DetectScreen
import torch

def main():

    detect = DetectScreen()

    with torch.no_grad():

        estimated_stage = 1
        previous_position = 0
        premature_advance = False
        premature_success = False

        while estimated_stage < 3:

            # Detection
            led_screen_image = detect.crop_screen(show=False)
            #adjusted_image = detect.apply_brightness_contrast(led_screen_image, brightness=0, contrast=0, show=False)
            slider_image, slider_length = detect.crop_slider(led_screen_image, show=True)
            
            if slider_length == 0:

                if estimated_stage == 2 and not premature_success:
                    print('Warning: no triangle detected in stage 2, retrying..')
                    premature_success = True
                    continue
                elif estimated_stage == 2 and premature_success:
                    print('Info: no triangle detected in stage 2, considered premature success!')
                    estimated_stage = 3
                    break
                else:
                    print('Warning: slider detection failed, retrying')

                continue
            
            triangle_pixels = detect.detect_triangle(slider_image, show=True)

            # No triangle detected
            if(len(triangle_pixels) == 0):
                if estimated_stage == 1:
                    print('Warning: no triangle detected in stage 1, retrying..')
                    continue
                if estimated_stage == 2 and not premature_success:
                    print('Warning: no triangle detected in stage 2, retrying..')
                    premature_success = True
                    continue
                elif estimated_stage == 2 and premature_success:
                    print('Info: no triangle detected in stage 2, considered premature success!')
                    estimated_stage = 3
                    break
            
            # One triangle detected
            elif(len(triangle_pixels) == 1):
                if estimated_stage == 1:
                    print('Info: expected number of triangles, press a key to proceed')
                    input()
                    previous_position = move_slider(slider_length, triangle_pixels)
                    estimated_stage == 2
                    continue
                if estimated_stage == 2:
                    print('Warning: expected 2 triangles, got 1, press a key to proceed')
                    input()
                    previous_position = move_slider(slider_length, triangle_pixels)
                    continue

            elif(len(triangle_pixels) == 2):
                if estimated_stage == 1 and not premature_advance:
                    print('Warning: expected 1 triangle got 2 in stage 1, retrying..')
                    premature_advance = True
                    continue
                elif estimated_stage == 1 and premature_advance:
                    print('Warning: expected 1 triangle got 2 in stage 1, considered premature advance, press a key to proceed')
                    input()
                    estimated_stage == 2
                    continue
                else:
                    print('Info: Detected all triangles, press a key to proceed')
                    input()
                    previous_position = move_slider(slider_length, triangle_pixels, previous_position)
                    premature_success = True
                    continue

            else:
                print('Warning: detected the wrong nomber of triangles, retrying..')
                continue

            def move_slider(slider_length, pixels_positions, previous_position=None):
                
                slider_size = 0.036
                slider_offsets = []
                for pos in pixels_positions:
                    percent = 100 - (pos * 100 / slider_length)
                    offset = percent * slider_size / 100
                    slider_offsets.append(offset)

                if len(pixels_positions) == 0:
                    print('goal position empty, not moving')
                    return 0
                
                elif len(pixels_positions) == 1:
                    print('single goal: %s', slider_offsets[0])
                    return slider_offsets[0]
                
                elif len(pixels_positions) == 2 and previous_position:
                    # Remove the previous pose for the result
                    dist0 = abs(slider_offsets[0] - previous_position)
                    dist1 = abs(slider_offsets[1] - previous_position)

                    if dist0 > dist1:
                        print('slected from 2 goals: %s', slider_offsets[0])
                        return slider_offsets[0]
                    else:
                        print('slected from 2 goals: %s', slider_offsets[1])
                        return slider_offsets[1]
                    
                else:
                    print('missing previous_position or wrong number of goals, not moving')
                    return 0

if __name__ == "__main__":

    main()