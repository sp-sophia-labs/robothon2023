from board_detection.board_detection import DetectMulticlass
import torch

def main():

    Detect = DetectMulticlass()

    with torch.no_grad():
        points = Detect.capture_points(show=True) 
        print(points)

if __name__ == "__main__":

    main()