import cv2
from ultralytics import YOLO
import csv
import pandas as pd
import os

model = YOLO('yolov8s.pt')  # Initialize
model = YOLO('/content/drive/MyDrive/staff_exclusion/runs/detect/yolov8/weights/best.pt')     # Load custom model
model.to('cuda')            # Use GPU 
video_path='/content/drive/MyDrive/staff_exclusion/datasamples/sample.mp4'
save_path="/content/drive/MyDrive/staff_exclusion/results/staff_yolov8.csv"

def detect():
    frame_num=0    
    fields=['x-1','y-1','x-2','y-2','staff_num','frame_num']
    cap = cv2.VideoCapture(video_path)
    
    with open(save_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        while cap.isOpened():
            success, frame = cap.read()
            key = cv2.waitKey(1) & 0xFF
            if success:
                
                staff_coordinate = []
                # Run YOLOv8 inference on the frame
                results = model(frame)
                
                # Change to Numpy for easier manipulation
                boxes = results[0].boxes
                boxes = boxes.to('cpu')
                boxes = boxes.numpy()
    
                # Record the coordinates of the staff
                # Loop through each Label
                # If Label is inside the bounding box of the staff, update the class of the corresponding bounding box to 2.0 (staff)
                for name_tag in boxes:
                    if name_tag.cls[0] == 0:
                        for box in boxes:
                            if (name_tag.xyxy[0][0] >= box.xyxy[0][0] and
                                name_tag.xyxy[0][1] >= box.xyxy[0][1] and
                                name_tag.xyxy[0][2] <= box.xyxy[0][2] and
                                name_tag.xyxy[0][3] <= box.xyxy[0][3] and 
                                box.cls[0] == 1.0):
                                staff_coordinate.append(box.xyxy[0].tolist())
                                
                if len(staff_coordinate) != 0:
                    print('--------------------------------------------------------------------------------------------------')
                    for i, coordinate in enumerate(staff_coordinate, start=1):
                        print(f"frame {frame_num} Staff {i} Coordinate: {coordinate}")
                        coordinate.append(i) 
                        coordinate.append(frame_num)
                        print(coordinate)
                        writer.writerow(coordinate)
                    
                    print('--------------------------------------------------------------------------------------------------\n')     

                frame_num +=1
                
                # Break the loop if 'q' is pressed, Pause the video if 'p' is pressed
                if key == ord("q"):
                    break
                elif key == ord("p"):
                    cv2.waitKey(-1) #wait until any key is pressed
            else:
            # Break the loop if the end of the video is reached
                break

    # Release the video capture object and close the display window
    file.close()  
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()
      