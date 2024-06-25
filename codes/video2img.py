"""
Project: staff exclusion

This is the preprocessing script to extract the video into image frames for model training.

"""
import pathlib
import cv2
import random
import os
import shutil

def vid2img():
    """
    Function: convert video to a folder of frame images
    Input: none
    Output: none
    """
    # video path 
    video_path = "/content/drive/MyDrive/staff_exclusion/datasamples/sample.mp4"
    #frames path
    sample_path='/content/drive/MyDrive/staff_exclusion/datasamples/'
    save_folder = os.path.join(sample_path, './frames')
    
    # make temp dir
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{}] ...'.format(save_folder))
        
    # capture video frame by frame
    capture = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        ret, frame = capture.read()
        if ret:
        # write the frame image
            cv2.imwrite(os.path.join(save_folder, str(frame_num)+'.png'),frame )

            frame_num += 1
        else:
            break

    capture.release()
    print("Done converting video to frame images!")
      
    
if __name__ == "__main__":
    vid2img()
