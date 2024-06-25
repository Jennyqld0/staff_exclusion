# Staff Exclusion

Writer: [Qingling Duan](https://jennyqld0.github.io/)
[Test requirement link](https://github.com/Jennyqld0/staff_exclusion/blob/main/datasamples/AI%20Evaluation%20Test.pdf).
Solution: Use the object detection model, specifically [Ultralytics YOLO](https://docs.ultralytics.com/models/).  Software and devices: [roboflow](https://app.roboflow.com/123-djtgh/staff-exclusion/), [google colab](https://drive.google.com/drive/my-drive).

---

## Data Labelling

**Data Preprocessing**: The video sample was extracted into video frames. See [codes/video2img.py](https://github.com/Jennyqld0/staff_exclusion/blob/main/codes/video2img.py) for details.

In this case, the objects to mark are people ( **sticking out
their chests** ) and name tags. Use [roboflow](https://app.roboflow.com/123-djtgh/staff-exclusion/).

See [datatset](https://github.com/Jennyqld0/staff_exclusion/tree/main/dataset_yolov8).

```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="uMKcGx4k10XKiTcybh8A")
project = rf.workspace("123-djtgh").project("staff-exclusion")
version = project.version(1)
dataset = version.download("yolov8")
```

### Model training

Use[ google calab](https://drive.google.com/drive/my-drive), switch to gpu environment, use YOLO library to train my database.

1. install:  `!pip install ultralytics`
2. train:` !yolo train model=yolov8s.pt data="data.yaml" epochs=150 imgsz=640`

see [detail code](https://github.com/Jennyqld0/staff_exclusion/blob/main/codes/yolov8_train.ipynb).
see [training result](https://github.com/Jennyqld0/staff_exclusion/tree/main/runs).

---

## Staff Exclusion

To predict staff: If the entire bounding box of the name tag is inside the people bounding box, then people are staff.

For the results of the YOLO model, the bounding box coordinates of the top right and bottom left corner of the target are output.

```
model = YOLO('yolov8s.pt') 
model = YOLO('best.pt')    
results = model(frame)             
boxes = results[0].boxes
for name_tag in boxes: 
	if name_tag.cls[0] == 0:
		for box in boxes:      
			if(name_tag.xyxy[0][0] >= box.xyxy[0][0] and        
			name_tag.xyxy[0][1] >= box.xyxy[0][1] and      
			name_tag.xyxy[0][2] <= box.xyxy[0][2] and     
			name_tag.xyxy[0][3] <= box.xyxy[0][3] and       
			box.cls[0] == 1.0):         
			staff_coordinate.append(box.xyxy[0].tolist())
```

see [detail code](https://github.com/Jennyqld0/staff_exclusion/blob/main/codes/predict.py).

---

### Output

see [csv](https://github.com/Jennyqld0/staff_exclusion/blob/main/results/staff_yolov8.csv).

