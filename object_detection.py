# Importing the neccesary modules
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

old_model = YOLO("yolo11n.pt")
old_model.export(format="openvino", dynamic = True)
model = YOLO("yolo11n_openvino_model")

# Invokes YOLO to return the objects found within an image
def detect_objects(frame, image_size):
    results = model(frame, imgsz = image_size)
    annotated_frame = results[0].plot()
    
    item_list = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            class_name = model.names[class_id]
            
            detection = {"Class": class_name, "Confidence": conf, "BBox": xyxy}
            item_list.append(detection)
    
    return annotated_frame, item_list