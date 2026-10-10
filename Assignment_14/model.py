import ultralytics
from ultralytics import YOLO

model_segmentation = YOLO("yolov8n-seg.pt")  # load a pretrained segmentation