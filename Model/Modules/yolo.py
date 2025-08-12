import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from Model.config.globe import model_path

model = YOLO(model_path)