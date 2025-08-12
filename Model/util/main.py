import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from Model.Modules.yolo import model
from Model.Modules.locater import objectRecorder

videofile = "data/video.mp4"
# Open the video file
cap = cv2.VideoCapture(videofile)

recorder = objectRecorder()

if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                obj_category = model.names[class_id]
                
                img = frame[y1:y2, x1:x2]
                img = np.array(img)
                exists, key = recorder.ifObjExists(obj_category)
                if exists:
                    recorder.history[key]["locations"].append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": confidence,
                        "class_id": class_id,
                        "img": img
                    })
                else:
                    recorder.history[f"{obj_category}_{recorder.index}"] = {
                        "template": img,
                        "locations": [{
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": confidence,
                            "class_id": class_id,
                            "img": img
                        }]
                    }
                    recorder.index += 1
