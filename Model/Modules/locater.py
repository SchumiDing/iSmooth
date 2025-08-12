import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
class objectRecorder:
    def __init__(self, video_file=None, Model=None):
        self.history = {}
        '''
        {
            f"{ObjCategory}_{Index}": {
                "template": np.array,  # Template image for the object
                "locations": [
                    {x1: int, y1: int, x2: int, y2: int, confidence: float, class_id: int, img: np.array},
                    {x1: int, y1: int, x2: int, y2: int, confidence: float, class_id: int, img: np.array},
                    ...
                ]
            }
            ...
        }
        '''
        self.index = 0
        self.movement = {}
        '''
        {
            f"{ObjCategory}_{Index}": {
                "template": np.array,
                "move": [
                    {"x_Move": int, "y_Move": int},
                    {"x_Move": int, "y_Move": int},
                    ...
                ]
            }
            ...
        }
        '''
        if video_file:
            self.have_video = True
            self.videoFile = video_file
        else:
            self.have_video = False
            self.videoFile = None
        if Model:
            self.have_model = True
            self.Model = Model
        else:
            self.have_model = False
            self.Model = None
        self.threshold = 0.8
        self.yolothreshold = 0.5
        
    def setThreshold(self, threshold: float):
        self.threshold = threshold
        return self.threshold
    
    def setYoloThreshold(self, yolothreshold: float):
        self.yolothreshold = yolothreshold
        return self.yolothreshold

    def getHistory(self):
        return self.history
    
    def getMovement(self):
        return self.movement
    
    def compareImages(self, img1: np.array, img2: np.array):
        # Compute the Structural Similarity Index (SSI) between the two images
        ssim = cv2.SSIM(img1, img2)
        # ssim is a value between -1 and 1, where 1 means the images are identical
        return ssim
    
    def ifObjExists(self, obj_category):
        comp = {}
        for key in self.history.keys():
            comp[key] = self.compareImages(self.history[key]["template"], obj_category)
        comp = sorted(comp.items(), key=lambda item: item[1], reverse=True)

        if comp and comp[0][1] > self.threshold:  # Threshold for similarity
            return True, comp[0][0]  # Return the key of the most similar object
        return False, None

    def calMovement(self):
        for key in self.history.keys():
            name = key
            self.movement[name] = {
                "template": self.history[key]["template"],
                "move": []
            }
            startP = self.history[key]["locations"][0]
            startX = (startP["x1"] + startP["x2"]) // 2
            startY = (startP["y1"] + startP["y2"]) // 2
            for loc in self.history[key]["locations"][1:]:
                X = (loc["x1"] + loc["x2"]) // 2
                Y = (loc["y1"] + loc["y2"]) // 2
                x_Move = X - startX
                y_Move = Y - startY
                self.movement[name]["move"].append({
                    "x_Move": x_Move,
                    "y_Move": y_Move
                })
        return self.movement
        # This method calculates the movement of each object based on its locations
        
    def detect(self):
        if not (self.have_video and self.have_model):
            raise ValueError("Video file or model not provided.")
        self.history = {}
        self.movement = {}
        cap = cv2.VideoCapture(self.videoFile)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.Model(frame)
            for result in results:
                for box in result.boxes:
                    if box.confidence > self.yolothreshold:
                        x1, y1, x2, y2 = map(int, box.xyxy)
                        class_id = int(box.cls)
                        img = frame[y1:y2, x1:x2]
                        obj_category = self.Model.names[class_id]
                        exist, key = self.ifObjExists(obj_category)
                        if exist:
                            self.history[key]["locations"].append({
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "confidence": box.confidence, "class_id": class_id, "img": img
                            })
                        else:
                            self.history[f"{obj_category}_{self.index}"] = {
                                "template": img,
                                "locations": [{
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "confidence": box.confidence, "class_id": class_id, "img": img
                                }]
                            }
                            self.index += 1
        cap.release()
        self.calMovement()
        return self.movement