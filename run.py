import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.Modules.locater import objectRecorder
from Model.config.globe import model_path

video_path = "test.mp4"

detector = objectRecorder(video_file=video_path, Model=model_path)

detector.setThreshold(0.7)
detector.setYoloThreshold(0.6)

detector.detect()

visData = detector.prepareVisData()

import json
json.dump(visData, open("visData.json", "w"), indent=4)