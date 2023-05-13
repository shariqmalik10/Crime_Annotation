import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join( script_dir, './')
sys.path.append(mymodule_dir)

from detect_txt import *


def detect_final(source):
    path, object_detections = detect(source=source,weights='yolov7/yolov7.pt')
    print(object_detections)
    # isCrime = 
    if len(object_detections) != 0 or object_detections.get("person")is not None:
        # isCrime = True
        path,weapon_detections = detect(source=path,weights='yolov7/best.pt')
        #weapons = set(['pistol','knife'])
        object_detections.update(weapon_detections)
        # else:
        #     isCrime = False
    return path, object_detections


# print(detect_final("yolov7/inference/images/Abuse005_x264.mp4"))