import cv2
import json

with open('data/partial_body_keypoints.json') as anno_file:
    probes = json.load(anno_file)

with open('data/whole_body_keypoints.json') as anno_file:
    galleries = json.load(anno_file)


