import os
import sys

import cv2
import numpy as np
import colorsys
import random

import cocoConfig as coco
import mrcnn.model as modellib

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    random.seed(6)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    color = []
    for col in list(colors):
        color.append(tuple(i*255 for i in list(col))) 
    return color


class MaskRcnn:
    MODEL_DIR = "logs"
    basedir = os.path.dirname(os.path.abspath(__file__))
    COCO_MODEL_PATH = os.path.join(basedir, "mask_rcnn_coco.h5")
    
    def __init__(self):
        self.class_names = coco.class_names
        colors = random_colors(len(self.class_names))
        self.class_dict = {
            name: color for name, color in zip(self.class_names, colors)
        }
        config = coco.InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

    def detect(self, frame):
        results = self.model.detect([frame], verbose=0)[0]
        return results
        
    def apply_mask(self, image, mask, color, alpha=0.3):
        """apply mask to image"""
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image


    def display_instances(self, image, results):
        """
            take the image and results and apply the mask, box, and Label
        """
        boxes, masks, ids, scores = results['rois'], results['masks'], results['class_ids'], results['scores']
        n_instances = boxes.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                print('----notAny----')
                continue

            y1, x1, y2, x2 = boxes[i]
            label = self.class_names[ids[i]]
            color = self.class_dict[label]
            score = scores[i] if scores is not None else None

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # ----- mask part ------
            mask = masks[:, :, i]
            self.apply_mask(image, mask, color)
            # ----- bbx part ------
            caption = '{} {:.2f}'.format(label, score) if score else label
            cv2.putText(
                image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1
            )
            
    
    def detect_bbx(self, image):
        results = self.detect(image)
        boxes, masks, ids, scores = results['rois'], results['masks'], results['class_ids'], results['scores']
        n_instances = boxes.shape[0]
        res = []
        for i in range(n_instances):
            if not np.any(boxes[i]):
                print('----notAny----')
                continue

            y0, x0, y1, x1 = boxes[i] # y0 x0 y1 x1
            predicted_class = self.class_names[ids[i]]
            score = scores[i] if scores is not None else None
            res.append((predicted_class, score, (x0 + 0.5*(x1- x0), y0 +0.5*(y1 -y0), x1-x0, y1-y0)))
            #cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return res

if __name__ == '__main__':
    detector = MaskRcnn()
    cap = cv2.VideoCapture('0195.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = detector.detect(frame)
            detector.display_instances(frame, results)
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break    
    cap.release()


