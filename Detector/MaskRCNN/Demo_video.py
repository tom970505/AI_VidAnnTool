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


def apply_mask(image, mask, color, alpha=0.3):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores, class_dict):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        
        mask = masks[:, :, i]
        #image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        """
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1
        )
        """

    return image


if __name__ == '__main__':
    MODEL_DIR = "logs"
    COCO_MODEL_PATH = "mask_rcnn_coco.h5"

    class_names = coco.class_names
    colors = random_colors(len(class_names))
    class_dict = {
        name: color for name, color in zip(class_names, colors)
    }

    config = coco.InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    cap = cv2.VideoCapture('0195.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model.detect([frame], verbose=0)[0]
            frame = display_instances(
                frame, results['rois'], results['masks'], results['class_ids'], class_names, results['scores'], class_dict
            )
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break    
    cap.release()


