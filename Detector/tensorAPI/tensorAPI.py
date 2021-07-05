import os, sys
import cv2
import numpy as np
import tensorflow as tf


from obj_detect_api.utils import label_map_util

class tfObjdectApi:
    def __init__(self, PATH_TO_CKPT):
        # =========== label maps ===================
        label_map_dict = label_map_util.get_label_map_dict(os.path.join(PATH_TO_CKPT, 'label_map.pbtxt')) #{name : id}
        self.inv_map = {v: k for k, v in label_map_dict.items()}
        # =========== model preparation ============
        ckpt_path = os.path.join(PATH_TO_CKPT, 'frozen_inference_graph.pb')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            gpu_options = tf.GPUOptions(allow_growth=True) ####
            self.sess = tf.Session(graph=detection_graph,config=tf.ConfigProto(gpu_options=gpu_options))
        # ============ output tensors ===============
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def detect_bbx(self, rgb):
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_expanded = np.expand_dims(bgr, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, class_ids, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        
        res = []
        h, w = rgb.shape[:2]
        for box, score, cid in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(class_ids)):
            y1,x1,y2,x2 = box
            x1, y1 = int(x1*w), int(y1*h)
            x2, y2 = int(x2*w), int(y2*h)
            predicted_class = self.inv_map.get(int(cid), None)
            ####### ==================================== #######
            if predicted_class is not None:
                res.append((predicted_class, score, (x1 + 0.5*(x2-x1), y1 + 0.5*(y2-y1), x2-x1, y2-y1)))
        return res
        
    def test_detect_bbx(self, rgb):
        #self.detector.detect_bbx(rgb)
        frame_expanded = np.expand_dims(rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, class_ids, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        
        return boxes, scores, class_ids, num
    
    def visualize(self, color_map, inv_map, score_thresh,  boxes, scores, class_ids, num):
        # Draw the results of the detection (aka 'visulaize the results')
        for box, score, cid in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(class_ids)):
            y1,x1,y2,x2 = box
            # id to color
            name = inv_map.get(int(cid), None)
            h, w = frame.shape[:2]
            if name is not None and score > score_thresh:
                cv2.rectangle(frame, (int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)), color_map[name], 2)


if __name__ == '__main__':
    COLOCOLORS_RGB = [(124,32,154), (20,235,201), (1,68,221), (255,129,190)]
    
    PATH_TO_VIDEO = '001.mp4'
    PATH_TO_CKPT = ['faserRes50_COCO','fasterRes101_KITTI','faster_nas_COCO'][2]
    
    score_thresh = 0.8
    
    label_map_dict = label_map_util.get_label_map_dict(os.path.join(PATH_TO_CKPT, 'label_map.pbtxt')) #{name : id}
    inv_map = {v: k for k, v in label_map_dict.items()} #{id : name}
    color_map = dict(zip(list(label_map_dict.keys()), [ COLOCOLORS_RGB[int(i%len(COLOCOLORS_RGB))] for i in range(len(label_map_dict))]))

    detector = tfObjdectApi(PATH_TO_CKPT)
    video = cv2.VideoCapture(PATH_TO_VIDEO)
     
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            boxes, scores, class_ids, num = detector.test_detect_bbx(frame)
            detector.visualize(color_map, inv_map, score_thresh,  boxes, scores, class_ids, num)
            cv2.imshow('frame_demo', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

