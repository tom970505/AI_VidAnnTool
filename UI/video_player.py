import time
import sys
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker
from PyQt5.QtWidgets import QApplication

#from PyQt5.QtCore import *
#from PyQt5.QtGui import *

import cv2 

class VideoBox: 

    VIDEO_TYPE_OFFLINE = 0
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""

    def __init__(self, video_url="", video_type=VIDEO_TYPE_OFFLINE, auto_play=False):
        
        # ===== global variable =====
        self.video_url = video_url
        self.video_type = video_type    # 0: offline  1: realTime
        self.auto_play = auto_play
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause
        self.total_frame = 0
        # ====== Some prepartion ===========
        
        # timer -------
        self.timer = VideoTimer()
        #self.timer.timeSignal.signal[str].connect(self.show_video_images)
        
        # video -------
        self.playCapture = cv2.VideoCapture()
        
        if self.video_url != "":
            self.playCapture.open(self.video_url)
            self.total_frame = int(self.playCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.playCapture.get(cv2.CAP_PROP_FPS)
            self.timer.set_fps(fps)
            self.playCapture.release()

    def reset(self):
        self.timer.stop()
        self.playCapture.release()
        self.status = VideoBox.STATUS_INIT
        self.total_frame = 0

class Communicate(QObject):
    signal = pyqtSignal(str)

class VideoTimer(QThread):
    def __init__(self, frequent=20): # signal, parent=None
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()
        # self.signal = signal

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return 
            else:
                self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)
            
    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps

