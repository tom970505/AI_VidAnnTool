import sys, os
import cv2
import numpy as np
# =========================================
from labeling import videoLabling
# =========================================
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basedir,'Tracker','Re3_tracker'))

from tracker import re3_tracker

# =========================================

class Re3TrackerRecordHandle:
    
    def __init__(self, trackerModel, recordModel, parent=None):
        self.tracker = trackerModel
        self.recordData = recordModel #videoLabling()
        self.itemLst = []
        self.objClassMap = {}
        self.objAttriMap = {}
        self.cap = None
        self.run = False
        
    def getData(self):
        return self.recordData.data
        
    # -------- new video or label ------------------------------------- 
    def openVideo(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        jsonFile = filepath.rsplit( ".", 1 )[0] +'.json'
        #print(jsonFile)
        if self.cap.isOpened():
            suss = self.recordData.loadData(jsonFile)
            if suss:
                print("Load Exist Label")
                #print('====== load exist data ======')
            else:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resol = [width, height]
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                totalFrame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #print('====== create NEW data ======')
                print("Make NEW Label: ", filepath, resol, fps, totalFrame)
                
                _, file = os.path.split(filepath)
                #print(file)
                self.createNewData(file, resol, fps, totalFrame)
        else:
            print('!!!! Video file not exist !!!!')
            self.recordData.resetData()
    
    def createNewData(self, filepath, resol, fps, totalFrame):
        self.recordData.newFileData(filepath, resol, fps, totalFrame)
    #-------- video part ------------------------------------------------ 
    def setFrameId(self, fid):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

    def readFrame(self):
        fid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = []
        return ret, rgb, frame, fid
    
    def seleObj(self, objLst): #####
        return objLst
    
    # ------load label ------------------------------------------------------
    def renderclassLabel(self, frame, fid, data, classColorMap): # bbx order according to object list order
        if str(fid) in data['Object bounding box']:
            if len(data['Object bounding box'][str(fid)]) > 0:
                for index, (objName, value) in enumerate(data['Object bounding box'][str(fid)].items()):
                    #print(index,objName,value['bbx'],value['class'])
                    
                    clas = value['class']
                    x, y, x1, y1 = value['bbx']
                    #### value['attribute']
                    color = classColorMap[clas]
                    
                    # bbx part
                    cv2.rectangle(frame,(int(x),int(y)), (int(x1), int(y1)),color, 2)
                    #text, font, font_scale , thickness = str(i), cv2.FONT_HERSHEY_SIMPLEX , 1, 2 
                    text, font, font_scale , thickness = objName, cv2.FONT_HERSHEY_SIMPLEX , 0.7, 1
                    
                    # text part
                    text_bgr = (0,0,0)
                    text_w, text_h = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
                    text_x, text_y = int(x)+2, int(y)-2
                    text_box0, text_box1 = (text_x, text_y), (text_x + text_w - thickness, text_y - text_h - thickness)
                    cv2.rectangle(frame, text_box0, text_box1, text_bgr, cv2.FILLED)
                    cv2.putText(frame, text, text_box0 , font, font_scale, color, thickness)
                    
    def renderLabel(self, frame, fid, data):
        if str(fid) in data['Object bounding box']:
            if len(data['Object bounding box'][str(fid)]) > 0:
                #print('=================')
                for index, (key, value) in enumerate(data['Object bounding box'][str(fid)].items()):
                    #print(index,key,value['bbx'],value['class'])
                    x, y, x1, y1 = value['bbx']
                    #### value['attribute']
                    cv2.rectangle(frame, (int(x),int(y)), (int(x1), int(y1)), (0, 0, 255), 2)
                    cv2.putText(frame, str(key), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
                #print('=================')
    
    def loadLabel(self):
        eventRange, dataKey, data = self.recordData.extractData()
        
        if self.cap is not None:
            if len(dataKey) > 0:
                start,end = eventRange
                self.setFrameId(start)
                while(self.cap.isOpened()):
                    ret, rgb, frame, fid = self.readFrame()
                    if ret:
                        self.renderLabel(frame, fid, data)
                        cv2.imshow('frame',frame)
                        if fid > end or cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) <1:
                            break
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break
                self.cap.release()
            else:
                print('No bounding box information exist!!')
    # ------ create label ------------------------------------------------------
    def setObjLst(self, objLst):
        self.itemLst = objLst

    def setFrameId(self, fid):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

    # ------- tracker --------------------------------------------------------------
    def getObjLst(self):
        return self.itemLst
    
    def setObjLst(self, objLst, objClassMap, objAttriMap): ###
        self.itemLst = objLst
        self.objClassMap = objClassMap
        self.objAttriMap = objAttriMap
        
    def newObjAppend(self, fid, objLst, rectDic, objClassMap, objAttriMap):
        #self.stFid = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.setObjLst(objLst, objClassMap, objAttriMap)
        self.setFrameId(fid)
        ret, rgb, frame, _ = self.readFrame()
        
        if len(self.itemLst) == 1: # total number of new obj
            self.tracker.track( self.itemLst[0] , rgb, list(rectDic.values())[0])
        elif len(self.itemLst) > 1:
            self.tracker.multi_track( self.itemLst , rgb, rectDic)
        self.startRecord(fid, objLst, rectDic) ###
        self.run = True
        
    def objTrack(self, frame, fid):
        #if self.stFid < fid:
        if len(self.itemLst) == 1: 
            bbox = self.tracker.track(self.itemLst[0], frame)
            return [bbox]
                
        elif len(self.itemLst) > 1:
            bbxes = self.tracker.multi_track(self.itemLst, frame)
            return bbxes.tolist()

    def renderTrack(self, frame, objLst, bbxes): # bbx order according to object list order
        for i, (objName, bbox) in enumerate(zip(objLst, bbxes)):
            # color part
            color = cv2.cvtColor(np.uint8([[[i * 255 / len(bbxes), 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze().tolist()
            
            # bbx part
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),color, 2)
            text, font, font_scale , thickness = objName, cv2.FONT_HERSHEY_SIMPLEX , 1, 2
            
            # text part
            text_bgr = (0,0,0)
            text_w, text_h = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            text_x, text_y = int(bbox[0])+2, int(bbox[1])-2
            text_box0, text_box1 = (text_x, text_y), (text_x + text_w - thickness, text_y - text_h - thickness)
            cv2.rectangle(frame, text_box0, text_box1, text_bgr, cv2.FILLED)
            cv2.putText(frame, text, text_box0 , font, font_scale, color, thickness)

            
    # ----Recording part ---------------------------------------------------------------------------
    def recorder(self, fid, objLst, bbxes):
        # record part
        res = {}
        for name, bbx in zip(objLst, bbxes):
            bbx = [int(bbx[0]), int(bbx[1]), int(bbx[2]), int(bbx[3])]
            classNam = self.objClassMap[name]
            stateFlag = self.objAttriMap[name]
            res.update({name : { "class": classNam , "bbx": bbx ,"event state": stateFlag}}) ##
        #print(fid, res)
        self.recordData.appendFrameData(fid, res, track=True)
        
    def startRecord(self, fid, objLst, rectDic):
        res = {}
        for name in objLst:
            bbx = rectDic[name]
            classNam = self.objClassMap[name]
            stateFlag = self.objAttriMap[name]
            res.update({name : { "class": classNam , "bbx": bbx, "event state" : stateFlag }}) ##
        #print(fid,res)
        self.recordData.appendFrameData(fid, res, track=True)  ####
        
    
    def doTrack(self):
       if self.cap is not None:
            while(self.cap.isOpened() and self.run): 
                ret, rgb, frame, fid = self.readFrame()
                if ret:
                    bbxes = self.objTrack(rgb, fid)
                    self.renderTrack(frame, self.itemLst, bbxes )
                    cv2.putText(frame, text=str(fid), org=(3, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255, 0, 0), thickness=2)
                    self.recorder(fid, self.itemLst, bbxes)   ###
                    #if self.recordData.data is not None:
                    #    self.renderLabel(frame, fid, self.recordData.data) # render together
                    cv2.imshow('frame',frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            #self.cap.release()
            cv2.destroyWindow('frame') 
    
    def saveLabel(self, filename):
        self.recordData.saveJsonFile(filename)
        
    # -------- Editing data ---------------------------------------------------
    def setEventInterData(self, interv):
        self.recordData.setEventInterv(interv)
    
    def reset(self):
        if self.cap is not None:
            self.cap.release()
        self.itemLst = []
        self.objClassMap = {}
        self.objAttriMap = {}
        self.cap = None
        self.run = False

def re3Track():    
    filepath = '20190223101654.MP4'
    model = re3_tracker.Re3Tracker()
    record = videoLabling() 
    # ----- load video
    track = Re3TrackerRecordHandle(model, record )
    track.openVideo(filepath)
    # ----- generate label ----
    mode = 'tracker'
    if mode == 'tracker':
        startFrame = 96
        objLst = ['Item1-1', 'Item2-1']
        rectDic = {'Item1-1': [437, 248, 479, 306], 'Item2-1': [475, 261, 516, 296]}
        objClassMap = {'Item1-1': "car" , 'Item2-1': "car"}
        #objAttriMap = {'Item1-1': [] , 'Item2-1': ["lane change"]}
        objAttriMap = {'Item1-1': True , 'Item2-1': False}
        track.newObjAppend(startFrame, objLst, rectDic, objClassMap, objAttriMap) ###
        track.doTrack()
        track.saveLabel('test00000.json')
        track.reset() # clear data
    else:
        # ----- data viewer -----
        track.loadLabel()
