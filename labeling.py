

import os, re
import cv2
import numpy as np
import json
from collections import OrderedDict

def trans_intKey(dict_key):
    try:
        return int(dict_key)
    except ValueError:
        return dict_key
    
class videoLabling:
    keyOrder = ['File name','Resolution','FPS','Total frame','Event type','Event interval','Frame event','Object bounding box']

    def __init__(self):
        self.data = {}
        self.itemNameSet = set()
        self.ObjClassMap = {}
        self.run = False
        
    # ===== load json =============================
    def loadData(self, file):
        if os.path.isfile(file):
            with open(file, 'r', encoding="utf-8") as f:
                self.data = json.load(f)
                self.getObjNameLst() ###
            return True
        else:
            return False
        
    # ===== object Name ==============================
    def getObjNameLst(self): # and class map 
        if 'Object bounding box' in self.data:
            res = set()
            
            for fid in self.data['Object bounding box'].keys():
                tmpName = list(self.data['Object bounding box'][fid].keys())
                res.update(tmpName)
                for item in tmpName:
                    tmpClass = self.data['Object bounding box'][fid][item]['class']
                    self.ObjClassMap.update({item:tmpClass})
            if len(res) > 0:
                self.itemNameSet.update(res)

    def resetObjNameLst(self): # over data
        self.itemNameSet = set()
        self.ObjClassMap = {}
        self.getObjNameLst() ###

    def isGoodObjName(self, name): # outer name 
        if name in self.itemNameSet:
            return False
        else:
            return True
        
    def autoGenNewName(self, tmpExistNam = []):
        cnt = 0
        tmpData = set(tmpExistNam) | self.itemNameSet
        while True:
            if str(cnt) not in tmpData:
                break
            cnt += 1
        return str(cnt)
    
    # ===== object Class ============================
    def setObjClassMap(self, objName, objclass):
        if objName in self.itemNameSet:
            self.ObjClassMap.update({objName: objclass})
        
    def resetObjClassOverData(self):
        for fid in self.data['Object bounding box'].keys():
            tmpName = list(self.data['Object bounding box'][fid].keys())
            for item in tmpName:
                
                if self.data['Object bounding box'][fid][item]['class'] != self.ObjClassMap[item]:
                    print('Class data unconsistant!!')
                
    # ===== create new data if no label =============        
    def newFileData(self, filename, resol, fps, totalFrame ):
        self.data = {'File name' :'' ,  # path
            'Resolution' : (0,0) ,      # w,h
            'FPS': 0,
            'Total frame': 0 ,
            'Event type': '' ,  
            'Event interval': [] ,
            'Frame event' : {} ,
            'Object bounding box': {}   # { frame id : {'itemName' : {'bbx':[x,y,w,h], 'class':'car' } }
            }
        self.data['File name'] = filename
        self.data['Resolution'] = resol
        self.data['FPS'] = fps
        self.data['Total frame'] = totalFrame
        
    # ===== check for labeling data =================
    def labelChecker(self, filepath):
        def readFrame(cap):
            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            return ret, frame, fid

        def renderLabel(frame, fid, data, classLst):
            if str(fid) in data['Object bounding box']:
                if len(data['Object bounding box'][str(fid)]) > 0:
                    #print('=================')
                    #print(str(fid))
                    for index, (key, value) in enumerate(data['Object bounding box'][str(fid)].items()):
                        #print(index,key,value['bbx'],value['class'])
                        i = classLst.index(value['class'])
                        color = cv2.cvtColor(np.uint8([[[i * 255 / len(classLst) , 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze().tolist()
                        x, y, x1, y1 = value['bbx']
                        cv2.rectangle(frame, (int(x),int(y)), (int(x1), int(y1)), color, 2)
                        cv2.putText(frame, str(key), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,0.5,color,1)
                    #print('=================')
        self.run = True        
        interv = self.data['Event interval']
        if len(interv) == 2:
            st, ed = interv[0], interv[1]
            self.getObjNameLst() # update exist name list
            classLst = list(set(self.ObjClassMap.values()))
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, st)
            while(cap.isOpened() and self.run):
                ret, frame, fid = readFrame(cap) ##
                if ret:
                    renderLabel(frame, fid, self.data, classLst) ##
                    content = "From "+ str(st)+" to "+str(ed)+" : "+ str(fid)
                    cv2.putText(frame, text=content, org=(3, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(255, 0, 0), thickness=2)
                    cv2.imshow('frame',frame)
                    if fid >= ed or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) <1:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
        
        
    # ===== time domain data ========================
    def setEventInterv(self, interval):
        total = self.data['Total frame']
        if interval[1] > total:
            self.data['Event interval'] = interval
        else:
            print('Invalid interval !!')
            
    # ===== get new data =======================
    def appendFrameType(self, fid, label):
        if len(label) > 0:
            self.data['Frame event'].update({str(fid): label } )
        elif len(label) == 0:
            if str(fid) in self.data['Frame event']:
                del self.data['Frame event'][str(fid)]
        
    def appendFrameData(self, fid, objDict, track=False): ## different ytwTracker, re3 did not have to assign name
        itemName = list(objDict.keys())
        self.itemNameSet.update(itemName)
        if str(fid) in self.data['Object bounding box']:
            if not track:
                self.data['Object bounding box'].update({str(fid): objDict } )
            else:
                self.data['Object bounding box'][str(fid)].update(objDict )
                
            if len(self.data['Object bounding box'][str(fid)]) == 0:
                del self.data['Object bounding box'][str(fid)]
        else:
            if len(objDict) > 0:
                self.data['Object bounding box'].update({str(fid): objDict } )
            
            
    # ===== get basic information =================
    # event interval
    def extractData(self):
        filepath = self.data['File name']
        dataKey = []
        eventRange = []
        if 'Event interval' in self.data: # label exist !
            eventRange = self.data['Event interval']
            if len(eventRange) ==0: # no event interval but have bbx
                if 'Object bounding box' in self.data:
                    dataKey = self.data['Object bounding box'].keys()
                    if len(dataKey) >0:
                        fids = [int(i) for i in dataKey]
                        eventRange = [min(fids), max(fids)]
                        #eventRange = [min(int(dataKey)), max(int(dataKey))]        elif 'Object bounding box' in self.data: # label exist !
            if len(self.data['Object bounding box'])>0:
                dataKey = self.data['Object bounding box'].keys()
                if len(dataKey) >0:
                    fids = [int(i) for i in dataKey]
                    eventRange = [min(fids), max(fids)]
                    #eventRange = [min(int(dataKey)), max(int(dataKey))]
        
        return eventRange, dataKey, self.data

    def getFrameData(self, fid):
        #====================================
        def stringSplitByNumbers(x):
            r = re.compile('(\d+)')
            l = r.split(x)
            return [int(y) if y.isdigit() else y for y in l]
        
        def Sort(sub_li): 
            # reverse = None (Sorts in Ascending order) 
            # key is set to sort using second element of  
            # sublist lambda has been used 
            return(sorted(sub_li, key = lambda x: stringSplitByNumbers(x[0])))
        #====================================
        res = []
        if 'Object bounding box' in self.data:
            if str(fid) in self.data['Object bounding box']:
                if len(self.data['Object bounding box'][str(fid)]) > 0:
                    #print('=================')
                    for index, (objName, value) in enumerate(self.data['Object bounding box'][str(fid)].items()):
                        #print(index,objName,value['bbx'],value['class'])
                        res.append([objName, value['bbx'],value['class'],value['event state'] ])
                    #print('=================')
                    res = Sort(res)
                    #print(res)
        frameLab = ''
        if 'Frame event' in self.data:
            if str(fid) in self.data['Frame event']:
                if len(self.data['Frame event'][str(fid)]) > 0:
                    frameLab = self.data['Frame event'][str(fid)]
        return res, frameLab
        
    # ===== remove object data =================
    def setDataBetween(self, start, end):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            dataKey = tmpData.keys()
            if len(dataKey) > 0:
                interKey = [int(i) for i in range(start, end+1)]
                for key in dataKey:
                    if int(key) not in interKey:
                        del self.data['Object bounding box'][key]
                self.resetObjNameLst()
                        
    def cleanDataAfter(self, start):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            dataKey = tmpData.keys()
            if len(dataKey) > 0:
                for key in dataKey:
                    if int(key) >= start: # include start
                        del self.data['Object bounding box'][key]
                self.resetObjNameLst()
            else:
                print('No data to delete')

    def removeObj(self, objLst):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            for fid in tmpData.keys():
                for obj in objLst:
                    if obj in tmpData[fid].keys():
                        del self.data['Object bounding box'][fid][obj]
                if len(self.data['Object bounding box'][fid]) ==0:  
                    del self.data['Object bounding box'][fid]
            self.resetObjNameLst()
            
    def removeObjAfter(self, start, objLst):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            fids = tmpData.keys()
            if len(fids) > 0:
                for fid in fids:
                    if int(fid) >= start: # include start
                        for obj in objLst:
                            if obj in self.data['Object bounding box'][fid]:
                                del self.data['Object bounding box'][fid][obj]
                self.resetObjNameLst()
            else:
                print('No data to delete')
                
     # ============================================================
    def setBatchObjAttriAfter(self, start, objNamLst, objStateMap):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            fids = tmpData.keys()
            if len(fids) > 0:
                for fid in fids:
                    if int(fid) >= start: # include start
                        for objNam in objNamLst:
                            if objNam in self.data['Object bounding box'][fid]:
                                self.data['Object bounding box'][fid][objNam]['event state'] = bool(objStateMap[objNam])
                            
    def setAttriAfter(self, start, objNam, attriFlag):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            fids = tmpData.keys()
            if len(fids) > 0:
                for fid in fids:
                    if int(fid) >= start: # include start
                        if objNam in self.data['Object bounding box'][fid]:
                            self.data['Object bounding box'][fid][objNam]['event state'] = bool(attriFlag)
                            
    def setObjClass(self, objNamClassMap):
        if 'Object bounding box' in self.data:
            tmpData = self.data['Object bounding box'].copy()
            fids = tmpData.keys()
            if len(fids) > 0:
                for fid in fids:
                    for obj in objNamClassMap.keys():
                        if obj in self.data['Object bounding box'][fid]:
                            self.data['Object bounding box'][fid][obj]['class'] = objNamClassMap[obj]
    
    # ===== save file =========================
    
    def saveJsonFile(self, filename = 'data.json'):
        if 'Object bounding box' in self.data:
            if len(self.data['Object bounding box']) ==0:
                print('--- Warning, no object exist ! ---')
            else:
                for fid in self.data['Object bounding box'].keys():
                    for itemKey in self.data['Object bounding box'][fid].keys():
                        self.data['Object bounding box'][fid][itemKey] = OrderedDict([(key, self.data['Object bounding box'][fid][itemKey][key]) for key in ['class','bbx', 'event state']])
                    self.data['Object bounding box'][fid] = OrderedDict([(k, v) for k, v in sorted(self.data['Object bounding box'][fid].items())])
                self.data['Object bounding box'] = OrderedDict([(key, self.data['Object bounding box'][key]) for key in sorted(self.data['Object bounding box'].keys(), key= trans_intKey)]) ##
        if 'Frame event' in self.data:
            if len(self.data['Frame event']) ==0:
                print('--- Warning, no frame label exist ! ---')
            else:
                self.data['Frame event'] = OrderedDict([(key, self.data['Frame event'][key]) for key in sorted(self.data['Frame event'].keys(), key= trans_intKey)]) 
            
        result = OrderedDict([(key, self.data[key]) for key in self.keyOrder])
        #print(json.dumps(result, indent=2, ensure_ascii=False))
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4 , ensure_ascii=False, sort_keys=False)
            print('~~~File Save Finished~~~')
            
    def resetData(self):
        self.data = {}
        self.itemNameSet = set()
        self.ObjClassMap = {}
        self.run = False
        
#def main():
if __name__ == "__main__":
    record = videoLabling()
    record.loadData('20190223101654.json')
    print("Totally : ",len(record.itemNameSet))
    #record.setDataBetween(0,12)
    #record.cleanDataAfter(12)
    record.removeObj(['0'])
    print(record.data['Object bounding box']['10'])
    record.removeObjAfter(10,['3'])
    #record.getFrameData(0)
    #record.removeObj(['1','3'])
    #record.remameObj(0, ['1','3'], ['2','5']) # old -> new
    
    #print(record.isGoodObjName('1'))
    #print(record.autoGenNewName())
    #print(record.autoGenNewName(['5']))
    
    print("Totally : ",len(record.itemNameSet))
    print(record.data['Object bounding box']['10'])
    print("Totally : ",len(record.ObjClassMap))
    #record.saveJsonFile('test.json')
    
