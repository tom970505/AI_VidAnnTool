import sys, os
import random
import cv2
import numpy as np
import colorsys
import cmapy
import re
# --------------------------------------------
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyleFactory, QSplashScreen,qApp, QFileDialog,QListView,QMessageBox, QShortcut, QSlider,QStyle, QGraphicsRectItem, QApplication, QGraphicsView, QGraphicsScene, QGraphicsItem, QHBoxLayout,QInputDialog
from PyQt5.QtCore import Qt, QRectF, QPointF, QRect, pyqtSlot, QStringListModel
from PyQt5.QtGui import QBrush, QPainterPath, QPainter, QColor, QPen, QPixmap, QImage,QIntValidator, QKeySequence 
# REF https://het.as.utexas.edu/HET/Software/PyQt/classes.html
# --------------------------------------------
from UI.ui_part import Ui_MainWindow
from UI.video_player import VideoBox,VideoTimer
from UI.bbx_part import CustomScene,BoxItem 
# --------------------------------------------
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(basedir,'Tracker','Re3_tracker'))
sys.path.append(os.path.join(basedir,'Detector','MaskRCNN'))
sys.path.append(os.path.join(basedir,'Detector','Yolo'))
sys.path.append(os.path.join(basedir,'Detector','tensorAPI'))

from yolo import YOLO
from MaskRCNN import MaskRcnn
from tensorAPI import tfObjdectApi
# --------------------------------------------
from tracker import re3_tracker
from tracker_handler import Re3TrackerRecordHandle
from labeling import videoLabling
# --------------------------------------------

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

class LabelTool(QMainWindow, Ui_MainWindow): # VideoBox
    def __init__(self, *args, **kwargs):

        # ========== GUI part ==================
        
        QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.setWindowTitle("Anotating Tool")
        
        # ========== global variable ===========  
        self.videoHandler = None
        self.fileList = None
        self.totalFile = 0
        self.curFileIdx = 0
        #-----------------------
        self.thisFile = ''
        self.thisFolder = ''
        self.pixmap = None 
        self.frame_id = 0
        self.prev_frame_id = 0
        self.total_frame = 0
        #-----------------------
        self.thisframe = None
        self.startFlag = False
        
        # ---------- frame level data ----------
        self.tmpNamLst = []
        self.bbxMapping = {}       # itemObj : cell obj
        # ---------- video level data ----------
        self.start = 0
        self.end = 0
        self.colorLst = []
        self.colorMapping = {}     # classNam : color data
        self.tmpInterv = []
        # ========== function binding ===========
        # 0. load initial page
        self.loadInitPage()
        
        # 1. load dir & video
        self.loadFolderBtn.clicked.connect(self.loadDir)
        
        # 2. prev & next video
        self.prevFileBtn.clicked.connect(self.preFile) ##
        self.nextFileBtn.clicked.connect(self.nextFile) ##

        self.folderPathLabel.setText('No Folder Opened')
        self.showTimeInfo.setText('No File Opened')
        self.fileNameLabel.setText('No File Opened')
        
        # ============ show video ================
        
        # 3. ply & pause button
        self.ply_pausBtn.clicked.connect(self.switch_video)

        # 4. video slider
        self.playerSlider.setRange(0, 0)
        self.playerSlider.setTickPosition(QSlider.NoTicks)
        self.playerSlider.sliderMoved.connect(self.slider_set_frameId)

        # 5. input frame jump
        self.inFrameId.returnPressed.connect(self.input_set_frameId)
        self.pIntvalidator = QIntValidator(self)
        self.pIntvalidator.setRange(0,0)
        self.inFrameId.setValidator(self.pIntvalidator)
        
        # prev and next frame
        self.prevFramBtn.clicked.connect(self.prevFrame)
        self.nextFramBtn.clicked.connect(self.nextFrame)
        self.reloadBtn.clicked.connect(self.reloadFrame)
        
        # ============ label file ================
        
        # 6. create / cancel:  bounding box 
        self.creLabelBtn.clicked.connect(self.switch_BbxMode) #.connect(self.creBbx)

        # 7. delete bounding box
        self.removLabelBtn.clicked.connect(self.delBbx)
        self.clrBbxBtn.clicked.connect(self.clrBbx)

        # 8. click listwidget
        
        self.LabelLst.itemSelectionChanged.connect(self.bbxSetSelected)
        
        # 9. create / delete : category
        self.catCombo.setEditable(False)
        self.newCatBtn.clicked.connect(self.newCatgo)
        self.delCatBtn.clicked.connect(self.delCatgo)

        self.saveBtn.clicked.connect(self.saveCurPage)
        # 10. Tracker part
        self.track_model = re3_tracker.Re3Tracker()
        self.record = videoLabling()
        self.tracker = Re3TrackerRecordHandle(self.track_model, self.record) ##
        self.startTrackBtn.clicked.connect(self.startTrack)
        self.stopTrackBtn.clicked.connect(self.stopTrack)
        
        # 11. Label editing part
        self.clrBbxAfterBtn.clicked.connect(self.clrObjAfter)
        self.removBbxAfterBtn.clicked.connect(self.removeObjAfter)
        self.setStartBtn.clicked.connect(self.setStart)
        self.setEndBtn.clicked.connect(self.setEnd)
        self.showPrevFramBtn.clicked.connect(self.showPrevFram)
        
        # 12. save label
        self.labelCheckBtn.clicked.connect(self.labelCheck) 
        self.labelCheckBtn_2.clicked.connect(self.labelCheck) 
        self.setLabelRangeBtn.clicked.connect(self.setRangeLabel)
        self.exportBtn.clicked.connect(self.saveLabel)

        # 13. redo / undo
        
        # ----- init state -----
        self.loadObjClass()
        self.loadFramClass()
        self.prev_objModl = ""
        self.objAlgriCombo.addItem("yolo3")
        self.objAlgriCombo.addItem("maskRcnn")
        self.objAlgriCombo.addItem("fasterRcnn")
        
        self.trackAlgriCombo.addItem("re3")
        
        self.showFramLab.setText('No File Opened')
        self.showFramLab_2.setText('No File Opened')
        
        self.framCombo.setEnabled(False)
        self.framCombo.currentTextChanged.connect(self.framComboChanged)
        # ----- model part -----------------------
        PATH_TO_CKPT = os.path.join(basedir,'Detector','tensorAPI','faster_COCO')
        
        self.yolo_model = YOLO()
        self.mask_model = MaskRcnn()
        self.faster_model = tfObjdectApi(PATH_TO_CKPT)
        
        self.detector = None
        self.autoDetectBtn.clicked.connect(self.autoCreBbx)

        # ---------------------------------------------------------
        #  Shortcut key
        # ---------------------------------------------------------
        # Key_Delete, Key_F3, Key_Enter
        self.PlayStateKey = QShortcut(Qt.Key_Space, self)
        self.PlayStateKey.activated.connect(self.switch_video)
        
        self.setPrevKey = QShortcut(Qt.Key_Left, self)
        self.setPrevKey.activated.connect(self.prevFrame)

        self.setNxtKey = QShortcut(Qt.Key_Right, self)
        self.setNxtKey.activated.connect(self.nextFrame)
        
        self.setReloadKey = QShortcut(QKeySequence("Ctrl+R"), self)
        self.setReloadKey.activated.connect(self.reloadFrame)
        self.showPrevKey = QShortcut(QKeySequence("Ctrl+P"), self)
        self.showPrevKey.activated.connect(self.showPrevFram)
        
        self.setStartKey = QShortcut(QKeySequence("Ctrl+S"), self)
        self.setStartKey.activated.connect(self.setStart)
        self.setEndKey = QShortcut(QKeySequence("Ctrl+E"), self)
        self.setEndKey.activated.connect(self.setEnd)
        
        # ----- uncomplete part ------------------
        self.redoBtn.setEnabled(False)
        self.undoBtn.setEnabled(False)
        self.reversSeleBbxBtn.setEnabled(False)
        self.edLabelToolBtn_1.setEnabled(False)
        #self.edLabelToolBtn_2.setEnabled(False)
        self.setAttriBtn.clicked.connect(self.setObjAttrAfter)
        
        self.edLabelToolBtn_2.setText("Reset Obj Class ")
        self.edLabelToolBtn_2.clicked.connect(self.resetObjClass)
        # loadSettings()
        # ==========================================
        # 8. get pixel position
        #self.frameViewer.photoClicked.connect(self.photoClicked)
        
        # ==========================================
        
    # ---------------------
    def framComboChanged(self):
        frameLabel = str(self.framCombo.currentText())
        self.showFramLab.setText(frameLabel)
        self.showFramLab_2.setText(frameLabel)
        
    def loadFramClass(self):
        if os.path.exists(os.path.join('ObjClasses','Frame_Classes.txt')):
            basedir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(basedir,'ObjClasses','Frame_Classes.txt')
            if len(filename)>0:
                self.framCombo.addItem('')
                with open(filename) as inputfile:
                    for line in inputfile:
                        lineContent = line.strip().lower().split(',')
                        existItem = [self.framCombo.itemText(i).strip().lower() for i in range(self.framCombo.count())]
                        if lineContent !=[''] and lineContent not in existItem:
                            text = lineContent[0]
                            self.framCombo.addItem(str(text))

    # ===== category related ============================================
    def newColor(self):
        custom_color_file = os.path.join('ObjClasses','Obj_Classes_Color.txt')
        
        if os.path.exists(custom_color_file):
            filename = os.path.join(custom_color_file)
            with open(filename) as inputfile:
                for line in inputfile:
                    lineContent = line.strip().lower().split(',')  # (R,G,B)
                    if lineContent !=['']:
                        color = tuple(map(int, lineContent))
                        if color not in self.colorLst:
                            return color
                        
        if os.path.exists(os.path.join('ObjClasses','Obj_Classes_Name.txt')):
            with open(os.path.join('ObjClasses','Obj_Classes_Name.txt')) as inputfile:
                class_num = 0
                for line in inputfile:
                    class_num+=1
            if len(self.colorLst) < class_num:      
                for i in range(class_num):
                    color = tuple(cv2.cvtColor(np.uint8([[[i * 255 / class_num , 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze())
                    if color not in self.colorLst:
                        return color
                    
        color = tuple(cmapy.color('viridis', random.randrange(0, 256), rgb_order=True))
        while(color in colorLst):
            color = tuple(cmapy.color('viridis', random.randrange(0, 256), rgb_order=True))
        return color   
      
    def loadObjClass(self):
        if os.path.exists(os.path.join('ObjClasses','Obj_Classes_Name.txt')):
            basedir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(basedir,'ObjClasses','Obj_Classes_Name.txt')
        else:
            filename = QFileDialog.getOpenFileName(self, 'Open class File', '','Text Files (*.txt)')[0]

        if len(filename)>0:
            with open(filename) as inputfile:
                for line in inputfile:
                    lineContent = line.strip().lower().split(',')
                    existItem = [self.catCombo.itemText(i).strip().lower() for i in range(self.catCombo.count())]
                    if lineContent !=[''] and lineContent not in existItem:
                        text = lineContent[0]
                        color = self.newColor()
                        #print(color)
                        self.colorLst.append(color)
                        self.colorMapping.update({text:color})
                        # 3. add category
                        self.catCombo.addItem(str(text))
                        #self.catCombo.setEditable(True)
            """
            print('=====================')
            
            for value in [self.catCombo.itemText(i).strip().lower() for i in range(self.catCombo.count())] :
                print(value)
            
            for key, value in self.colorMapping.items() :
                print(key, value)
            
            print('=====================')
            """
        else:
            print('Create a new category ~')
            text, ok = QInputDialog.getText(self, 'Input settings', 'Create a new category:')
            if ok:
                print("Create a new category:", str(text))
                # 1. check if name repeat
                items = [self.catCombo.itemText(i).strip().lower() for i in range(self.catCombo.count())]
                if not text.strip().lower() in items:
                    # 2. assign a color
                    color = self.newColor()
                    self.colorLst.append(color)
                    self.colorMapping.update({text:color})
                    # 3. add category
                    self.catCombo.addItem(str(text))
                    #self.catCombo.setEditable(True)
                else:
                    QMessageBox.information(self,'Warning','You enter an exist category !')
        
    def newCatgo(self):
        #def getText(self):
        # 1. if combo is empty : load class file, else none
        if self.catCombo.count() == 0:
            self.loadObjClass()
        elif self.catCombo.count() >=1:
            text, ok = QInputDialog.getText(self, 'Input settings', 'Create a new category:')
            if ok:
                # 1. check if name repeat
                items = [self.catCombo.itemText(i).strip().lower() for i in range(self.catCombo.count())]
                if not text.strip().lower() in items:
                    # 2. assign a color
                    color = self.newColor()
                    self.colorLst.append(color)
                    self.colorMapping.update({text:color})
                    # 3. add category
                    self.catCombo.addItem(str(text))
                    #self.catCombo.setEditable(True)
                else:
                    QMessageBox.information(self,'Warning','You enter an exist category !')  
    
    #def renamCatgo(self):
    def delCatgo(self):
        if self.catCombo.count()>0:
            items = [self.catCombo.itemText(i) for i in range(self.catCombo.count())]
            item, ok = QInputDialog.getItem(self, 'Select item to delete', 'list of ', items, 0, False)
            if ok and item:
                # ===== delete class related bbx =====
                # 1. search for color listitem
                # 2. select bbxitem
                # 3. delbbx
                # ====================================
                # disable drawing
                self.frameViewer._scene.newItem = False
                self.creLabelBtn.setText("Create")
                # find the index of text
                index = self.catCombo.findText(item)
                # query item color
                for bbxit in self.frameViewer._scene.items():
                    if isinstance(bbxit, BoxItem):
                        if bbxit.col == self.colorMapping[item]:
                            # delete bbx 
                            self.frameViewer._scene.removeItem(bbxit)
                            # delete label view
                            cellItem = self.bbxMapping[bbxit]
                            table_row = self.LabelLst.row(cellItem)
                            self.LabelLst.removeRow(table_row)
                            del self.bbxMapping[bbxit]
                # delete category
                self.colorLst.remove(self.colorMapping[item])
                self.catCombo.removeItem(index)
                del self.colorMapping[item]

    #==========================================================       
    # ----- Bounding box Related -------------    
                 
    def switch_BbxMode(self): # create bbx
        if self.pixmap is not None:
            if not self.frameViewer._scene.newItem:
                # --- create a bounding box ----
                self.frameViewer._scene.newItem = True
                self.creLabelBtn.setText("Cancel")
            else:
                self.frameViewer._scene.newItem = False
                self.creLabelBtn.setText("Create")
              
    def clrBbx(self):
        if self.pixmap is not None:
            self.frameViewer.cnt = 0 
            self.frameViewer._scene.newItem = False
            items = self.frameViewer._scene.items()
            for it in items:
                if isinstance(it, BoxItem):
                    #print(it.getPosition())
                    #print(self.frameViewer.mapFromScene(it.getPosition()).boundingRect())
                    self.frameViewer._scene.removeItem(it)
            self.bbxMapping = {}
            self.tmpNamLst = []
            self.LabelLst.itemSelectionChanged.disconnect()
            self.LabelLst.clean()
            self.LabelLst.itemSelectionChanged.connect(self.bbxSetSelected)
            self.creLabelBtn.setText("Create")
  
    def delBbx(self):
        if self.pixmap is not None:
            self.frameViewer._scene.newItem = False
            self.creLabelBtn.setText("Create")
            
            # 1. get selectedItems
            items = self.frameViewer._scene.selectedItems() ###
            #self.frameViewer.cnt -= len(items)
            print(len(items),' items')
            
            # 2. remove label lst
            if len(self.bbxMapping) > 1:
                for bbxit in items:
                    cellItem = self.bbxMapping[bbxit]
                    table_row = self.LabelLst.row(cellItem) # item Binding
                    self.LabelLst.removeRow(table_row)
                    self.frameViewer._scene.removeItem(bbxit)
                    del self.bbxMapping[bbxit]
            
            elif len(self.bbxMapping) == 1:
                bbxit = items[0]
                self.frameViewer._scene.removeItem(bbxit)
                del self.bbxMapping[bbxit]
                self.LabelLst.itemSelectionChanged.disconnect()
                self.LabelLst.clean()
                self.LabelLst.itemSelectionChanged.connect(self.bbxSetSelected)
                
            # 3. remove bbx
            #while items:
            #    item = items.pop()
            #    self.frameViewer._scene.removeItem(item)

    def autoCreBbx(self):
        if self.pixmap is not None and self.thisframe is not None:
            # --- create a bounding box ----
            self.startTrackBtn.setEnabled(False)
            self.ply_pausBtn.setEnabled(False)
            self.autoDetectBtn.setEnabled(False)
            self.objAlgriCombo.setEnabled(False)

            cur_objModl  = str(self.objAlgriCombo.currentText())
            
            if cur_objModl != self.prev_objModl:
                print("Current Obj Model:", cur_objModl) 
                self.prev_objModl = cur_objModl
                self.detector = None
                if cur_objModl =="yolo3":
                    self.detector = self.yolo_model
                elif cur_objModl =="maskRcnn":
                    self.detector = self.mask_model
                elif cur_objModl =='fasterRcnn':
                    self.detector = self.faster_model
                
            if self.detector is not None:
                bbxes = self.detector.detect_bbx(self.thisframe)
                
                # class , score, [cx,cy,w,h]
                if len(bbxes) > 0:
                    for bbx in bbxes:
                        clas, score, cbbx = bbx[0], bbx[1], bbx[2]
                        if clas in self.colorMapping.keys() and score>0.3: # minimum accept accuracy for object detection
                            #print(clas, score, cbbx)
                            #w,h = cbbx[2], cbbx[3]
                            objName = self.tracker.recordData.autoGenNewName(self.tmpNamLst)
                            # ---- convert to PyQt format --------------
                            x0,y0 = int(cbbx[0]- 0.5*cbbx[2]), int(cbbx[1] - 0.5*cbbx[3])
                            x1,y1 = int(x0+cbbx[2]), int(y0+cbbx[3])
                            rect = QRectF(x0, y0, x1-x0, y1-y0)
                            # ------------------------------------------
                            color = self.colorMapping[clas]
                            self.frameViewer.createNewBbx(objName, color, rect) ### default = false
                            self.tmpNamLst.append(objName)
                else:
                    QMessageBox.information(self,'Warning','No object Detect !')
                    
                self.autoDetectBtn.setEnabled(True)
                self.objAlgriCombo.setEnabled(True)
                self.startTrackBtn.setEnabled(True)
                self.ply_pausBtn.setEnabled(True)

    # ====================================================================
    # ---- Object Related -----
    
    def bbxSetSelected(self):
        """
        while table row change, bbx set selected
        """
        self.frameViewer._scene.clearSelection()
        items = self.LabelLst.selectedItems()   # select listItem
        self.LabelLst.itemSelectionChanged.disconnect()
        for it in items:
            cell = self.LabelLst.item(it.row(), 0)  # bind with first column item
            bbx = [bbxIt for bbxIt, tableIt in self.bbxMapping.items() if tableIt == cell][0]
            bbx.setSelected(True)
            #bbx.setFocus()
        self.LabelLst.itemSelectionChanged.connect(self.bbxSetSelected)
   
    # =============================  
    def getBbxInfo(self,bbxItem): ###
        getClass = dict(zip(self.colorMapping.values(), self.colorMapping.keys()))
        #  This would work only if the values are hashable and unique
        itemClass = getClass[bbxItem.col]
        # ---------------------------------------------------------------------------------------
        # rect = self.frameViewer.mapFromScene(bbxItem.getPosition()).boundingRect() # x y w h 
        rect = bbxItem.getPosition() # ~~~~ for label ~~~~
        itemRect = [int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())]          # x0 y0 x1 y1
        table_row =  self.LabelLst.row(self.bbxMapping[bbxItem])
        
        itemNam, attriFlag = self.LabelLst.getDataByRow(table_row)
        
        # ----------------------------------------------------------------------------------------
        return itemNam, itemClass, itemRect, attriFlag
    
    def getFrameSeleInfo(self):
        # list all bbxitem: name , bbx
        # --- item list ---
        rectDic = {}
        classDic = {}
        stateFlagDic = {}
        seleNamLst = []
        for it in self.frameViewer._scene.selectedItems():
            if isinstance(it, BoxItem):
                itemNam, itemClass, itemRect, attriFlag = self.getBbxInfo(it)
                rectDic.update({itemNam : itemRect })
                classDic.update({itemNam : itemClass})
                stateFlagDic.update({itemNam : attriFlag}) ###
                seleNamLst.append(itemNam)
        return seleNamLst, classDic, rectDic, stateFlagDic
    
    # =============================  
    def getFrameAllInfo(self):
        rectDic = {}
        classDic = {}
        seleNamLst = []
        stateFlagDic = {}
        for it in self.frameViewer._scene.items():
            if isinstance(it, BoxItem):
                itemNam, itemClass, itemRect, attriFlag = self.getBbxInfo(it)
                rectDic.update({itemNam : itemRect })
                classDic.update({itemNam : itemClass})
                stateFlagDic.update({itemNam : attriFlag}) ###
                seleNamLst.append(itemNam)
        return seleNamLst, classDic, rectDic, stateFlagDic

    #------------------------------------------------------------------
    # get information from label
    def getFrameIdInfo(self, fid):
        if self.pixmap is not None and self.thisframe is not None:
            labelData = self.tracker.recordData.getFrameData(fid) ####
            rectDic = {}
            classDic = {}
            objNamLst = []
            stateFlagDic = {}
            for data in labelData:
                itemNam, itemRect, itemClass, attriFlag = data[0], data[1], data[2], data[3]
                rectDic.update({itemNam : itemRect })
                classDic.update({itemNam : itemClass})
                stateFlagDic.update({itemNam : attriFlag})
                objNamLst.append(itemNam)
            return objNamLst, classDic, rectDic

    def showPrevFram(self):
        if self.pixmap is not None and self.thisframe is not None:
            prevfid = max(self.frame_id-1, 0)
            classColorMap = {}
            totalclassNum = self.catCombo.count()
                
            if totalclassNum > 0:
                
                for idx in range(totalclassNum):
                    classNam = self.catCombo.itemText(idx)
                    color = cv2.cvtColor(np.uint8([[[idx * 255 / totalclassNum, 128, 200]]]),cv2.COLOR_HSV2RGB).squeeze().tolist()
                    classColorMap.update({classNam: color })
                    
                self.tracker.setFrameId(prevfid)
                ret, _, frame, fid = self.tracker.readFrame()
                if ret:
                    self.tracker.renderclassLabel( frame, fid ,self.tracker.recordData.data,classColorMap )
                    cv2.putText(frame, text=str(fid), org=(3, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(238, 0, 224), thickness=2)
                    cv2.imshow('frame',frame)
                    cv2.waitKey(0)
                    
    # auto create Bbx
    def renderLabel(self, fid):
        if self.pixmap is not None and self.thisframe is not None:
            # ----- handle tracking ------
            #cv2.imshow('image',self.thisframe)
            #cv2.waitKey(0)
            """
            index = 0
            self.framCombo.setCurrentIndex(index)
            """
            labelData, framLab = self.tracker.recordData.getFrameData(fid) ####
            #### framLab
            if len(framLab) == 0:
                self.framCombo.setCurrentIndex(0)
            elif len(framLab) > 0:
                index = self.framCombo.findText(framLab, Qt.MatchFixedString)
                if index >= 0:
                    self.framCombo.setCurrentIndex(index)
                else:
                    print('=== Add new frame label ===')
                    self.framCombo.addItem(str(framLab))
                    print('===========================')
            
                
            #if len(self.colorMapping)>0:
            for label in labelData:
                objName = label[0]
                bbox = label[1]
                cls = label[2]
                stateFlag = bool(label[3])
                #print(stateFlag)
                if cls not in self.colorMapping.keys(): # auto generate unique color and label
                    color = self.newColor()
                    self.colorLst.append(color)
                    self.colorMapping.update({cls: color})
                    self.catCombo.addItem(str(cls))
                    
                color = self.colorMapping[cls]  
                # -----------------------------------------
                
                x0, y0, x1, y1 = bbox
                rect = QRectF(x0, y0, x1-x0, y1-y0)
                self.frameViewer.createNewBbx(objName, color, rect, stateFlag= stateFlag)
                # -----------------------------------------
    # ====================================================================
                    
    def labelCheck(self):
        if self.pixmap is not None:
            #self.show_video_images() ###
            filepath = '/'.join([self.thisFolder,self.thisFile])
            self.tracker.recordData.labelChecker(filepath) ###
        
    # ----- Time Related ------
    def startTrack(self):
        if self.pixmap is not None and self.thisframe is not None:
            self.frameViewer._scene.newItem = False
            self.creLabelBtn.setText("Create")
            
            self.startTrackBtn.setEnabled(False)
            self.ply_pausBtn.setEnabled(False)
            
            # ========== Track start from current frame ================
            #  --- new track for selected item ----
            fid = self.frame_id
            #prevfid = max(fid-1, 0)
            seleNamLst, classDic, rectDic, stateFlagDic = self.getFrameSeleInfo() ###
        
            if len(seleNamLst) > 0:
                askReply = QMessageBox.question(self, 'Warning', "Remember to rename object before track!", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if askReply == QMessageBox.Yes:
                    self.tracker.newObjAppend(fid, seleNamLst, rectDic, classDic, stateFlagDic) ##
                    self.tracker.doTrack()
            elif len(seleNamLst) ==0:
                QMessageBox.information(self,'Warning',"You don't select any item") 
                
            self.startTrackBtn.setEnabled(True)
            self.ply_pausBtn.setEnabled(True)
    def stopTrack(self):
        self.tracker.run = False
        #self.reloadFrame()

    # ---------------------------------------------------------------- 
    
            
    # ========== select then open file ============
    
    def selectFile(self,qModelIndex):
        # ? save current file
        self.reset()
        self.curFileIdx = qModelIndex.row()
        self.thisFile = self.fileList[self.curFileIdx]
        self.fileNameLabel.setText(self.thisFile )
        #print(self.thisFile)
        self.openFile()
                
    def preFile(self):
        # ? save current file
        self.reset()
        if len(self.fileList) !=0 :
            if self.curFileIdx >= 1:
                self.curFileId -= 1
                self.thisFile = self.fileList[self.curFileIdx]
                self.fileNameLabel.setText(self.thisFile )
                #print(self.thisFile)
                self.openFile()
            
    def nextFile(self):
        # ? save current file
        self.reset()
        if len(self.fileList) !=0 :
            if self.curFileIdx < self.totalFile-1 :
                self.curFileIdx += 1
                self.thisFile = self.fileList[self.curFileIdx]
                self.fileNameLabel.setText(self.thisFile )
                #print(self.thisFile)
                self.openFile()
            
    # ========== open the source file ===================
        
    def loadDir(self):
        print('LoadDir...')
        folder = QFileDialog.getExistingDirectory(self, "choose a folder", "./")
        if len(folder)!= 0:
            self.thisFolder = folder                 ##
            self.folderNameLabel.setText(folder.split('/')[-1])
            self.folderPathLabel.setText(folder)
            self.listFile(folder)
            print(folder)
            
    def listFile(self, path):
        if os.path.isdir(path):
            self.reset()
            support_files =  ['mp4','avi'] # + ['jpg','jpeg', 'bmp', 'png'] +
            self.fileList = [fn for fn in os.listdir(path) if any(fn.lower().endswith(ext) for ext in support_files)] #os.listdir(path)
            if len(self.fileList) !=0 :
                # Notice that:
                # self.fileLst (element) & self.fileList (variable)
                self.totalFile = len(self.fileList)
                self.fileList.sort(key=natural_keys)

                slm = QStringListModel()
                slm.setStringList(self.fileList)
                
                self.fileLst.setModel(slm)
                self.fileLst.clicked.connect(self.selectFile) #function
                
                self.curFileIdx = 0
                self.thisFile = self.fileList[self.curFileIdx] ##
                self.fileNameLabel.setText(self.thisFile)
            else:
                slm = QStringListModel()
                self.fileLst.setModel(slm)
                QMessageBox.information(self,'Warning','Only support file format below: \n For Videos:    mp4, avi')
                self.reset()
                # 'Only support file format below: \n For images:   jpg, jpeg, bmp, png \n For Videos:    mp4, avi'

    def openFile(self):
        print('~~~~ OpenFile ~~~~')
        self.tracker.reset()
        path = '/'.join([self.thisFolder,self.thisFile])
        
        #if path.lower().endswith(('.jpg','.jpeg', '.bmp', '.png')): 
        #    self.loadImage(path)
        
        if path.lower().endswith(('.mp4', '.avi')):
            print("Load Video:", path)
            self.loadVideo(path)

    # ============ load the files ===========================
    
    def loadInitPage(self):
        self.frameViewer.init_state() ##
        self.frameViewer.setPhoto(QPixmap('UI/Movie.png'),init=True)
        self.frame_id = 0

    def loadImage(self,path):
        self.pixmap = QPixmap(path)
        self.frameViewer.init_state() ##
        self.frameViewer.setPhoto(self.pixmap )
        
    def loadVideo(self,path):
        # 1. load label
        # 2. load images (create thread and player)
        
        self.videoHandler = VideoBox(path)
        
        # ====== function connect ======
        # 1. video widget
        # 2. slider, timeinfo
        # ==============================
        
        self.total_frame = self.videoHandler.total_frame
        print('Totally frame:', self.total_frame)

        self.pIntvalidator.setRange(0, self.total_frame-1) ###
        
        self.playerSlider.setRange(0, self.total_frame-1) ###
        self.playerSlider.setTickPosition(QSlider.TicksBothSides)
        self.playerSlider.setTickInterval(int(self.total_frame*0.05))

        self.showTimeInfo.setText(str(self.frame_id) +" / "+str(self.total_frame))
        self.videoHandler.timer.timeSignal.signal[str].connect(self.show_video_images)

        self.videoHandler.playCapture.open(self.videoHandler.video_url)
        # ==========================================
        width = int(self.videoHandler.playCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.videoHandler.playCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resol = [width, height]
        fps = int(self.videoHandler.playCapture.get(cv2.CAP_PROP_FPS))
        # ==========================================
        self.show_video_images(init=True) ###
        self.videoHandler.playCapture.release()
        #print(self.thisFolder,self.thisFile)
        
        filepath = '/'.join([self.thisFolder,self.thisFile])
        
        self.tracker.openVideo(filepath)
        data = self.tracker.getData()
        if (int(data['Total frame']) != self.total_frame) or (data['Resolution'] != resol) or (data['FPS'] !=fps):
            QMessageBox.information(self,'Warning',"The total frame or Resolution Mismatch! Is this the correct label file?")

        if len(data['Event interval']) ==2:
            self.start = int(data['Event interval'][0])
            self.end = int(data['Event interval'][1])
            # check for label in interval
            
        self.showStartLabel.setText(str(self.start))
        self.showEndLabel.setText(str(self.end))
        self.framCombo.setEnabled(True)
        self.framCombo.setCurrentIndex(0)
        self.showFramLab.setText('')
        self.showFramLab_2.setText('')

    # ========== video control ===================    
    
    def slider_set_frameId(self, val):
        # set frame index
        if self.frameViewer is not None and self.videoHandler is not None:
            self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.show_video_images()

    def input_set_frameId(self):
        # set frame index
        if self.frameViewer is not None and self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                inputNumber = self.inFrameId.text()
                if inputNumber.isdigit():
                    val = int(inputNumber)
                    self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
                    self.show_video_images()
        self.inFrameId.clear()

    # -----------------------------------------------
    def prevFrame(self):
        if self.frameViewer is not None and self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                val = max(self.frame_id-1, 0 )
                self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.show_video_images()
            
    def nextFrame(self):
        if self.frameViewer is not None and self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                val = min(self.frame_id+1, self.total_frame-1 )
                self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.show_video_images()

    def reloadFrame(self):
        if self.frameViewer is not None and self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                val = self.frame_id 
                self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.show_video_images(reload=True)
            
    def saveCurPage(self):
        if self.frameViewer is not None and self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                val = self.frame_id 
                self.videoHandler.playCapture.set(cv2.CAP_PROP_POS_FRAMES, val)
            self.show_video_images()
            
    # ===== Set Event Interval ===================
    
    def setStart(self):
        fid = max(0, self.frame_id )
        self.start = fid
        self.showStartLabel.setText(str(self.start))
        if self.start >= self.end:
            self.end = 0
            self.tracker.recordData.data['Event interval'] = []
        self.showEndLabel.setText(str(self.end))
        
    def setEnd(self):
        fid = min(self.total_frame-1, self.frame_id )
        if self.start < fid:
            self.end = fid
        else:
            self.end = 0
            
        if self.start < self.end and self.end != 0:
            self.tracker.recordData.data['Event interval'] = [self.start, self.end]
            print("Interval Saved", [self.start, self.end])
        self.showEndLabel.setText(str(self.end))
    
    def setRangeLabel(self):
         if self.start < self.end and self.end != 0:
            # but no obj
            askReply = QMessageBox.question(self, 'Remove Obj Between Time', "Are you sure to clean outer interval data?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if askReply == QMessageBox.Yes:
                interv = self.tracker.recordData.data['Event interval']
                self.tracker.recordData.setDataBetween(interv[0],interv[1])
                
    # ============================================================================
    def resetObjClass(self):
        if self.pixmap is not None and self.thisframe is not None:
            _, classDic, _, _ = self.getFrameAllInfo()
            if len(classDic)>0:
                print(classDic)
                self.tracker.recordData.setObjClass( classDic )
    
    def setObjAttrAfter(self):
        if self.pixmap is not None and self.thisframe is not None:
            fid = self.frame_id
            objLst, _, _, stateFlagDic = self.getFrameAllInfo()
            self.tracker.recordData.setBatchObjAttriAfter( fid, objLst, stateFlagDic )
            
    def clrObjAfter(self):
        if self.pixmap is not None:
            askReply = QMessageBox.question(self, 'Remove Obj Over Time', "Clean All Objects After?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if askReply == QMessageBox.Yes:
                self.tracker.recordData.cleanDataAfter(self.frame_id)
                self.clrBbx()
                self.reloadFrame()
               
    def removeObjAfter(self):
        if self.pixmap is not None:
            seleNamLst, _, _ , _= self.getFrameSeleInfo() # first, check for pause  ??
            if len(seleNamLst) >0:
                # Ask for remove 
                askReply = QMessageBox.question(self, 'Remove Obj Over Time', "Delete "+str(len(seleNamLst))+" Selected Items?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if askReply == QMessageBox.Yes:
                    self.tracker.recordData.removeObjAfter(self.frame_id, seleNamLst )
                    self.clrBbx()
                    self.reloadFrame()
                # render again ???
            else:
                QMessageBox.information(self,'Warning',"No selected item !")
                     
    def saveEdLabel(self, fid, reload=False, init=False): # save before clear scene
        if self.pixmap is not None and self.thisframe is not None:
            
            objLst, classDic, rectDic, stateFlagDic = self.getFrameAllInfo()
            framLab = str(self.framCombo.currentText())
            
            if not init: #len(objLst) >= 0 :
                
                objDict = {}
                newname = []
                newClassMap = {}
                for name in objLst:
                    bbx = rectDic[name]
                    classNam = classDic[name]
                    stateFlag = stateFlagDic[name]
                    objDict.update({name : { "class": classNam , "bbx": bbx , "event state": bool(stateFlag)}})
                
                self.tracker.recordData.appendFrameType(fid, framLab )
                    
                if not reload :
                    self.tracker.recordData.itemNameSet.update(objLst)
                    self.tracker.recordData.ObjClassMap.update(objDict)
                    self.tracker.recordData.appendFrameData(fid, objDict)   #.data['Object bounding box'].update({str(fid): objDict } ) #####
                    
                    #print('==========>',self.frame_id, self.prev_frame_id )
            
    def show_video_images(self, reload=False, init= False):
        
        if self.videoHandler is not None:
            if self.videoHandler.playCapture.isOpened():
                
                # 1. read frame
                self.frame_id = int(self.videoHandler.playCapture.get(cv2.CAP_PROP_POS_FRAMES ))
                #print(self.frame_id) # current frame id
                self.playerSlider.setValue(self.frame_id)
                self.showTimeInfo.setText(str(self.frame_id) +" / "+str(self.total_frame))
                #print(self.playerSlider.value())

                if self.frame_id == self.total_frame-1:
                    self.videoHandler.status = VideoBox.STATUS_INIT
                    self.videoHandler.timer.stop() 
                    self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                    self.ply_pausBtn.setText("Play")
                else:
                    success, frame = self.videoHandler.playCapture.read()
                    
                    # 2. set frame into window
                    if success:
                        height, width = frame.shape[:2]

                        if frame.ndim == 3:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        elif frame.ndim == 2:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                        self.thisframe = rgb
                        temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                        self.pixmap = QPixmap.fromImage(temp_image)

                        # save previous label then clear
                       
                        self.saveEdLabel(self.prev_frame_id, reload, init)
                        self.clrBbx()
                        cv2.destroyAllWindows()
                        self.frameViewer.setPhoto(self.pixmap)
                        # load label
                        self.renderLabel(self.frame_id)
                        # save Label ????
                        self.prev_frame_id = self.frame_id ###
                        
                    else:
                        print("Read failed, no frame data")
                        success, frame = self.videoHandler.playCapture.read()
                        if not success and self.videoHandler.video_type is VideoBox.VIDEO_TYPE_OFFLINE:
                            print("Play finished")  
                            self.reset()
                            self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
                            self.ply_pausBtn.setText("Stop")
                        #return
                
            else:
                print("Open file or capturing device error, init again")
                self.reset()

    #--------------------------------------------------------------------
    def switch_video(self):
        if self.videoHandler is not None:
            if self.videoHandler.video_url == "" or self.videoHandler.video_url is None:
                print("None---")
                return
            if self.videoHandler.status is VideoBox.STATUS_INIT:
                print("---Video Initialize---") # ,self.videoHandler.status
                
                self.videoHandler.playCapture.open(self.videoHandler.video_url)
                self.videoHandler.timer.start()
                self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.ply_pausBtn.setText("Pause")
                
            elif self.videoHandler.status is VideoBox.STATUS_PLAYING:
                print("---Playing---") # ,self.videoHandler.status
                self.videoHandler.timer.stop()
                if self.videoHandler.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                    self.videoHandler.playCapture.release()    
                self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.ply_pausBtn.setText("Play")
                
            elif self.videoHandler.status is VideoBox.STATUS_PAUSE:
                print("---Pause---")  # ,self.videoHandler.status
                
                if self.videoHandler.video_type is VideoBox.VIDEO_TYPE_REAL_TIME:
                    self.videoHandler.playCapture.open(self.videoHandler.video_url)
                    
                self.videoHandler.timer.start()
                self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.ply_pausBtn.setText("Pause")
            
            
            self.videoHandler.status = (VideoBox.STATUS_PLAYING,
                           VideoBox.STATUS_PAUSE,
                           VideoBox.STATUS_PLAYING)[self.videoHandler.status]

    # ======= output part ==========================================================
    def saveFileDialog(self, defaultFile):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save label file",defaultFile,"Text Files (*.json)", options=options)
        if fileName:
            print(fileName)
        return fileName
      
    def saveLabel(self): 
        # ask to confirm ???
        if self.pixmap is not None:
            self.saveEdLabel(fid = self.frame_id, reload = False)
            data = self.tracker.recordData.data
            bbxRange = []
            if 'Object bounding box' in data:
                dataKey = data['Object bounding box'].keys()
                if len(dataKey) >0:
                    fids = [int(i) for i in dataKey]
                    bbxRange = [min(fids), max(fids)] 
                    
            QMessageBox.information(self,'Check for Information',
               """Event Interval : {0} \nBounding box Exist Range: {1}""".format( 
                          str(data['Event interval']),
                          str(bbxRange)
                          ))
              
            defaultPath = os.path.join(self.thisFolder,self.thisFile)
            jsonFile = defaultPath.rsplit( ".", 1 )[0] +'.json' 
            fileNam = self.saveFileDialog(jsonFile)
            if fileNam: 
                self.exportBtn.setEnabled(False)
                self.tracker.saveLabel(fileNam)
                self.exportBtn.setEnabled(True)
                QMessageBox.information(self,'Success',"Save file successfully !!")
            
    # --------------------------------------------------               
    def initVar(self):
        self.videoHandler = None
        self.pixmap = None
        self.thisFile = ''
        self.totalFile = 0

        self.frame_id = 0
        self.prev_frame_id = 0
        self.total_frame = 0
        self.start = 0
        self.end = 0
        self.tracker.recordData.resetData()
        self.tmpNamLst = []
        self.bbxMapping = {}
        self.pIntvalidator.setRange(0, 0)
        self.showStartLabel.setText(str(self.start))
        self.showEndLabel.setText(str(self.end))

        self.showFramLab.setText('No File Opened')
        self.showFramLab_2.setText('No File Opened')
        self.framCombo.setCurrentIndex(0)
        self.framCombo.setEnabled(False)
        
        self.showTimeInfo.setText('No File Opened')
        self.fileNameLabel.setText('No Supported files')
        
        self.frameViewer._scene.newItem = False
        self.creLabelBtn.setText("Create")
        
        
    def reset(self):
        if self.videoHandler is not None:
            self.videoHandler.reset()
            print('File open:',self.videoHandler.playCapture.isOpened())
            
            self.loadInitPage()
            self.initVar()
            self.tracker.reset()
            self.playerSlider.setRange(0, 0)
            self.playerSlider.setTickPosition(QSlider.NoTicks)
            self.ply_pausBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.ply_pausBtn.setText("Play")
            
    #--------------------------------------------------------------------
            
if __name__ == "__main__":
    import sys
    
    # ==========================
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    
    splash_pix = QPixmap("UI/logo.png")
    splash = QSplashScreen(splash_pix,Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.showMessage("Loading...", Qt.AlignHCenter | Qt.AlignBottom, Qt.white)
    splash.show()
    
    app.processEvents()
    # ==========================
    window = LabelTool()
    splash.finish(window)
    window.show()
    sys.exit(app.exec_())

# pyuic5 -x main.ui -o output.py
