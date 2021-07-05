from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsPixmapItem, QFrame, QStyle
from PyQt5.QtGui import QBrush, QPen, QPainterPath, QPainter, QColor, QPixmap, QTransform
from PyQt5.QtCore import pyqtSignal, Qt, QRectF, QPointF, QStringListModel

class BoxItem(QGraphicsRectItem):
    
    rectChanged = pyqtSignal(QRectF)
    
    handleTopLeft = 1
    handleTopRight = 2
    handleBottomLeft = 3
    handleBottomRight = 4
    
    # ---- boundingRect part -----
    handleSpace = -5 # point offset from bbx         # -4
    handleSize = 7  # size of the small point # 20
    # ----------------------------
    # cursors dictionary mapping
    handleCursors = {
        handleTopLeft: Qt.SizeFDiagCursor,
        handleTopRight: Qt.SizeBDiagCursor,
        handleBottomLeft: Qt.SizeBDiagCursor,
        handleBottomRight: Qt.SizeFDiagCursor,
    }

    #itemPos = QtCore.pyqtSignal(QtCore.QRectF)
    
    def __init__(self, color=(0,225,225), style= Qt.DashLine, # scene
                 rect=None, flag=None , useTable=None, bindCellit=None ,auto=False, assist=None,  matrix= QTransform()):
        super(BoxItem, self).__init__() ####
        
        # ---- Init state -------------
        self.handles = {}
        self.handleSelected = None # trace selected point
        self.mousePressPos = None  # Press : mouseEvent.pos()
        self.mousePressRect = None # Press : boundingRect()
        
        # ---- Settings --------------
        self.setAcceptHoverEvents(True)
        
        self.setFlags(QGraphicsItem.ItemIsSelectable|
                      QGraphicsItem.ItemIsMovable|
                      QGraphicsItem.ItemIsFocusable|
                      QGraphicsItem.ItemSendsGeometryChanges|
                      QGraphicsItem.ItemSendsScenePositionChanges)

        # ---- Create bbx --------------
        self.flag = flag   # on image
        self.col = color
        self.setRect(rect) ###
        self.style = style
        #self.setPos(position)

        # ----
        self.cellitem = bindCellit
        self.useTable = useTable
        self.signalConn = assist
        # ---- Bbx Settings ------------
        self.setTransform(matrix)
        #scene.clearSelection()
        #scene.addItem(self)
        
        if not auto:
            self.setSelected(True) ### 
            #self.setFocus()       ####
        # ----------------------------
        
        self.updateHandlesPos()
        self.cnt =0
    # ==============================
    
    def setColor(self, color):
        self.col = color
        
    def getPosition(self):
        return self.sceneBoundingRect() 
    
    def updateHandlesPos(self):
        """
        Update current resize handles according to the shape size and position.
        """
        # ------- Appearance ----------------
        s = self.handleSize     # constant
        b = self.boundingRect() # coordinate of bbx points
        # ------- Draw Surrounding Points ---
        self.handles[self.handleTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.handles[self.handleTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.handles[self.handleBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.handles[self.handleBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)
        
    # QGraphicsItem.boundingRect() is abstract and must be overridden
    def boundingRect(self):
        """
        Returns the bounding rect of the shape (including the resize handles).
        """
        o = self.handleSize + self.handleSpace     # offset # self.rect()
        return self.rect().adjusted(0,0,0,0) 
    
    # QGraphicsItem.paint() is abstract and must be overridden
    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        
        painter.setBrush(QBrush(QColor( *self.col , 100)))              # 填滿: color : R G B alpha (紅)
        painter.setPen(QPen(Qt.black, 2.0, self.style))                 # 外框: color (灰) (127, 127, 127) | style (虛線)
        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(Qt.blue, 2.0, self.style))
        painter.drawRect(self.rect())                                 # 畫出外觀

        painter.setRenderHint(QPainter.Antialiasing)                  ## 反鋸齒
        painter.setBrush(QBrush(QColor(*self.col, 200)))             # 填滿: color (淺藍) (81,168,220)
        painter.setPen(QPen(Qt.black, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)) # 外框: color (黑)
        if option.state & QStyle.State_Selected:
            self.useTable.itemSelectionChanged.disconnect()
            painter.setPen(QPen(Qt.blue, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            table_row = self.useTable.row(self.cellitem) 
            self.useTable.selectRow(table_row)
            self.useTable.itemSelectionChanged.connect(self.signalConn)
            
            
        for handle, rect in self.handles.items():
            if self.handleSelected is None or handle == self.handleSelected:
                painter.drawRect(rect)                                    # 畫出小框框 (也可以改形狀)

    # =========================================
    # only handle interactive while mouse is pressed and moving
    def interactiveResize(self, mousePos):
        """
        Perform shape interactive resize.
        """
        if self.flag.isUnderMouse():
            # ----- 1. get bbx size before change ---------------
            offset = self.handleSize + self.handleSpace
            boundingRect = self.boundingRect()          
            rect = self.rect()
            # ----- 2. compute bbx position then resize bbx -----
            diff = QPointF(0, 0)
            
            # Call this function before changing the bounding rect 
            self.prepareGeometryChange()

            # change depends on select which cursor
            if self.handleSelected == self.handleTopLeft:

                fromX = self.mousePressRect.left()
                fromY = self.mousePressRect.top()
                toX = fromX + mousePos.x() - self.mousePressPos.x()
                toY = fromY + mousePos.y() - self.mousePressPos.y()
                # -----------------------
                diff.setX(toX - fromX)
                diff.setY(toY - fromY)
                # ------------------------
                boundingRect.setLeft(toX)
                boundingRect.setTop(toY)
                # -------------------------
                rect.setLeft(boundingRect.left() + offset)
                rect.setTop(boundingRect.top() + offset)
                self.setRect(rect)

            elif self.handleSelected == self.handleTopRight:

                fromX = self.mousePressRect.right()
                fromY = self.mousePressRect.top()
                toX = fromX + mousePos.x() - self.mousePressPos.x()
                toY = fromY + mousePos.y() - self.mousePressPos.y()
                # -------------------------
                diff.setX(toX - fromX)
                diff.setY(toY - fromY)
                # -------------------------
                boundingRect.setRight(toX)
                boundingRect.setTop(toY)
                # -------------------------
                rect.setRight(boundingRect.right() - offset)
                rect.setTop(boundingRect.top() + offset)
                self.setRect(rect)

            elif self.handleSelected == self.handleBottomLeft:

                fromX = self.mousePressRect.left()
                fromY = self.mousePressRect.bottom()
                toX = fromX + mousePos.x() - self.mousePressPos.x()
                toY = fromY + mousePos.y() - self.mousePressPos.y()
                # -------------------------
                diff.setX(toX - fromX)
                diff.setY(toY - fromY)
                # -------------------------
                boundingRect.setLeft(toX)
                boundingRect.setBottom(toY)
                # -------------------------
                rect.setLeft(boundingRect.left() + offset)
                rect.setBottom(boundingRect.bottom() - offset)
                self.setRect(rect)


            elif self.handleSelected == self.handleBottomRight:

                fromX = self.mousePressRect.right()
                fromY = self.mousePressRect.bottom()
                toX = fromX + mousePos.x() - self.mousePressPos.x()
                toY = fromY + mousePos.y() - self.mousePressPos.y()
                # -------------------------
                diff.setX(toX - fromX)
                diff.setY(toY - fromY)
                # -------------------------
                boundingRect.setRight(toX)
                boundingRect.setBottom(toY)
                # -------------------------
                rect.setRight(boundingRect.right() - offset)
                rect.setBottom(boundingRect.bottom() - offset)
                self.setRect(rect)

            # 3. -------- update and render bbx ----------------
        self.updateHandlesPos()
        
    # =========================================
    
    def handleAt(self, point):
        """
        Returns the resize handle below the given point.
        """
        for k, v, in self.handles.items():
            if v.contains(point):
                return k          # return which point is on
        return None

    def mousePressEvent(self, mouseEvent):
        """
        Executed when the mouse is pressed on the item.
        """
        self.handleSelected = self.handleAt(mouseEvent.pos())
        if self.handleSelected:
            self.mousePressPos = mouseEvent.pos()
            self.mousePressRect = self.boundingRect()
        super().mousePressEvent(mouseEvent)

    def mouseMoveEvent(self, mouseEvent):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """
       
        if self.handleSelected is not None:
            self.interactiveResize(mouseEvent.pos())  # only handle interactive when mouse press and move
        else:
            super().mouseMoveEvent(mouseEvent)
    
    def mouseReleaseEvent(self, mouseEvent):
        """
        Executed when the mouse is released from the item.
        """
        super().mouseReleaseEvent(mouseEvent)

        self.itemRectChange() ###
        # ----- reset state -----
        self.handleSelected = None
        self.mousePressPos = None
        self.mousePressRect = None
        #----- update render -----
        self.update()           
     
    def hoverMoveEvent(self, moveEvent): # setCursor
        """
        Executed when the mouse moves over the shape (NOT PRESSED).
        """
        if self.isSelected():
            handle = self.handleAt(moveEvent.pos()) # check if on point and return handle point
            cursor = Qt.ArrowCursor if handle is None else self.handleCursors[handle] 
            self.setCursor(cursor)
        super().hoverMoveEvent(moveEvent)

    def hoverLeaveEvent(self, moveEvent): # setCursor
        """
        Executed when the mouse leaves the shape (NOT PRESSED).
        """
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(moveEvent)

    # =========================================
    
class CustomScene(QGraphicsScene):

    def __init__(self, parent=None):
        super(CustomScene, self).__init__(parent)
        self.pen = QPen(Qt.black, 2, Qt.SolidLine)
        self.brush = QBrush(Qt.white, Qt.SolidPattern)
        self.parent = parent
        self.newItem = None
        self.startPos = None
        self.endPos = None

    def mousePressEvent(self, mouseEvent):
        if self.newItem and self.parent._photo.isUnderMouse():
            self.startPos = mouseEvent.scenePos()
        else:
            self.startPos = None
            super(CustomScene, self).mousePressEvent(mouseEvent)

    def mouseReleaseEvent(self, mouseEvent):
        if self.newItem and self.startPos is not None and self.parent._photo.isUnderMouse():
             
            self.endPos = mouseEvent.scenePos()
            xs = self.startPos.x()
            ys = self.startPos.y()
            xe = self.endPos.x()
            ye = self.endPos.y()
            #print('=================')

            rect = self.parent.mapFromScene(QRectF(xs, ys, xe-xs, ye-ys)).boundingRect()
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            # print(x,y,w,h)
            # ==== create New Bbx
            self.parent.createNewBbx(rect=QRectF(xs, ys, xe-xs, ye-ys),auto=False)
        else:
            self.startPos = None
            self.endPos = None
