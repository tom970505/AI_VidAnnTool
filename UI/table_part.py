

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont,QColor, QBrush
from PyQt5.QtWidgets import QWidget,QHBoxLayout,QTableWidget,QPushButton,QApplication,QVBoxLayout,QTableWidgetItem,QCheckBox,QComboBox,QAbstractItemView,QHeaderView,QLabel,QFrame
#from PyQt5.QtWidgets import *
#from PyQt5.QtGui import *
#from PyQt5.QtCore import *
import sys
#from PyQt5 import QtWidgets
#from faker import Factory


#from faker import Factory


class MyTable(QTableWidget):
    def __init__(self,parent=None):
        super(MyTable, self).__init__(parent)
        self.init_state()
        
    def init_state(self):
        self.setColumnCount(2)   ##设置列数
        self.headers = ['obj name','event flag'] #,'obj attr2', 'obj attr3']
        self.setHorizontalHeaderLabels(self.headers)
        #self.setSelectionMode(QAbstractItemView.SingleSelection) 
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.verticalHeader().setVisible(False)
        self.setColumnWidth(0,120)
        self.setColumnWidth(1,80)
        
    def setEditable(self,flag):
        if flag:
            self.setEditTriggers(QAbstractItemView.DoubleClicked) # DoubleClicked # AllEditTriggers
        else:
            self.setEditTriggers(QAbstractItemView.NoEditTriggers) 

    def setRowColor(self, rowIndex, color):
        for j in range(self.columnCount()):
            try:
                self.item(rowIndex, j).setBackground(QBrush(color))
            except:
                pass
            
    def addRow(self, objname, color, objAttri ):
        # self.resizeColumnsToContents()
        # ====== append a row ============
        row = self.rowCount()
        self.setRowCount(row + 1)
        # ====== set data ================
        # 1. obj name
        objItem = QTableWidgetItem(objname)
        objItem.setTextAlignment(Qt.AlignCenter)
        
        #----------------------------------
        # 2. attribute
        """
        cell_widget = QWidget()
        chk_bx = QCheckBox()
        chk_bx.setChecked(objAttri)
        lay_out = QHBoxLayout(cell_widget)
        lay_out.addWidget(chk_bx)
        lay_out.setAlignment(Qt.AlignCenter)
        lay_out.setContentsMargins(0,0,0,0)
        cell_widget.setLayout(lay_out)

        #mycom = QComboBox()
        #mycom.addItems(["", "dd", "kk"])
        """
        checkIt = QTableWidgetItem()
        if objAttri:
            checkIt.setCheckState(Qt.Checked)
        else:
            checkIt.setCheckState(Qt.Unchecked)
        
        # =================================
        self.setItem(row,0,objItem)
        self.setItem(row,1,checkIt)
        #self.setCellWidget(row,1,cell_widget)
        self.setRowColor(row, color)
        return objItem

    def getRowIdxbyItem(self, cellItem): # {bbx : cell} -> row select
        row = self.row(cellItem)
        return row
    
    def getDataByRow(self, row):
        if row is not None:
            if self.item(row, 0) is not None:
                #attriFlag = False
                objname = self.item(row, 0).text()
                attriFlag = self.item(row, 1).checkState()
                """
                ckbox = self.cellWidget(row, 1).findChild(type(QtGui.QCheckBox()))
                if ckbox.isChecked():
                    attriFlag = True
                """
                return objname, bool(attriFlag)     
                        
    #def removRow(self, row):
    #    self.removeRow(row)
    #def selectRow(self, row):
    #    self.selectRow(row)
        
    def clean(self):
        self.setRowCount(0)
        #self.clearContents()
        

# https://blog.csdn.net/u014115390/article/details/82929719   
 
