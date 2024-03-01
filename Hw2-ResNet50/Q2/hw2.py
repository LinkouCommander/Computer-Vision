from PyQt5 import QtCore, QtGui, QtWidgets
import os
from scipy import misc
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv2 import findChessboardCorners
from PIL import Image
import time
import glob


img1 = cv2.imread('1.bmp')
img2 = cv2.imread('2.bmp')
img3 = cv2.imread('3.bmp')
img4 = cv2.imread('4.bmp')
img5 = cv2.imread('5.bmp')
img6 = cv2.imread('6.bmp')
img7 = cv2.imread('7.bmp')
img8 = cv2.imread('8.bmp')
img9 = cv2.imread('9.bmp')
img10 = cv2.imread('9_10.bmp')
img11 = cv2.imread('9_11.bmp')
img12 = cv2.imread('9_12.bmp')
img13 = cv2.imread('9_13.bmp')
img14 = cv2.imread('9_14.bmp')
img15 = cv2.imread('9_15.bmp')
files = [img1,img2,img3,img4,img5,img6,img7,img8,img9,img10,img11,img12,img13,img14,img15]

w = 11
h = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp=np.zeros((w*h, 3), np.float32)
objp[:, :2]=np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints=[]
imgpoints=[]
imgs = glob.glob('*.bmp')
for fname in files:
    img = fname
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def click21():
    for fname in imgs:
        img_1 = cv2.imread(fname)
        gray1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        ret, corners1 = cv2.findChessboardCorners(gray1, (w,h), None)
        if ret:
            objpoints.append(objp)
            corners1_1 = cv2.cornerSubPix(gray1,corners1, (11,11), (-1,-1), criteria)
            imgpoints.append(corners1)
            fn1 = cv2.drawChessboardCorners(img_1, (11, 8), corners1_1, ret)
            fn2 = cv2.resize(fn1, (512, 512)) 
            cv2.imshow("img", fn2)
            cv2.waitKey(500)

def click22():
    print("Intrinsic Matrix is: ")
    print(mtx)

def click24():
    print("Distortion Matrix is: ")
    print(dist)

def click25():
    for fname in files:
        img_2 = fname
        h, w = img_2.shape[:2]
        newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),2,(w,h))
        dst = cv2.undistort(img_2, mtx, dist, None, newcameramtx)
        dst2 = cv2.resize(dst, (512, 512))
        img_2_1 = cv2.resize(img_2, (512, 512))
        cv2.imshow("Distorted image",img_2_1)
        cv2.imshow("Undistorted image",dst2)
        cv2.waitKey(500)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(240, 311)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 191, 251))
        self.groupBox.setObjectName("groupBox")
        self.pushButton21 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton21.setGeometry(QtCore.QRect(20, 30, 151, 23))
        self.pushButton21.setObjectName("pushButton21")
        self.pushButton22 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton22.setGeometry(QtCore.QRect(20, 60, 151, 23))
        self.pushButton22.setObjectName("pushButton22")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 90, 151, 81))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 20, 81, 16))
        self.label.setObjectName("label")
        self.lineEdit2 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit2.setGeometry(QtCore.QRect(80, 20, 61, 20))
        self.lineEdit2.setObjectName("lineEdit2")
        self.pushButton23 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton23.setGeometry(QtCore.QRect(10, 50, 131, 21))
        self.pushButton23.setObjectName("pushButton23")
        self.pushButton24 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton24.setGeometry(QtCore.QRect(20, 180, 151, 23))
        self.pushButton24.setObjectName("pushButton24")
        self.pushButton25 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton25.setGeometry(QtCore.QRect(20, 210, 151, 23))
        self.pushButton25.setObjectName("pushButton25")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 429, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        def click23():
            w = 11
            h = 8
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp=np.zeros((w*h, 3), np.float32)
            objp[:, :2]=np.mgrid[0:w, 0:h].T.reshape(-1, 2)
            objpoints=[]
            imgpoints=[]
            get_input = self.lineEdit2.text()
            go_input = int(get_input)
            i = go_input - 1
            img = files[i]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (w,h), None)
            if ret == True:
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)
                imgpoints.append(corners)
            ret, mtx, dist, r, t = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)            
            r1 = np.concatenate(r)
            R = cv2.Rodrigues(r1)[0]
            T = np.concatenate(t)
            extrin = np.hstack([R, T])
            print("Extrinsic Matrix of ",go_input, ".bmp is: ",sep="")
            print(extrin)


        self.pushButton21.clicked.connect(click21)
        self.pushButton22.clicked.connect(click22)
        self.pushButton23.clicked.connect(click23)
        self.pushButton24.clicked.connect(click24)
        self.pushButton25.clicked.connect(click25)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "2.Calibration"))
        self.pushButton21.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.pushButton22.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.label.setText(_translate("MainWindow", "Select image:"))
        self.pushButton23.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.pushButton24.setText(_translate("MainWindow", "2.4 Find Distortion"))
        self.pushButton25.setText(_translate("MainWindow", "2.5 Show result"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
#####################