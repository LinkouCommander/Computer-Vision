from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(701, 425)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 50, 171, 31))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(180, 50, 171, 31))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(350, 50, 171, 31))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(520, 50, 171, 31))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.pushButton11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton11.setGeometry(QtCore.QRect(30, 90, 131, 41))
        self.pushButton11.setObjectName("pushButton11")
        self.pushButton12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton12.setGeometry(QtCore.QRect(30, 140, 131, 41))
        self.pushButton12.setObjectName("pushButton12")
        self.pushButton13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton13.setGeometry(QtCore.QRect(30, 190, 131, 41))
        self.pushButton13.setObjectName("pushButton13")
        self.pushButton14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton14.setGeometry(QtCore.QRect(30, 240, 131, 41))
        self.pushButton14.setObjectName("pushButton14")
        self.pushButton21 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton21.setGeometry(QtCore.QRect(200, 90, 131, 41))
        self.pushButton21.setObjectName("pushButton21")
        self.pushButton22 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton22.setGeometry(QtCore.QRect(200, 140, 131, 41))
        self.pushButton22.setObjectName("pushButton22")
        self.pushButton23 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton23.setGeometry(QtCore.QRect(200, 190, 131, 41))
        self.pushButton23.setObjectName("pushButton23")
        self.pushButton31 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton31.setGeometry(QtCore.QRect(370, 90, 131, 41))
        self.pushButton31.setObjectName("pushButton31")
        self.pushButton32 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton32.setGeometry(QtCore.QRect(370, 140, 131, 41))
        self.pushButton32.setObjectName("pushButton32")
        self.pushButton33 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton33.setGeometry(QtCore.QRect(370, 190, 131, 41))
        self.pushButton33.setObjectName("pushButton33")
        self.pushButton34 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton34.setGeometry(QtCore.QRect(370, 240, 131, 41))
        self.pushButton34.setObjectName("pushButton34")
        self.pushButton42 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton42.setGeometry(QtCore.QRect(540, 140, 131, 41))
        self.pushButton42.setObjectName("pushButton42")
        self.pushButton41 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton41.setGeometry(QtCore.QRect(540, 90, 131, 41))
        self.pushButton41.setObjectName("pushButton41")
        self.pushButton43 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton43.setGeometry(QtCore.QRect(540, 190, 131, 41))
        self.pushButton43.setObjectName("pushButton43")
        self.pushButton44 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton44.setGeometry(QtCore.QRect(540, 240, 131, 41))
        self.pushButton44.setObjectName("pushButton44")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 701, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.pushButton11.clicked.connect(clicked11)
        self.pushButton12.clicked.connect(clicked12)
        self.pushButton13.clicked.connect(clicked13)
        self.pushButton14.clicked.connect(clicked14)
        self.pushButton21.clicked.connect(clicked21)
        self.pushButton22.clicked.connect(clicked22)
        self.pushButton23.clicked.connect(clicked23)
        self.pushButton31.clicked.connect(clicked31)
        self.pushButton32.clicked.connect(clicked32)
        self.pushButton33.clicked.connect(clicked33)
        self.pushButton34.clicked.connect(clicked34)
        self.pushButton41.clicked.connect(clicked41)
        self.pushButton42.clicked.connect(clicked42)
        self.pushButton43.clicked.connect(clicked43)
        self.pushButton44.clicked.connect(clicked44)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">1. Image Processing</span></p></body></html>"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">2. Image Smoothing</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">3. Edge Detection</span></p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">4. Transforms</span></p></body></html>"))
        self.pushButton11.setText(_translate("MainWindow", "1.1 Load Image File"))
        self.pushButton12.setText(_translate("MainWindow", "1.2 Color Separation"))
        self.pushButton13.setText(_translate("MainWindow", "1.3 Color Transformation "))
        self.pushButton14.setText(_translate("MainWindow", "1.4 Blending"))
        self.pushButton21.setText(_translate("MainWindow", "2.1 Gaussian blur"))
        self.pushButton22.setText(_translate("MainWindow", "2.2 Bilateral filter"))
        self.pushButton23.setText(_translate("MainWindow", "2.3 Median filter"))
        self.pushButton31.setText(_translate("MainWindow", "3.1 Gaussian Blur "))
        self.pushButton32.setText(_translate("MainWindow", "3.2 Sobel X "))
        self.pushButton33.setText(_translate("MainWindow", "3.3 Sobel Y "))
        self.pushButton34.setText(_translate("MainWindow", "3.4 Magnitude "))
        self.pushButton42.setText(_translate("MainWindow", "4.2 Translation"))
        self.pushButton41.setText(_translate("MainWindow", "4.1 Resize"))
        self.pushButton43.setText(_translate("MainWindow", "4.3 Rotation, Scaling"))
        self.pushButton44.setText(_translate("MainWindow", "4.4 Shearing"))

img1 = cv2.imread(
    'Sun.jpg')
img1_1 = cv2.imread(
    'Dog_Strong.jpg')
img1_2 = cv2.imread(
    'Dog_Weak.jpg')
img2_1 = cv2.imread(
    'Lenna_whiteNoise.jpg')
img2_2 = cv2.imread(
    'Lenna_pepperSalt.jpg')
img3 = cv2.imread(
    'House.jpg')
img4 = cv2.imread(
    'SQUARE-01.png')


def clicked11():
    cv2.imshow('image', img1)
    height, width = img1.shape[:2]
    print('Height = ', height)
    print('Width = ', width)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked12():
    cv2.imshow('image', img1)
    b, g, r = cv2.split(img1)
    zeros = np.zeros(b.shape, np.uint8)
    blueBGR = cv2.merge((b, zeros, zeros))
    greenBGR = cv2.merge((zeros, g, zeros))
    redBGR = cv2.merge((zeros, zeros, r))
    cv2.imshow('Blue BGR', blueBGR)
    cv2.imshow('Green BGR', greenBGR)
    cv2.imshow('Red BGR', redBGR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked13():
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    b = img1[:, :, 0]
    g = img1[:, :, 1]
    r = img1[:, :, 2]
    gray2 = 1/3 * b + 1/3 * g + 1/3 * r
    img_gray2 = gray2.astype(np.uint8)
    cv2.imshow('OpenCV function', img_gray1)
    cv2.imshow('Average weighted', img_gray2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked14():
    cv2.namedWindow('image')
    cv2.createTrackbar('Blending', 'image', 0, 255, updateblending)
    cv2.setTrackbarPos('Blending', 'image', 128)
    while (True):
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

def updateblending(x):
    blending = cv2.getTrackbarPos('Blending', 'image')
    dst = cv2.addWeighted(img1_1, blending/255, img1_2, 1-blending/255, 0)
    cv2.imshow('image', dst)

##############################

def clicked21():
    cv2.imshow('image', img2_1)
    gauss = cv2.GaussianBlur(img2_1, (3, 3), 0)
    cv2.imshow('Gaussian Blur', gauss)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clicked22():
    cv2.imshow('image', img2_1)
    bilateral = cv2.bilateralFilter(img2_1, 9, 90, 90)
    cv2.imshow('Bilateral Filter', bilateral)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked23():
    cv2.imshow('image', img2_2)
    median3 = cv2.medianBlur(img2_2, 3)
    median5 = cv2.medianBlur(img2_2, 5)
    cv2.imshow('Median Blur3', median3)
    cv2.imshow('Median Blur5', median5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##############################

imgg = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
x, y = np.mgrid[-1:2, -1:2]
gaussian_kernel = np.exp(-(x**2+y**2))
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
gaussian = cv2.filter2D(imgg, -1, gaussian_kernel)

x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobelx = cv2.filter2D(gaussian, -1, x)
y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobely = cv2.filter2D(gaussian, -1, y)

def clicked31():
    cv2.imshow('img', img3)
    cv2.imshow('Gray Scale', imgg)
    cv2.imshow('Gaussian', gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clicked32():
    cv2.imshow('image', imgg)
    cv2.imshow('SobelX', sobelx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clicked33():
    cv2.imshow('image', imgg)
    cv2.imshow('SobelY', sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def clicked34(self):
    cv2.imshow('SobelX', sobelx)
    cv2.imshow('SobelY', sobely)
    m = cv2.add(sobelx, sobely)
    cv2.imshow('Magnitude', m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##############################

def clicked41():
    image = cv2.resize(img4, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imshow('Resize', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked42():

    image = cv2.resize(img4, (256, 256), interpolation=cv2.INTER_AREA)
    matShift = np.float32([[1,0,0],[0,1,60]])
    trans = cv2.warpAffine(image,matShift,(400,400))
    cv2.imshow('Translation', trans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def clicked43():
    image = cv2.resize(img4,(0, 0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    matShift = np.float32([[1,0,100],[0,1,60]])
    trans = cv2.warpAffine(image,matShift,(400,400))
    img_info=image.shape
    image_weight=img_info[1]
    rota = cv2.getRotationMatrix2D((0,image_weight),10,1)
    r_s = cv2.warpAffine(trans,rota,(400,300))
    cv2.imshow('Rotation & Scaling', r_s)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clicked44():
    image = cv2.resize(img4, (128, 128), interpolation=cv2.INTER_AREA)
    matShift = np.float32([[1,0,100],[0,1,50]])
    trans = cv2.warpAffine(image,matShift,(400,400))
    mat_src=np.float32([[50,50],[200,50],[50,200]])
    mat_dst=np.float32([[10,100],[200,50],[100,250]])
    mat_Affine=cv2.getAffineTransform(mat_src,mat_dst)
    dst=cv2.warpAffine(trans,mat_Affine,(400,300))
    cv2.imshow('Shearing',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())