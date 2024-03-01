from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import  plot_model
from tensorflow.python.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import load_model

from keras.layers import Dense, Flatten, Dropout
from keras.models import Model


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 311)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(220, 20, 191, 191))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton51 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton51.setGeometry(QtCore.QRect(20, 30, 151, 23))
        self.pushButton51.setObjectName("pushButton51")
        self.pushButton52 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton52.setGeometry(QtCore.QRect(20, 60, 151, 23))
        self.pushButton52.setObjectName("pushButton52")
        self.pushButton53 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton53.setGeometry(QtCore.QRect(20, 90, 151, 23))
        self.pushButton53.setObjectName("pushButton53")
        self.pushButton54 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton54.setGeometry(QtCore.QRect(20, 150, 151, 23))
        self.pushButton54.setObjectName("pushButton54")
        self.lineEdit5 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit5.setGeometry(QtCore.QRect(20, 120, 151, 20))
        self.lineEdit5.setObjectName("lineEdit5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 429, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        def click51():
            ishape = 64
            model_resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(ishape,ishape, 3))
            for layers in model_resnet50.layers:
                layers.trainable = False
            model = Flatten()(model_resnet50.output)
            model = Dense(4096, activation='relu',name='fc1')(model)
            model = Dropout(0.5)(model)
            model = Dense(4096, activation='relu',name='fc2')(model)
            model = Dropout(0.5)(model)
            model = Dense(10, activation='softmax',name='prediction')(model)
            model_resnet50_pretrain = Model(inputs=model_resnet50.input, outputs=model, name='ResNet50')
            model_resnet50_pretrain.summary()
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click52():
            img2 = cv2.imread('tensorboard.jpg')
            cv2.imshow('TensorBoard', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        def click53():
            datapath1 = 'sample/test/dogs/'
            datapath2 = 'sample/test/cats/'
            rare = '.jpg'
            get_input = self.lineEdit5.text()
            the_input = int(get_input)
            choose = random.randint(0, 1)
            print(choose)
            if choose == 1:
                datapath = datapath1
            else:
                datapath = datapath2

            img_datapath = datapath + str(the_input +3760) + rare
            img = image.load_img(os.path.join(img_datapath),
                                 target_size=(224, 224))

            net = load_model('model-resnet50-final.h5')

            cls_list = ['Class:cat', 'Class:dog']

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            pred = net.predict(x)[0]

            if pred[0] > pred[1]:
                img_title = cls_list[0]
            else:
                img_title = cls_list[1]

            plt.figure('Randomly Select')
            plt.imshow(img)
            plt.axis('on')
            plt.title(img_title)
            plt.show()

        def click54():
            img = cv2.imread('random-erasing.jpg')
            cv2.imshow("code of augmentation method", img)
            plt.figure('comparison table of accuracy')
            plt.bar(['Before Random-Erasing', 'After Random-Erasing'],
                    [86.15, 90.13], 0.5)
            plt.title('Random-Erasing augmentation comparison')
            plt.ylabel('Accuracy')
            plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='y')
            plt.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()



        self.pushButton51.clicked.connect(click51)
        self.pushButton52.clicked.connect(click52)
        self.pushButton53.clicked.connect(click53)
        self.pushButton54.clicked.connect(click54)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "5.ResNet50"))
        self.pushButton51.setText(_translate("MainWindow", "5.1 Show Model Structure"))
        self.pushButton52.setText(_translate("MainWindow", "5.2 Show TensorBoard"))
        self.pushButton53.setText(_translate("MainWindow", "5.3 Test"))
        self.pushButton54.setText(_translate("MainWindow", "5.4 Data Augmentation"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
#####################
