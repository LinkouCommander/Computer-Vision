from PyQt5 import QtCore, QtWidgets
# import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import load_model
import random
from skimage.transform import resize
from tensorflow.keras.utils import  plot_model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def click51():

    def show_label(img_data, label_data):
        label_array = ["airplane", "automobile", "bird", "cat","deer", "dog", "frog", "horse", "ship", "truck"]
        fig = plt.gcf()
        for i in range(0, 9):
            the_random = random.randint(0, 9999)
            showfig = plt.subplot(3, 3, i + 1)
            showfig.imshow(img_data[the_random], cmap='binary')
            label = label_array[label_data[the_random][0]]
            showfig.set_title(label, fontsize=10)
            showfig.set_xticks([])
            showfig.set_yticks([])
        plt.show()
    show_label(x_test, y_test)

def click52():
    print("hyperparameters:")
    print("Batch Size: 32")
    print("Learning Rate: 0.001")
    print("Optimizer: SGD")

def click53():
    ishape = 64
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(ishape,ishape, 3))
    for layers in model_vgg.layers:
        layers.trainable = False
    model = Flatten()(model_vgg.output)
    model = Dense(4096, activation='relu',name='fc1')(model)
    model = Dropout(0.5)(model)
    model = Dense(4096, activation='relu',name='fc2')(model)
    model = Dropout(0.5)(model)
    model = Dense(10, activation='softmax',name='prediction')(model)
    model_vgg_cifar10_pretrain = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    model_vgg_cifar10_pretrain.summary()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def click54():
    img_accuracy = cv2.imread("loss_accuracy.png")
    cv2.imshow("Loss & Accuracy", img_accuracy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(461, 443)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 10, 421, 371))
        self.groupBox.setObjectName("groupBox")
        self.pushButton51 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton51.setGeometry(QtCore.QRect(20, 40, 381, 41))
        self.pushButton51.setObjectName("pushButton51")
        self.pushButton52 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton52.setGeometry(QtCore.QRect(20, 100, 381, 41))
        self.pushButton52.setObjectName("pushButton52")
        self.pushButton53 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton53.setGeometry(QtCore.QRect(20, 160, 381, 41))
        self.pushButton53.setObjectName("pushButton53")
        self.pushButton54 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton54.setGeometry(QtCore.QRect(20, 220, 381, 41))
        self.pushButton54.setObjectName("pushButton54")
        self.pushButton55 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton55.setGeometry(QtCore.QRect(20, 310, 381, 41))
        self.pushButton55.setObjectName("pushButton55")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(130, 280, 161, 20))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 461, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        def click55():
            label_array = ["plane", "car", "bird", "cat","deer", "dog", "frog", "horse", "ship", "truck"]
            model = load_model('transfer_cifar10.h5')
            get_input = self.lineEdit.text()
            the_input = int(get_input)

            (X_train, y_train), (X_test, y_test) = cifar10.load_data()

            class MainPredictImg(object):
                def __init__(self):
                    pass

                def pre(self):
                    b = the_input
                    pred_img = np.array(X_test[b])
                    pred_img = resize(pred_img, (64, 64))
                    pred_img = pred_img.reshape(-1, 64, 64, 3)

                    cv2.namedWindow("picture", 0)
                    cv2.resizeWindow("picture", 640, 480)
                    cv2.imshow("picture", X_test[b])

                    votes = [0.1, 0.2, 0.6, 0.1, 0.2, 0.6, 0.1, 0.2, 0.6, 0.9]
                    prediction = model.predict(pred_img)
                    labels = ['airplane', 'automobile', 'bird', 'cat',
                              'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                    Final_prediction = [result.argmax()
                                        for result in prediction][0]
                    Final_prediction = labels[Final_prediction]
                    a = 0
                    for i in prediction[0]:
                        print(labels[a])
                        print('Percent:{:.30%}'.format(i))
                        votes[a] = i
                        a = a+1

                    candidates = ['plane', 'mobile', 'bird', 'cat', 'deer',
                                  'dog', 'frog', 'horse', 'ship', 'truck']
                    x = np.arange(len(candidates))
                    plt.bar(x, votes, tick_label=candidates)
                    plt.ylim([0, 1])
                    plt.show()
                    return Final_prediction

            def main():
                Predict = MainPredictImg()
                res = Predict.pre()
                print('your picture is :-->', res)

            if __name__ == '__main__':
                main()

        self.pushButton51.clicked.connect(click51)
        self.pushButton52.clicked.connect(click52)
        self.pushButton53.clicked.connect(click53)
        self.pushButton54.clicked.connect(click54)
        self.pushButton55.clicked.connect(click55)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "5. Training Cifar10 Classifier Using VGG16 "))
        self.pushButton51.setText(_translate("MainWindow", "5.1 Show Training Images "))
        self.pushButton52.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.pushButton53.setText(_translate("MainWindow", "5.3 Show Model Structure"))
        self.pushButton54.setText(_translate("MainWindow", "5.4 Show Accuracy and Loss"))
        self.pushButton55.setText(_translate("MainWindow", "5.5 Test"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
