from PyQt5 import QtCore,QtGui,QtWidgets
import sys
import qtawesome
import cv2
import sys, cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets,QtCore
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import pygame
import time

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(960,700)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout) # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QWidget() # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout) # 设置右侧部件布局为网格
        self.left_close = QtWidgets.QPushButton("")  # 关闭按钮
        self.left_visit = QtWidgets.QPushButton("")  # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.right_label_1 = QtWidgets.QPushButton("学生信息")
        self.right_label_1.setObjectName('right_label')
        self.right_label_2 = QtWidgets.QPushButton("考试信息")
        self.right_label_2.setObjectName('right_label')
        self.right_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.right_label_3.setObjectName('right_label')

        self.left_label_1 = QtWidgets.QPushButton("学生信息")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("考试信息")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("联系与帮助")
        self.left_label_3.setObjectName('left_label')
        self.left_label_4 = QtWidgets.QPushButton("姓名")
        self.left_label_4.setObjectName('left_label')
        self.left_label_5 = QtWidgets.QPushButton("学号")
        self.left_label_5.setObjectName('left_label')
        self.left_label_6 = QtWidgets.QPushButton("班级")
        self.left_label_6.setObjectName('left_label')
        self.left_label_7 = QtWidgets.QPushButton("科目")
        self.left_label_7.setObjectName('left_label')
        self.left_label_8 = QtWidgets.QPushButton("时间")
        self.left_label_8.setObjectName('left_label')

        self.left_button_1 = QtWidgets.QPushButton(qtawesome.icon('fa.music', color='white'), "人脸认证")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(qtawesome.icon('fa.sellsy', color='white'), "头部姿态检测")
        self.left_button_2.setObjectName('left_button')
        self.left_button_3 = QtWidgets.QPushButton(qtawesome.icon('fa.sellsy', color='white'), "可疑行为检测")
        self.left_button_3.setObjectName('left_button')

        self.left_xxx = QtWidgets.QPushButton(" ")

        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_button_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_3, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_1, 4, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_2, 8, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_3, 11, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_4, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_5, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_6, 7, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_7, 9, 0, 1, 3)
        self.left_layout.addWidget(self.left_label_8, 10, 0, 1, 3)
        self.left_close.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15)  # 设置最小化按钮大小
        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        self.left_widget.setStyleSheet('''
            QPushButton{border:none;color:white;}
            QPushButton#left_label{
                border:none;
                border-bottom:1px solid white;
                font-size:18px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
            
            QWidget#left_widget{
                background: gray;
                border - top: 1px solid white;
                border - bottom: 1px solid white;
                border - left: 1px solid white;
                order - top - left - radius: 10px;
                border - bottom - left - radius: 10px;
            }
            QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        ''')
        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:16px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

        self.setWindowOpacity(0.9)  # 设置窗口透明度
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框

        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self.left_widget,0,0,12,2) # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget,0,2,12,10) # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget) # 设置窗口主部件

        self.btnOpen = QPushButton('启动', self)
        self.btnOpen.setToolTip("点击此按钮将启动应用程序")
        self.btnOpen.setStatusTip("点击此按钮将启动应用程序")
        self.btnOpen.clicked.connect(self.button_open_camera_clicked)
        self.btnOpen.resize(160, 70)
        self.btnOpen.move(1350, 600)

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                              buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.btnOpen.setText('关闭')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.vF.clear()  # 清空视频显示区域
            self.btnOpen.setText('启动')
def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()