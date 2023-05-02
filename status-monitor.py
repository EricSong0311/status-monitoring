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
#import dlib
import threading
from mtcnn import MTCNN
from keras.models import load_model

# count用来计算闭眼次数
count = 0
# num用来计数嘴部有哈欠嫌疑数
num = 0
# 记录警报的次数
warning = 1
# 记录人脸认证是否成功
flag1 = 0
# 记录是否是活体检测
flag2 = 0
# 活体检测标志
ht = 0

file1 = '语音/警报.wav'
file2 = '语音/匹配中.wav'
file3 = '语音/匹配成功.wav'
file4 = '语音/活体检测.wav'
'''
model = load_model("model/fas.h5")
mtcnn = MTCNN("model/mtcnn.pb")
'''
# 眼镜阈值
eyerestrict = 0.26
#嘴巴阈值
mouthrestrict = 0.50
# 连续画面阈值
frame_check = 30
'''
#导入dlib库已有的人脸模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#确定人脸的左右眼以及嘴巴的起始和最终坐标点
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart,mEnd)= face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
'''
# 用来存放所有录入人脸特征的数组
features_known_arr = [[-0.047701251693069935, 0.053784324546103116, 0.025724446702817524, -0.01826150628686365, -0.08448294701951521, -0.023902302271583013, -0.06759535881725175, -0.15941417824338985, 0.14698936528077833, -0.10141650290676843, 0.26383963227272034, -0.014303302409610263, -0.2364443868950561, -0.12198539312790943, -0.07331302122178453, 0.187747382455402, -0.13278684089029275, -0.14141405677353894, -0.030786407749272056, 0.04368206053213389, 0.10993288002080387, -0.03658479615545797, 0.04572618588874185, 0.037888613839944206, -0.058860816268457315, -0.3428637722024211, -0.09971125499793776, -0.07014137430599443, 0.005828365235141029, -0.033878023768516466, -0.05061100150837943, 0.033629430772702175, -0.2036584825427444, -0.057980781593532475, -0.001390545077070042, 0.047709844040649915, -0.026634568378708704, -0.056873445019677824, 0.21504321997916254, -0.0058574623307558115, -0.2676929412064729, 0.02672571215899316, 0.0045265345144327035, 0.2014767908387714, 0.16344240707931695, 0.08472404332348595, 0.04190091088552166, -0.12496973836311588, 0.10068856131423402, -0.21215630019152606, 0.09134217448256632, 0.14203428849577904, 0.11695933410966837, 0.018066834132359537, -0.03924742174610772, -0.16215436615877682, 0.02290875021437252, 0.10525451372894976, -0.1804672744539049, 0.03736644502108296, 0.13021692619831474, -0.12029965980737298, -0.033353539589033636, -0.09023314238422447, 0.22435147591211174, 0.10286071675795096, -0.13121231231424546, -0.16773141341076958, 0.11764253193029653, -0.15726439158121744, -0.06081824280597545, 0.05834226586200573, -0.16670985423304416, -0.20177187936173546, -0.3658432607297544, 0.0353296329267323, 0.3833423505226771, 0.13208872535162502, -0.16895013219780392, 0.002120748903878309, -0.04685960320272931, 0.07035808949041422, 0.11051634105819244, 0.12892282726588072, 0.003908988536783942, -0.0023770814209624572, -0.13315269419992412, -0.04349418025877741, 0.1844405181430004, -0.06969924422877806, -0.06387133854958746, 0.24158916456831825, -0.02864446292665821, 0.09051005680252004, -0.018494016180435814, 0.07104813390307957, -0.03830040986132291, 0.03618223534000141, -0.09306324445814998, 0.04204894932573317, 0.015961399526093847, 0.007485033211263794, 0.029490304334710043, 0.10721914070071996, -0.10675203248306556, 0.090096370489509, -0.074686743353528, 0.06467700556472496, 0.04012659407669195, -0.025679631406631046, -0.12215991877019405, -0.10350577064134456, 0.10094214824062807, -0.20307157768143547, 0.21270872017851586, 0.16360780707112066, 0.08840954151970369, 0.10159852266035696, 0.1387663802338971, 0.11605266481637955, -0.02684397205572437, -0.042275658774155164, -0.2628658126901697, -0.032586667630649, 0.1038549814235281, -0.03260503650677425, 0.11003170231426204, 0.02682656833591561]]
features_cap_arr = []
pos_namelist = []
name_namelist = []
e_distance_list = []

# 报警函数
def sound_alarm(file,time1):
    # play an alarm sound
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play()
    time.sleep(time1)
    pygame.mixer.music.stop()

# 计算眼睛的纵横比
'''
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizon
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
'''
#检测嘴巴张开程度
'''
def mouth_open(mouth):
    A = dist.euclidean(mouth[2],mouth[10])
    B = dist.euclidean(mouth[3],mouth[9])
    C = dist.euclidean(mouth[4],mouth[8])
    D = dist.euclidean(mouth[0],mouth[6])
    L = ( A + B + C ) / 3
    mar = L / D
    return mar
'''
# 计算两个128D向量间的欧式距离
'''
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist
'''
'''
def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X, (224, 224)) - 127.5) / 127.5  # 归一化操作，加快收敛速度
    t = model.predict(np.array([X]))[0]
    time_end = time.time()
    return t
'''
class CMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800,400)
        self.setObjectName("mainWindow")

        # ToolTip设置
        QToolTip.setFont(QFont('华文楷体', 10))
        self.setWindowOpacity(0.98)  # 设置透明度
        #self.setStyleSheet('#Form{border-image:url(e:/python/lunwen/1.jpg);}')  # 设置背景图

        # statusBar设置
        self.statusBar().showMessage('准备就绪')

        # 菜单栏设置
        # 退出Action设置
        exitAction = QAction(QIcon('图片/退出.jpg'), '&Exit', self)
        exitAction.setShortcut('ctrl+Q')
        exitAction.setStatusTip('点击此按钮将退出应用程序！')
        exitAction.triggered.connect(self.close)
        # 举手Action设置
        handAction = QAction(QIcon('图片/举手.jpg'), '&hand', self)
        handAction.setShortcut('ctrl+w')
        handAction.setStatusTip('点击此按钮将举手')
        handAction.triggered.connect(self.message1)
        # 疑问Action设置
        queAction = QAction(QIcon('图片/疑问.jpg'), '&question', self)
        queAction.setShortcut('ctrl+e')
        queAction.setStatusTip('对本app的介绍')
        queAction.triggered.connect(self.message2)

        # menuBar设置
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&文件')
        fileMenu.addAction(exitAction)
        fileMenu.addAction(handAction)
        fileMenu.addAction(queAction)
        # toolBar设置  工具栏设置
        self.toolbar = self.addToolBar('退出')
        self.toolbar.addAction(exitAction)

        self.toolbar = self.addToolBar('举手')
        self.toolbar.addAction(handAction)

        self.toolbar = self.addToolBar('疑问')
        self.toolbar.addAction(queAction)

        self.center()#主窗口居中显示
        self.setFont(QFont('华文楷体', 18))
        # 设置主窗体内控件字体
        self.setWindowTitle('好学不倦')
        self.setWindowIcon(QIcon('图片/app.jpg'))

        # 初始化传入的摄像头句柄为实例变量,并得到摄像头宽度和高度
        self.cap = cv2.VideoCapture(0)
        self.CAM_NUM = 0
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 设置GUI窗口的位置和尺寸
        self.setGeometry(300, 200, self.w + 200, self.h + 200)
        self.vF = QLabel()
        self.setCentralWidget(self.vF)
        self.vF.setFixedSize(1000, 700)  # 设置背景大小

        lb = QLabel("用户：", self)
        lb.setGeometry(QtCore.QRect(30, 100, 100, 50))

        self.lb1 = QLabel('未认证', self)
        self.lb1.setGeometry(QtCore.QRect(100, 100, 170, 50))


        lb = QLabel("考试类型：", self)
        lb.setGeometry(QtCore.QRect(30, 140, 100, 50))

        self.lb2 = QLabel('线上考试、纸质考试', self)
        self.lb2.setGeometry(QtCore.QRect(115, 140, 170, 50))

        ft = QFont()  # 实例一个QFont对象,设置字体
        ft.setPointSize(18)  # 设置字体大小
        ft.setFamily('楷体')
        self.lb1.setFont(ft)
        self.lb2.setFont(ft)

        lb = QLabel("学生状态：", self)
        lb.setGeometry(QtCore.QRect(30, 180, 150, 50))

        self.lb2 = QLabel('学生状态',self)
        self.lb2.setGeometry(QtCore.QRect(30, 350, 200, 70))
        self.lb2.setStyleSheet('background-color:grey')
        self.lb2.setIndent(10)  # 设置缩进
        ft = QFont()    # 实例一个QFont对象,设置字体
        ft.setPointSize(14) # 设置字体大小
        ft.setFamily('楷体')
        self.lb2.setFont(ft)
        qp = QPalette() # 实例调色板对象，设置字体颜色
        qp.setColor(QPalette.Foreground,QColor(255,255,255))
        self.lb2.setPalette(qp)

        lb = QLabel("提醒次数：", self)
        lb.setGeometry(QtCore.QRect(1300, 320, 150, 50))


        lb = QLabel("当前人脸数：", self)
        lb.setGeometry(QtCore.QRect(1300, 390, 180, 50))


        lb = QLabel("是否是真人:", self)
        lb.setGeometry(QtCore.QRect(1300, 470, 180, 50))

        self.lb3 = QLabel('否', self)
        self.lb3.setGeometry(QtCore.QRect(1480, 470, 45, 45))
        self.lb3.setStyleSheet('background-color:red')
        self.lb3.setIndent(5)  # 设置缩进
        ft = QFont()  # 实例一个QFont对象,设置字体
        ft.setPointSize(12)  # 设置字体大小
        ft.setFamily('楷体')
        self.lb3.setFont(ft)
        qp = QPalette()  # 实例调色板对象，设置字体颜色
        qp.setColor(QPalette.Foreground, QColor(255, 255, 255))
        self.lb3.setPalette(qp)

        lb = QLabel(self)
        lb.setGeometry(1375, 900, 200, 200)
        pic = QPixmap('图片/app.jpg').scaled(lb.width(), lb.height())
        lb.setPixmap(pic)

        # 按钮设置
        # PushButton设置 -- 退出
        btnQuit = QPushButton('退出', self)
        btnQuit.setToolTip("点击此按钮将退出应用程序！")
        btnQuit.setStatusTip("点击此按钮将退出应用程序！")
        btnQuit.clicked.connect(self.close)
        btnQuit.resize(160, 70)
        btnQuit.move(1050, 700)


        self.btnhuoti = QPushButton('身份认证（活体检测）', self)
        self.btnhuoti.setToolTip("点击此按钮将进行身份认证")
        self.btnhuoti.setStatusTip("点击此按钮将进行身份认证")
        self.btnhuoti.clicked.connect(self.button_start_recognition)
        self.btnhuoti.resize(100, 40)
        self.btnhuoti.move(30, 50)
        self.btnhuoti.setStyleSheet("background-color:#ff0000;")

        self.btnstexam = QPushButton('开始考试', self)
        self.btnstexam.setToolTip("点击此按钮将开始考试")
        self.btnstexam.setStatusTip("点击此按钮将开始考试")
        self.btnstexam.clicked.connect(self.button_start_exam)
        self.btnstexam.resize(100, 40)
        self.btnstexam.move(30,300 )
        '''
        # PushButton设置 -- 启动
        self.btnOpen = QPushButton('启动', self)
        self.btnOpen.setToolTip("点击此按钮将启动应用程序")
        self.btnOpen.setStatusTip("点击此按钮将启动应用程序")
        self.btnOpen.clicked.connect(self.button_open_camera_clicked)
        self.btnOpen.resize(160, 70)
        self.btnOpen.move(1350, 600)

        # PushButton设置 -- 活体检测
        self.btnhuoti = QPushButton('活体检测', self)
        self.btnhuoti.setToolTip("点击此按钮将设置活体检测")
        self.btnhuoti.setStatusTip("点击此按钮将设置活体检测")
        self.btnhuoti.clicked.connect(self.button_start_huoti)
        self.btnhuoti.resize(160, 70)
        self.btnhuoti.move(1350, 800)
        
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(1281, 961)  # 给显示视频的Label设置大小为641x481
        # 设置定时器 每25毫秒执行实例的show_camera函数以刷新图像
        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.show_camera)
        '''
        self.show()



        #开始函数
    '''
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
    '''
    # 活体检测函数
    '''
    def button_start_huoti(self):
        global flag2
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            QtWidgets.QMessageBox.warning(self, 'warning', "未打开摄像头",
                                                buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.lb2.setStyleSheet('background-color:grey')
            self.lb2.setText('活体检测')
            if flag2 == 0:
                self.btnhuoti.setText('返回')
                flag2 = 1
            else:
                flag2 = 0
                self.btnhuoti.setText('活体检测')
    '''

    def button_start_recognition(self):
        global flag2
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            QtWidgets.QMessageBox.warning(self, 'warning', "未打开摄像头",
                                          buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.lb2.setStyleSheet('background-color:grey')
            self.lb2.setText('活体检测')
            if flag2 == 0:
                self.btnhuoti.setText('返回')
                flag2 = 1
            else:
                flag2 = 0
                self.btnhuoti.setText('活体检测')

    def button_start_exam(self):
        global flag3
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            QtWidgets.QMessageBox.warning(self, 'warning', "未打开摄像头",
                                          buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.lb2.setStyleSheet('background-color:grey')
            self.lb2.setText('活体检测')
            if flag3 == 0:
                self.btnhuoti.setText('返回')
                flag3 = 1
            else:
                flag3 = 0
                self.btnhuoti.setText('活体检测')


    def center(self):
        # 得到主窗体的框架信息
        qr = self.frameGeometry()
        # 得到桌面的中心
        cp = QDesktopWidget().availableGeometry().center()
        # 框架的中心与桌面中心对齐
        qr.moveCenter(cp)
        # 自身窗体的左上角与框架的左上角对齐
        self.move(qr.topLeft())

    # 提示框
    def message(self,msg):
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "提示",
                                        msg,
                                        QMessageBox.Yes)

    def message1(self):
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "提示",
                                        '已成功举手',
                                        QMessageBox.Yes)
    def message2(self):
        reply = QMessageBox.information(self,  # 使用infomation信息框
                                        "本App说明",
                                        '本App是基于网课环境下的学生精神状态检测系统，'
                                         '旨在通过实时检测学生精神状态，监督并提高学生听课效率\n如有疑问欢迎联系qq:XXXXXXXX',
                                        QMessageBox.Yes)

    def closeEvent(self, QCloseEvent):
        reply = QMessageBox.question(self,
                                     '好学不倦',
                                     "是否要退出应用程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()
'''
    def show_camera(self):
        # 声明全局变量
        global count, num, warning,ht
        global flag1,flag2
        global features_known_arr,features_cap_arr
        global pos_namelist,name_namelist,e_distance_list

        _, frame = self.cap.read()
        # 横向翻转
        frame = cv2.flip(frame, 1)
        # 返回脸的个数
        dets = detector(frame, 0)
        pos_namelist = []
        name_namelist = []

        self.lcd2.display(len(dets))
        if flag1 <= 15:
            # 处于人脸识别阶段

            if flag1 == 1:
                threading.Thread(target=sound_alarm(file2,3)).start()
                self.message('开始进行身份认证')
                self.lb2.setText('人脸识别中')

            if flag1 == 15:
                threading.Thread(target=sound_alarm(file3,3)).start()
                self.message('身份认证成功，开始听课')
                self.lb1.setText('苏华昇')
            features_cap_arr = []
            for i in range(len(dets)):
                shape = predictor(frame, dets[i])
                features_cap_arr.append(facerec.compute_face_descriptor(frame, shape))

            # 5. 遍历捕获到的图像中所有的人脸
            for k in range(len(dets)):
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")

                # # 每个捕获人脸的名字坐标 the positions of faces captured
                pos_namelist.append(
                    tuple([dets[k].left(), int(dets[k].bottom() + (dets[k].bottom() - dets[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                e_distance_list = []
                for i in range(len(features_known_arr)):
                    # 如果 person_X 数据不为空
                    e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                    e_distance_list.append(e_distance_tmp)

                if min(e_distance_list) < 0.4:
                    name_namelist[k] = "Suhuasheng"
                    flag1 += 1

                # 矩形框
                for kk, d in enumerate(dets):
                    # 绘制矩形框
                    cv2.rectangle(frame, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255),
                                  2)

            # 6. 在人脸框下面写人脸名字
            for i in range(len(dets)):
                cv2.putText(frame, name_namelist[i], pos_namelist[i], cv2.FONT_ITALIC, 0.8, (0, 255, 255), 1,
                            cv2.LINE_AA)



        else:
            if flag2 == 0:
                for i in range(len(dets)):
                    shape = predictor(frame, dets[i])
                    shape = face_utils.shape_to_np(shape)
                    # 转换为numpy格式
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    mouth = shape[mStart:mEnd]

                    # 计算眼睛和嘴巴横纵比
                    ear = (leftEAR + rightEAR) / 2.0
                    mar = mouth_open(mouth)

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    mouthHull = cv2.convexHull(mouth)

                    # 画出我们的眼睛和嘴轮廓
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                    # check to see if the eye aspect ratio is below the blink
                    # threshold, and if so, increment the blink frame counter
                    if ear < eyerestrict or mar > mouthrestrict:
                        if ear < eyerestrict:
                            count += 1
                            print("eye close times", count)
                        if mar > mouthrestrict:
                            num += 1
                            print("you may yawn for", num, "times")
                        if count >= frame_check or num >= frame_check:
                            # draw an alarm on the frame
                            cv2.putText(frame, "************!!!!Attention Please!!!**********", (80, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 97, 255), 2)
                            Text = ("*************!Fatigue Alert %d time!**********" % warning)
                            cv2.putText(frame, Text, (80, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 97, 255), 2)
                            # 文本函数，cv2.putText(src, text, place, Font, Font_Size, Font_Color, Font_Overstriking)
                            # 调用警报函数
                            # sound_alarm(file1, 0.2)
                        if warning >= 3 and (count >= frame_check or num >= frame_check):
                            threading.Thread(target=sound_alarm(file1,1)).start()

                    else:
                        # 如触发警报，次数加一
                        if (count >= frame_check or num >= frame_check):
                            self.lcd1.display(warning/3)
                            warning += 1

                        count = 0
                        num = 0

                    # 当警告超过次数，标定精神状态低下
                    if warning >= 4:
                        self.lb2.setStyleSheet('background-color:red')
                        self.lb2.setText('状态低下')
                        with open('精神状态报告.txt', 'w') as f:
                            f.write("date:" + time.strftime("%Y-%m-%d", time.localtime(time.time()))+'\n')
                            f.write("time::" + time.strftime("%H:%M:%S", time.localtime(time.time()))+'\n')
                            f.write("******当前第{}次警告，已反馈至教师端******\n".format(warning-1))
                            f.write("*****{}处于疲劳状态，听课效率低下*****\n".format('苏华昇'))
                            f.close()
                    elif 2 <= warning:
                        self.lb2.setStyleSheet('background-color:orange')
                        self.lb2.setText('状态一般')
                    else:
                        self.lb2.setStyleSheet('background-color:green')
                        self.lb2.setText('状态良好')

                    # 输出横纵比
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "MAR: {:.2f}".format(mar), (250, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:


                img_size = np.asarray(frame.shape)[0:2]

                bounding_boxes, scores, landmarks = mtcnn.detect(frame)

                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    # 并行遍历
                    for det, pts in zip(bounding_boxes, landmarks):

                        det = det.astype('int32')
                        # print("face confidence: %2.3f" % confidence)
                        det = np.squeeze(det)  # 得到矩形框两点坐标
                        y1 = int(np.maximum(det[0], 0))
                        x1 = int(np.maximum(det[1], 0))
                        y2 = int(np.minimum(det[2], img_size[1] - 1))
                        x2 = int(np.minimum(det[3], img_size[0] - 1))

                        w = x2 - x1
                        h = y2 - y1
                        _r = int(max(w, h) * 0.6)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 整除

                        x1 = cx - _r
                        y1 = cy - _r

                        x1 = int(max(x1, 0))
                        y1 = int(max(y1, 0))

                        x2 = cx + _r
                        y2 = cy + _r

                        h, w, c = frame.shape
                        x2 = int(min(x2, w - 2))
                        y2 = int(min(y2, h - 2))

                        _frame = frame[y1:y2, x1:x2]
                        score = test_one(_frame)

                        # 输出分数
                        cv2.putText(frame, "SCORE: {:.2f}".format(score[0]), (250, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        if score > 0.7:  # 设定阈值为0.7，大于阈值为真人


                            self.lb3.setText('是')
                            self.lb3.setStyleSheet('background-color:green')
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, "TRUE", (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                            ht += 1
                            if ht == 20:
                                ht = 0
                                threading.Thread(target=sound_alarm(file4, 3)).start()
                                msg = QtWidgets.QMessageBox.warning(self, '提示', "活体检测是真人",
                                                                    buttons=QtWidgets.QMessageBox.Ok)
                                self.lb2.setText('检测通过')
                                self.lb2.setStyleSheet('background-color:green')

                        else:
                            self.lb2.setStyleSheet('background-color:grey')
                            self.lb2.setText('活体检测')
                            self.lb3.setText('否')
                            self.lb3.setStyleSheet('background-color:red')
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, "FALSE", (x1, y1), cv2.FONT_ITALIC, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                            ht = 0



        show = cv2.resize(frame, (1280, 960))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.vF.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
'''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = CMainWindow()
    sys.exit(app.exec_())
