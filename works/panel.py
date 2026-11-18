#!/usr/bin/env python

import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from uart import *

# Resolve resources against this file so the panel works from any cwd.
RES_DIR = Path(__file__).resolve().parent / "res"


class Panel:
    """
    面板数据显示类
    """

    def __init__(self) -> None:
        self.speedTrans = 0.0  # 初始化传送台速度
        self.fps = 0  # 初始化FPS
        self.gestur = None  # 初始化手势标志
        self.button = self.Button()  # 初始化按钮状态
        self.error = [0] * 12  # 初始化错误代码状态
        self.font = ImageFont.truetype(
            str(RES_DIR / "utils/QingNingYouYuan.ttf"), 30
        )  # 系统日志

    class Button:
        """
        按钮状态
        """

        def __init__(self):
            self.buttonA = False
            self.buttonB = False
            self.buttonC = False

    def showError(self, frame, img):
        """
        显示错误代码信息
        """
        if uart.sensors.errorCode != 0:
            errorLogo = cv2.imread(str(RES_DIR / "images/panel/errorcode.png"))
            errorLogo = cv2.resize(errorLogo, (158, 158))
            img[28 : 28 + 158, 229 : 229 + 158] = errorLogo  # 绘制错误代码标志

            errorCode = uart.sensors.errorCode  # 接收错误代码
            if errorCode > 0:
                for i in range(11, -1, -1):  # 遍历错误代码
                    if errorCode - 2**i >= 0:
                        errorCode = errorCode - 2**i
                        self.error[i] = 1
                    else:
                        self.error[i] = 0
                frame = Image.fromarray(frame)  # 将数组转换成图像
                draw = ImageDraw.Draw(frame)  # 创建绘图对象

                for i in range(12):
                    if self.error[i]:  # 打印光强传感器错误提示
                        if i == 0:
                            txt = "环境光强传感器故障"
                        elif i == 1:
                            txt = "激光测距传感器故障"
                        elif i == 2:
                            txt = "温湿度传感器故障"
                        elif i == 3:
                            txt = "心率血氧传感器故障"
                        elif i == 4:
                            txt = "空气质量传感器故障"
                        elif i == 5:
                            txt = "电磁感应传感器故障"
                        elif i == 6:
                            txt = "惯性检测传感器故障"
                        elif i == 7:
                            txt = "手势检测传感器故障"
                        elif i == 8:
                            txt = "毫米波雷达传感器故障"
                        elif i == 9:
                            txt = "无线通信传感器故障"
                        elif i == 10:
                            txt = "板间通信故障"
                        elif i == 11:
                            txt = "传送台模组故障"

                        draw.text(
                            (30, 30 * i + 20), txt, font=self.font, fill=(0, 0, 255)
                        )  # 显示文本

                frame = np.array(frame)  # 将图片转换为数组格式
        img[200 : 200 + 480, 597 : 597 + 640] = frame  # 将视频绘制到实验箱图片上

        return img

    def showData(self, img):
        """
        显示串口接收到的数据
        """
        # 检测串口数据是否正常发送
        if uart.connect:
            connect = cv2.imread(str(RES_DIR / "images/panel/data.png"))
            connect = cv2.resize(connect, (158, 158))
            img[22 : 22 + 158, 35 : 35 + 158] = connect

        # 激光测距
        cv2.putText(
            img,
            str(uart.sensors.rangeLaser),
            (230, 307),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (216, 187, 129),
            3,
        )

        # 红外传感器
        cv2.putText(
            img,
            str(uart.sensors.adcInfra),
            (200, 405),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (11, 65, 228),
            3,
        )

        # 毫米波
        cv2.putText(
            img,
            str(uart.sensors.rangeRadar),
            (245, 525),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (49, 125, 237),
            3,
        )

        # 光强
        cv2.putText(
            img,
            str(uart.sensors.illume),
            (260 if uart.sensors.illume - 1000 < 0 else 250, 647),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (107, 255, 255),
            3,
        )

        # 音量
        cv2.putText(
            img,
            str(uart.sensors.adcMic),
            (230, 760),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (219, 150, 18),
            3,
        )

        # co2
        cv2.putText(
            img,
            str(uart.sensors.co2),
            (210, 970),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 249, 205),
            3,
        )

        # tvoc2
        cv2.putText(
            img,
            str(uart.sensors.tvoc),
            (210, 1040),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 249, 205),
            3,
        )

        # 传送台速度
        cv2.putText(
            img,
            str(round(self.speedTrans, 2)),
            (1630, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (145, 203, 105),
            3,
        )

        # 湿度
        cv2.putText(
            img,
            str(round(uart.sensors.humi, 1)),
            (1540, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (233, 160, 0),
            3,
        )

        # 温度
        cv2.putText(
            img,
            str(round(uart.sensors.temp, 1)),
            (1540, 295),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (233, 160, 0),
            3,
        )

        # 心率
        cv2.putText(
            img,
            str(uart.sensors.heartRate),
            (1540 if uart.sensors.heartRate - 100 < 0 else 1535, 415),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (143, 193, 163),
            3,
        )

        # 血氧饱和度
        cv2.putText(
            img,
            str(uart.sensors.sop2),
            (1560 if uart.sensors.sop2 - 100 < 0 else 1535, 493),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (143, 193, 163),
            3,
        )

        # 判断手势序号
        if uart.sensors.ges.gesture == 1:
            self.gestur = "UP"
        elif uart.sensors.ges.gesture == 2:
            self.gestur = "DOWN"
        elif uart.sensors.ges.gesture == 3:
            self.gestur = "LEFT"
        elif uart.sensors.ges.gesture == 4:
            self.gestur = "RIGHT"
        elif uart.sensors.ges.gesture == -1:
            self.gestur = self.gestur
        else:
            self.gestur = "None"
        uart.sensors.ges.gesture = -1

        # 手势序号
        cv2.putText(
            img,
            self.gestur,
            (1761, 355),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 157, 222),
            3,
        )

        # 手势光强
        cv2.putText(
            img,
            str(uart.sensors.ges.gesBrightness),
            (1761, 418),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 157, 222),
            3,
        )

        # 手势探测面积
        cv2.putText(
            img,
            str(uart.sensors.ges.gesAreaDet),
            (1761, 485),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 157, 222),
            3,
        )

        # X轴加速度
        cv2.putText(
            img,
            str(uart.sensors.imu.accx),
            (1457, 570),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # X轴角速度
        cv2.putText(
            img,
            str(uart.sensors.imu.gyrox),
            (1617, 570),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # Y轴加速度
        cv2.putText(
            img,
            str(uart.sensors.imu.accy),
            (1457, 645),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # Y轴角速度
        cv2.putText(
            img,
            str(uart.sensors.imu.gyroy),
            (1617, 645),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # Z轴加速度
        cv2.putText(
            img,
            str(uart.sensors.imu.accz),
            (1457, 710),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # Z轴角速度
        cv2.putText(
            img,
            str(uart.sensors.imu.gyroz),
            (1617, 710),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (219, 150, 18),
            3,
        )

        # 俯仰角
        cv2.putText(
            img,
            str(round(uart.sensors.imu.pitch, 1)),
            (1460, 785),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (219, 150, 18),
            3,
        )

        # 横滚角
        cv2.putText(
            img,
            str(round(uart.sensors.imu.roll, 1)),
            (1620, 785),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (219, 150, 18),
            3,
        )

        # 偏航角
        cv2.putText(
            img,
            str(round(uart.sensors.imu.yaw, 1)),
            (1799, 785),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (219, 150, 18),
            3,
        )

        # 键盘值
        cv2.putText(
            img,
            str(uart.sensors.keyCount),
            (1595, 918),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (68, 75, 123),
            3,
        )

        # 如果旋钮被按下则清空数值
        if uart.sensors.knob.btnPress != 0:
            uart.sensors.knob.count = 0
            uart.sensors.knob.btnPress = 0

        # 旋钮值
        cv2.putText(
            img,
            str(uart.sensors.knob.count),
            (1620, 1018),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (68, 75, 123),
            3,
        )

        # 如果按键A被按下则Ui开关打开
        if self.button.buttonA == True:
            cv2.circle(img, (1800, 857), 20, (15, 37, 240), -1)
        elif self.button.buttonA == False:
            cv2.circle(img, (1873, 857), 20, (15, 37, 240), -1)

        # 如果按键B被按下则Ui开关二打开
        if self.button.buttonB == True:
            cv2.circle(img, (1800, 939), 20, (77, 181, 224), -1)
        elif self.button.buttonB == False:
            cv2.circle(img, (1873, 939), 20, (77, 181, 224), -1)

        # 如果按键C被按下则Ui开关三打开
        if self.button.buttonC == True:
            cv2.circle(img, (1800, 1022), 20, (157, 109, 83), -1)
        elif self.button.buttonC == False:
            cv2.circle(img, (1873, 1022), 20, (157, 109, 83), -1)

        # 控制继电器开关
        uart.setRelaySwitch(
            self.button.buttonA, self.button.buttonB, self.button.buttonC
        )

        # 按钮值
        if uart.sensors.btn.btnPressA != 0:
            self.button.buttonA = bool(1 - self.button.buttonA)
            uart.sensors.btn.btnPressA = 0
            uart.sensors.btn.btnPressB = 0
            uart.sensors.btn.btnPressC = 0

        elif uart.sensors.btn.btnPressB != 0:
            self.button.buttonB = bool(1 - self.button.buttonB)
            uart.sensors.btn.btnPressA = 0
            uart.sensors.btn.btnPressB = 0
            uart.sensors.btn.btnPressC = 0

        elif uart.sensors.btn.btnPressC != 0:
            self.button.buttonC = bool(1 - self.button.buttonC)
            uart.sensors.btn.btnPressA = 0
            uart.sensors.btn.btnPressB = 0
            uart.sensors.btn.btnPressC = 0

        if uart.sensors.keyCount == 0:
            uart.sensors.keyCount = -1
            if self.speedTrans == 0.0:
                self.speedTrans = -0.02
            elif self.speedTrans != 0.0:
                self.speedTrans = 0.0
            uart.setTransSpeed(self.speedTrans)  # 控制传送台速度
        return img


"""
UT: 下位机传感器数据测试
"""
if __name__ == "__main__":
    # 开启数据接收串口
    uart = Uart("/dev/ttyUSB0")
    uart.start()  # 开启串口通信接收子线程
    signal.signal(signal.SIGINT, uart.stop)  # 定义软件退出信号量

    # 开启摄像头
    capture = cv2.VideoCapture("/dev/deepCamera")
    capture.set(cv2.CAP_PROP_FPS, 50)  # 设置视频的读取速率
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建窗口
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    # 将窗口铺满屏幕
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 初始化面板数据显示类
    panel = Panel()

    uart.transmitSysHeart()
    time.sleep(1)  # 等待下位机初始化完成!
    uart.robotArmMoveit(uart.robotArm.Movement.photoTrans)  # 初始化机械手臂位置
    while uart.running:
        # 读取摄像头数据
        ret, frame = capture.read()
        uart.transmitSysHeart()
        # 检查是否成功读取视频帧
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 画面水平翻转

        uart.comStatusCheck()  # 通信连接状态检测
        img = cv2.imread(str(RES_DIR / "images/panel/AiBox.png"))  # 读取实验箱图片
        img = cv2.resize(img, (1920, 1080))  # 调整图片大小
        img = panel.showError(frame, img)  # 显示摄像头视频和错误代码提示
        img = panel.showData(img)  # 显示传感器数据
        cv2.imshow("Video", img)
        if cv2.waitKey(1) == 27:  # 按ESC退出程序
            break

    # 释放资源
    uart.stop(0, 0)  # 串口线程退出
    capture.release()
    cv2.destroyAllWindows()
