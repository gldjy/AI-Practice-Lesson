#!/usr/bin/env python3
# preprocess_image.py <video|picture> <device_or_image_path>

import sys

import cv2
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <video|picture> <source>")
        sys.exit(1)

    type = sys.argv[1]
    source = sys.argv[2]

    # 读取
    if type == "video":
        capture = cv2.VideoCapture(source)
        image = True
    elif type == "picture":
        frame = cv2.imread(source)
        image = True

    while image:
        if type == "video":
            ret, frame = capture.read()
            frame = cv2.flip(frame, 0)  # 手机画面水平翻转

            if not ret:
                break
        # 调整图像大小
        # 第一个参数为输入的图像；第二个参数为修改后的图片尺寸
        frame = cv2.resize(frame, (320, 320))

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 将灰度图转换为三通道图像
        # 利用opencv的merge函数将三张灰度图进行堆叠得到三通道的灰度图
        gray = cv2.merge([gray, gray, gray])

        # 使用高斯滤波处理图像
        # 第二个参数 (5, 5) 代表高斯核的大小，它们都必须为正奇数；第三个参数为标准差
        gaussianfilter = cv2.GaussianBlur(frame, (5, 5), 1)

        # 边缘检测（此步骤抠代码）
        # ##### 代码填空开始处 ##### #
        """
            代码填空提示：
            1. 通过 cv2 调用 Canny 函数；
            2. Canny 函数一共包括三个参数：
            第一个参数为高斯滤波后的图像，第二个参数为最小阈值；第三个参数为最大阈值
            （阈值的大小可自由设定）。
            3. 使用变量 edge 去承接边缘提取后的图片结果。
        """
        edge = cv2.Canny(gaussianfilter, 100, 200)  # 参考填空实现
        # ##### 代码填空结束处 ##### #
        edge = cv2.merge([edge, edge, edge])

        # 二值化图像
        # 第二个参数代表阈值；第三个代表最大值，超过阈值的部分都会变成最大值
        _, binaryzation = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 彩色图像均衡化
        b, g, r = cv2.split(frame)
        frameB = cv2.equalizeHist(b)
        frameG = cv2.equalizeHist(g)
        frameR = cv2.equalizeHist(r)

        # 合并每一个通道c
        equalization = cv2.merge((frameB, frameG, frameR))

        cv2.putText(
            frame,
            "frame",
            (110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            gray,
            "gray",
            (120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            gaussianfilter,
            "gaussianfilter",
            (60, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            edge,
            "edge",
            (120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            binaryzation,
            "binaryzation",
            (60, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            equalization,
            "equalization",
            (60, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 拼接图像
        # 拼接的时候必须确保所有图像的维度都一样；axis=1表示水平拼接,axis=0表示垂直拼接
        jointImage1 = np.concatenate([frame, gray, gaussianfilter], axis=1)
        jointImage2 = np.concatenate([edge, binaryzation, equalization], axis=1)
        jointImage = np.concatenate([jointImage1, jointImage2], axis=0)

        if cv2.waitKey(1) == 27:  # 按下ESC退出程序

            break
        # 显示图像

        cv2.imshow("cat", jointImage)
        if type == "picture":
            image = False

            while True:
                if cv2.waitKey(1) == 27:  # 按下ESC退出程序

                    cv2.destroyAllWindows()
                    break

    if type == "video":
        capture.release()
    # 释放窗口

    cv2.destroyAllWindows()
