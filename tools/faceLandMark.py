#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file          :faceLandmark.py
@Description   :人脸关键点检测模型 - 基于Paddle Inference
@Date          :2023/07/25 21:30:43
@Autor         :HC
@Version       :v1.1 - 适配Paddle Inference推理框架
"""
import os
import time
import yaml
import cv2
import numpy as np
from paddle.inference import Config, create_predictor


class FaceKeyPoint:
    """
    人脸关键点检测类
    读取配置文件并进行模型推理和绘制
    path1: 人脸检测模型所在的文件夹路径
    path2: 人脸关键点检测模型所在的文件夹路径
    """
    def __init__(self, path1, path2):
        # 人脸检测模型路径
        self.detect_model_dir = path1

        # 人脸关键点模型路径
        self.keypoint_model_dir = path2
        self.keypoint_model_file = os.path.join(path2, "model.pdmodel")
        self.keypoint_params_file = os.path.join(path2, "model.pdiparams")

        # 读取关键点模型配置
        with open(os.path.join(path2, "infer_cfg.yml"), "r") as f:
            self.keypoint_inferYml = yaml.safe_load(f)

        # 解析预处理参数
        self._parse_preprocess_config()

        # 加载人脸检测器 (使用已完成的Detection类)
        self.detector = Detector(path1)

        # 加载人脸关键点Paddle Inference模型
        self.keypoint_predictor = self._create_predictor()
        self.keypoint_input_names = self.keypoint_predictor.get_input_names()
        self.keypoint_output_names = self.keypoint_predictor.get_output_names()

    def _parse_preprocess_config(self):
        """解析人脸关键点预处理配置"""
        self.target_size = (96, 96)  # 默认值
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.is_scale = True

        # 从配置文件中读取参数
        for op in self.keypoint_inferYml.get("Preprocess", []):
            if op.get("type") == "Resize":
                target_size = op.get("target_size", [96, 96])
                self.target_size = tuple(target_size) if isinstance(target_size, list) else (target_size, target_size)
            elif op.get("type") == "NormalizeImage":
                self.mean = op.get("mean", self.mean)
                self.std = op.get("std", self.std)
                self.is_scale = op.get("is_scale", True)

        # 关键点数量
        self.keypoint_num = self.keypoint_inferYml.get("keypoint_num", 15)

    def _create_predictor(self):
        """创建Paddle Inference预测器"""
        # 创建Config
        config = Config(self.keypoint_model_file, self.keypoint_params_file)

        # 启用GPU推理 (Jetson)
        config.enable_use_gpu(1024, 0)

        # 启用内存优化
        config.enable_memory_optim()

        # 创建预测器
        predictor = create_predictor(config)
        return predictor

    def preprocess_keypoint(self, im):
        """
        人脸关键点预处理函数
        将图像处理成模型输入格式
        """
        # BGR转RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Resize到目标尺寸
        im = cv2.resize(im, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 归一化 [0,1]
        if self.is_scale:
            im = im.astype(np.float32) / 255.0

        # 标准化
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        im = (im - mean) / std

        # HWC -> CHW
        im = im.transpose((2, 0, 1))

        # 添加batch维度
        im = np.expand_dims(im, axis=0)

        return im

    def postprocess_keypoint(self, kpt, box):
        """
        人脸关键点后处理函数
        将归一化关键点转换回原图坐标

        Args:
            kpt: 归一化关键点坐标 [30] - 15个点,每个点x,y
            box: 人脸框坐标 [xmin, ymin, xmax, ymax]
        """
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin

        # 转换到原图坐标
        kpt_orig = kpt.copy()
        # 偶数索引是x坐标，奇数索引是y坐标
        kpt_orig[0::2] = kpt[0::2] * h + ymin  # x坐标 -> 对应高度方向
        kpt_orig[1::2] = kpt[1::2] * w + xmin  # y坐标 -> 对应宽度方向

        return kpt_orig

    def render(self, img, res, threshold):
        """
        模型预测函数
        对图像进行预测并输出预测结果

        Args:
            img: 输入图像
            res: 人脸检测结果
            threshold: 置信度阈值

        Returns:
            res: 包含boxes和kpts的字典
        """
        # 提取检测到的人脸
        boxes = []
        for box in res:
            if box.score < threshold:
                continue
            xmin, ymin, xmax, ymax = box.xMin, box.yMin, box.xMax, box.yMax
            boxes.append([xmin, ymin, xmax, ymax])

        # 对每个人脸框进行关键点检测
        kpts = []
        valid_boxes = []

        for box in boxes:
            xmin, ymin, xmax, ymax = box
            subImg = img[int(ymin):int(ymax), int(xmin):int(xmax)]

            if subImg.shape[0] > 0 and subImg.shape[1] > 0:
                # 预处理
                input_data = self.preprocess_keypoint(subImg)

                # 设置输入
                input_tensor = self.keypoint_predictor.get_input_handle(self.keypoint_input_names[0])
                input_tensor.copy_from_cpu(input_data)

                # 执行推理
                self.keypoint_predictor.run()

                # 获取输出
                output_tensor = self.keypoint_predictor.get_output_handle(self.keypoint_output_names[0])
                kpt_result = output_tensor.copy_to_cpu()

                # 解析关键点结果
                if kpt_result is not None and len(kpt_result) > 0:
                    kpt = kpt_result[0]  # 取第一个batch的结果

                    # 后处理：转换到原图坐标
                    kpt_processed = self.postprocess_keypoint(kpt, box)
                    kpts.append(kpt_processed)
                    valid_boxes.append(box)

        # 整理结果
        if len(kpts) > 0:
            res = {
                "kpts": np.array(kpts),
                "boxes": np.array(valid_boxes)
            }
        else:
            res = {"kpts": np.array([]), "boxes": np.array([])}

        return res

    def drawBox(self, img, res):
        """
        绘制识别的方框和关键点
        img: 图像
        res: 预测结果
        """
        kpts = res["kpts"]
        boxes = res["boxes"]

        for i, kpt in enumerate(kpts):
            # 绘制关键点
            for x, y in zip(kpt[0::2], kpt[1::2]):
                cv2.drawMarker(
                    img,
                    (int(x), int(y)),
                    (0, 255, 0),  # 绿色
                    markerType=cv2.MARKER_CROSS,
                    markerSize=6,
                    thickness=1,
                )

            # 绘制人脸框
            if i < len(boxes):
                xmin, ymin, xmax, ymax = boxes[i]
                cv2.rectangle(
                    img,
                    (int(xmin), int(ymin)),
                    (int(xmax), int(ymax)),
                    (0, 255, 255)  # 黄色
                )

        return img


class Detector:
    """
    人脸检测器 - 使用Paddle Inference
    复用 faceDetection.py 的 Detection 类
    """
    def __init__(self, path):
        # 复用Detection类的实现
        self.detection = Detection(path)

    def render(self, img):
        """
        人脸检测
        返回检测结果
        """
        return self.detection.render(img)


class Detection:
    """
    人脸检测类 - 复用 faceDetection.py
    """
    def __init__(self, path):
        # 模型路径
        self.model_dir = path
        self.model_file = os.path.join(path, "model.pdmodel")
        self.params_file = os.path.join(path, "model.pdiparams")

        # 读取yml文件配置模型参数
        with open(os.path.join(path, "infer_cfg.yml"), "r") as f:
            self.inferYml = yaml.safe_load(f)

        # 获取预处理参数
        self._parse_preprocess_config()

        # 加载Paddle Inference模型
        self.predictor = self._create_predictor()

        # 获取输入输出名称
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

    def _parse_preprocess_config(self):
        """解析预处理配置"""
        self.target_size = (192, 192)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.is_scale = True

        for op in self.inferYml.get("Preprocess", []):
            if op.get("type") == "Resize":
                target_size = op.get("target_size", [192, 192])
                self.target_size = tuple(target_size) if isinstance(target_size, list) else (target_size, target_size)
            elif op.get("type") == "NormalizeImage":
                self.mean = op.get("mean", self.mean)
                self.std = op.get("std", self.std)
                self.is_scale = op.get("is_scale", True)

    def _create_predictor(self):
        """创建Paddle Inference预测器"""
        config = Config(self.model_file, self.params_file)
        config.enable_use_gpu(1024, 0)
        config.enable_memory_optim()
        predictor = create_predictor(config)
        return predictor

    def preprocess(self, im):
        """图像预处理"""
        if isinstance(im, str):
            im = cv2.imread(im)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_shape = np.array(im.shape[:2], dtype=np.float32)

        im = cv2.resize(im, self.target_size, interpolation=cv2.INTER_LINEAR)

        if self.is_scale:
            im = im.astype(np.float32) / 255.0

        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        im = (im - mean) / std

        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        scale_factor = np.array([
            self.target_size[0] / im_shape[0],
            self.target_size[1] / im_shape[1]
        ], dtype=np.float32).reshape(1, 2)

        return {
            "image": im,
            "im_shape": im_shape.reshape(1, 2),
            "scale_factor": scale_factor
        }

    def postprocess(self, result, im_shape, scale_factor):
        """后处理"""
        label_list = self.inferYml.get("label_list", [])
        score_threshold = self.inferYml.get("draw_threshold", 0.5)

        if result is None:
            return []

        result = np.array(result)
        if result.size == 0:
            return []

        class Result:
            score = 0.0
            lable = ""
            xMin = 0
            yMin = 0
            xMax = 0
            yMax = 0

        results = []

        for det in result:
            det = np.array(det)
            if det.size < 6:
                continue

            cat_id = int(det[0])
            score = float(det[1])
            x1, y1, x2, y2 = det[2], det[3], det[4], det[5]

            if score < score_threshold:
                continue

            orig_h, orig_w = int(im_shape[0]), int(im_shape[1])

            x1_final = x1 * scale_factor[1]
            y1_final = y1 * scale_factor[0]
            x2_final = x2 * scale_factor[1]
            y2_final = y2 * scale_factor[0]

            x1_final = max(0, x1_final)
            y1_final = max(0, y1_final)
            x2_final = max(x1_final + 1, x2_final)
            y2_final = max(y1_final + 1, y2_final)

            res_obj = Result()
            res_obj.score = score
            res_obj.lable = label_list[cat_id] if cat_id < len(label_list) else "face"
            res_obj.xMin = max(0, min(int(x1_final), orig_w - 1))
            res_obj.yMin = max(0, min(int(y1_final), orig_h - 1))
            res_obj.xMax = max(0, min(int(x2_final), orig_w))
            res_obj.yMax = max(0, min(int(y2_final), orig_h))

            results.append(res_obj)

        return results

    def render(self, images):
        """推理"""
        preprocessed = self.preprocess(images)

        for input_name in self.input_names:
            input_tensor = self.predictor.get_input_handle(input_name)

            if input_name == "image" or input_name == "x":
                input_tensor.copy_from_cpu(preprocessed["image"])
            elif input_name == "scale_factor":
                input_tensor.copy_from_cpu(preprocessed["scale_factor"])
            elif input_name == "im_shape":
                input_tensor.copy_from_cpu(preprocessed["im_shape"])
            else:
                input_tensor.copy_from_cpu(preprocessed["image"])

        self.predictor.run()

        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        result = output_tensor.copy_to_cpu()

        im_shape = preprocessed["im_shape"][0]
        scale_factor = preprocessed["scale_factor"][0]

        return self.postprocess(result, im_shape, scale_factor)


if __name__ == "__main__":
    # 加载模型文件
    faceLandMark = FaceKeyPoint(
        "../../res/models/faceDetection",
        "../../res/models/faceLandmark"
    )

    # 深度相机设备路径
    DEEP_CAMERA_DEV = "/dev/deepCamera"
    RGB_CAMERA_DEV = "/dev/rgbCamera"

    # 打开摄像头
    capture = cv2.VideoCapture(DEEP_CAMERA_DEV)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建窗口
    cv2.namedWindow("Face Landmark - Real-time FPS", cv2.WINDOW_NORMAL)

    # FPS 计算变量
    frame_count = 0
    fps = 0.0
    start_time = time.time()
    display_fps = 0.0

    while True:
        ret, frame = capture.read()

        if not ret:
            break

        # #### 代码填空开始处 #### #           
        # 使用人脸检测模型检测人脸，结果存入 face_res
        face_res = faceLandMark.detector.render(frame)
        # 使用关键点检测模型检测关键点：图像、人脸检测结果、置信度阈值 0.5
        res = faceLandMark.render(frame, face_res, 0.5)
        # #### 代码填空结束处 #### #  

        # 绘制结果
        frame = faceLandMark.drawBox(frame, res)

        # 计算FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            display_fps = fps
            frame_count = 0
            start_time = current_time

        # 显示FPS
        fps_text = f"FPS: {display_fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Face Landmark - Real-time FPS", frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()