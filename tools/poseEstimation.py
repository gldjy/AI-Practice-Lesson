#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@file          :poseEstimation.py
@Description   :人体姿态检测模型 - 基于Paddle Inference
@Date          :2023/07/26 17:30:22
@Autor         :HC
@Version       :v2.0 - 迁移至Paddle Inference (从ppnc替换)
"""
import os
import time
import yaml
import cv2
import numpy as np
import math
from paddle.inference import Config, create_predictor

class Tinypose:
    """
    姿态检测类
    读取配置文件并进行模型推理和绘制
    path:
        第一个是人体检测模型文件夹路径，第二个是关键点模型文件夹路径.
    """
    def __init__(self, path1, path2):
        # 人体检测模型路径 (person detection)
        self.detector_model_dir = path1
        self.detector_model_file = os.path.join(path1, "model.pdmodel")
        self.detector_params_file = os.path.join(path1, "model.pdiparams")

        # 关键点检测模型路径 (keypoint detection)
        self.keypoint_model_dir = path2
        self.keypoint_model_file = os.path.join(path2, "model.pdmodel")
        self.keypoint_params_file = os.path.join(path2, "model.pdiparams")

        # 读取人体检测模型配置
        with open(os.path.join(path1, "infer_cfg.yml"), "r") as f:
            self.detector_inferYml = yaml.safe_load(f)

        # 读取关键点检测模型配置
        with open(os.path.join(path2, "infer_cfg.yml"), "r") as f:
            self.keypoint_inferYml = yaml.safe_load(f)

        # 解析关键点模型的预处理配置
        self.preprocessOps = []
        for opInfo in self.keypoint_inferYml.get("Preprocess", []):
            newOpInfo = opInfo.copy()
            opType = newOpInfo.pop("type")
            self.preprocessOps.append(eval(opType)(**newOpInfo))

        # 加载人体检测模型 (使用Paddle Inference)
        self.detector = Detector(self.detector_model_dir)

        # 加载关键点检测模型 (使用Paddle Inference)
        self.predictor = self._create_predictor()
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

        # 获取关键点模型配置参数
        self.keypoint_threshold = self.keypoint_inferYml.get("draw_threshold", 0.5)

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
        
    class Result:
        """
        AI检测结果
        """
        score = 0.0  # 置信度: 0.0~1.0
        lable = ""  # 标签
        xMin = 0  # 目标检测坐标: 左上x
        yMin = 0  # 目标检测坐标: 左上y
        xMax = 0  # 目标检测坐标: 右下x
        yMax = 0  # 目标检测坐标: 右下y
           
    def preprocess(self, image, results):
        """
        图像预处理
        根据人体检测模型得到的人体信息来进行预处理     
        """
        
        # 从图像中裁剪人的结果
        validRects = results["boxes"]
        rectImages = []
        newRects = []
        orgRects = []
        for rect in validRects:
            imgh, imgw, c = image.shape
            label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
            if label != 0:
                return None, None, None
            orgRect = [xmin, ymin, xmax, ymax]
            hHalf = (ymax - ymin) * (1 + 0.3) / 2.
            wHalf = (xmax - xmin) * (1 + 0.3) / 2.
            if hHalf > wHalf * 4 / 3:
                wHalf = hHalf * 0.75
            center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
            ymin = max(0, int(center[0] - hHalf))
            ymax = min(imgh - 1, int(center[0] + hHalf))
            xmin = max(0, int(center[1] - wHalf))
            xmax = min(imgw - 1, int(center[1] + wHalf))
            rectImage, newRect = image[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax]
        
            if rectImage is None or rectImage.size == 0:
                continue
            rectImages.append(rectImage)
            newRects.append(newRect)
            orgRects.append(orgRect)
        return rectImages, newRects, orgRects
    
    def postprocess(self, inputs, result):
        """
        图像后处理
        """
        npHeatmap = result["heatmap"]
        npMasks = result["masks"]
        results = {}
        imshape = inputs["im_shape"][:, ::-1]
        center = np.round(imshape / 2.0)
        scale = imshape / 200.0


        heatmaps = npHeatmap
        """
        从分数地图中获得预测结果
        """
        assert isinstance(
            heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
        assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batchSize = heatmaps.shape[0]
        numJoints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmapsReshaped = heatmaps.reshape((batchSize, numJoints, -1))
        idx = np.argmax(heatmapsReshaped, 2)
        maxvals = np.amax(heatmapsReshaped, 2)

        maxvals = maxvals.reshape((batchSize, numJoints, 1))
        idx = idx.reshape((batchSize, numJoints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        predMask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        predMask = predMask.astype(np.float32)

        preds *= predMask

        coords = preds
    
        heatmapHeight = npHeatmap.shape[2]
        heatmapWidth = npHeatmap.shape[3]

        """
        refer to https://github.com/ilovepose/DarkPose/lib/core/inference.py
        高斯滤波
        """
        heatmap = npHeatmap
        kernel = 3  # 高斯核的大小: 3*3
        border = (kernel - 1) // 2
        batchSize = heatmap.shape[0]
        numJoints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batchSize):
            for j in range(numJoints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        npHeatmap = heatmap
    
        npHeatmap = np.maximum(npHeatmap, 1e-10)
        npHeatmap = np.log(npHeatmap)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = npHeatmap[n][p]
                coord = coords[n][p]
                heatmapHeight = hm.shape[0]
                heatmapWidth = hm.shape[1]
                px = int(coord[0])
                py = int(coord[1])
                if 1 < px < heatmapWidth - 2 and 1 < py < heatmapHeight - 2:
                    dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
                    dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
                    dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
                    dxy = 0.25 * (
                        hm[py + 1][px + 1]
                        - hm[py - 1][px + 1]
                        - hm[py + 1][px - 1]
                        + hm[py - 1][px - 1]
                    )
                    dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py]
                                [px] + hm[py - 2 * 1][px])
                    derivative = np.matrix([[dx], [dy]])
                    hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
                    if dxx * dyy - dxy**2 != 0:
                        hessianinv = hessian.I
                        offset = -hessianinv * derivative
                        offset = np.squeeze(np.array(offset.T), axis=0)
                        coord += offset
                coords[n, p] = coord
                
        preds = coords.copy()

        for i in range(coords.shape[0]):
            inputSize = scale[i] * 200
            shift = (0.0, 0.0)
            targetCoords = np.zeros(coords[i].shape)
            assert len(center[i]) == 2
            assert len([heatmapWidth, heatmapHeight]) == 2
            assert len(shift) == 2
            if not isinstance(inputSize, (np.ndarray, list)):
                inputSize = np.array([inputSize, inputSize], dtype=np.float32)
            scaleTmp = inputSize

            shift = np.array(shift)
            srcW = scaleTmp[0]
            dstW = [heatmapWidth, heatmapHeight][0]
            dstH = [heatmapWidth, heatmapHeight][1]

            rotRad = np.pi * 0 / 180
            assert len([0.0, srcW * -0.5]) == 2
            sn, cs = np.sin(rotRad), np.cos(rotRad)
            newX = [0.0, srcW * -0.5][0] * cs - [0.0, srcW * -0.5][1] * sn
            newY = [0.0, srcW * -0.5][0] * sn + [0.0, srcW * -0.5][1] * cs
            srcDir = [newX, newY]
            dstDir = np.array([0.0, dstW * -0.5])

            src = np.zeros((3, 2), dtype=np.float32)
            src[0, :] = center[i] + scaleTmp * shift
            src[1, :] = center[i] + srcDir + scaleTmp * shift

            # 获得第三个点用于计算仿射矩阵
            assert len(src[0, :]) == 2
            assert len(src[1, :]) == 2
            direction = src[0, :] - src[1, :]
            src[2, :] = src[1, :] + np.array([-direction[1], direction[0]], dtype=np.float32)

            dst = np.zeros((3, 2), dtype=np.float32)
            dst[0, :] = [dstW * 0.5, dstH * 0.5]
            dst[1, :] = np.array([dstW * 0.5, dstH * 0.5]) + dstDir

            # 获得第三个点用于计算仿射矩阵
            assert len(dst[0, :]) == 2
            assert len(dst[1, :]) == 2
            direction = dst[0, :] - dst[1, :]
            dst[2, :] = dst[1, :] + np.array([-direction[1], direction[0]], dtype=np.float32)

  
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

            t = trans
            for p in range(coords[i].shape[0]):
                newPt = np.array([coords[i][p, 0:2][0], coords[i][p, 0:2][1], 1.0]).T
                newPt = np.dot(t, newPt)
                targetCoords[p, 0:2] = newPt[:2]
            preds[i] = targetCoords


        kpts, scores =  np.concatenate((preds, maxvals), axis=-1), np.mean(maxvals, axis=1)
    
        results["keypoint"] = kpts
        results["score"] = scores
        return results
    
    def render(self, image, detRes):
        """
        AI预测结果
        """
        if detRes["boxesNum"] <= 0:
            detRes["keypoint"] = [[], []]
            return detRes

        keypointRes = {}
        recImages, records, detRects = self.preprocess(image, detRes)


        if len(detRects) == 0:
            detRects["keypoint"] = [[], []]
            return detRects

        keypointVector = []
        scoreVector = []

        rectVector = detRects

        if not isinstance(recImages, list):
            recImages = [recImages]
        res = {"heatmap": [], "masks": None}
        inputs = {"im_shape": [], "scale_factor": [], "image": []}
        for i in recImages:
            inp = self.detector.preprocess(i, self.preprocessOps)
            for k in inputs.keys():
                inputs[k].append(inp[k])

            # 使用Paddle Inference进行关键点检测推理
            input_tensor = self.predictor.get_input_handle(self.input_names[0])
            input_tensor.copy_from_cpu(inp["image"])

            if "scale_factor" in inp and len(self.input_names) > 1:
                scale_tensor = self.predictor.get_input_handle(self.input_names[1])
                scale_tensor.copy_from_cpu(inp["scale_factor"])

            if "im_shape" in inp and len(self.input_names) > 2:
                im_shape_tensor = self.predictor.get_input_handle(self.input_names[2])
                im_shape_tensor.copy_from_cpu(inp["im_shape"])

            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(self.output_names[0])
            resS = output_tensor.copy_to_cpu()
            output = {"heatmap": resS}

            res["heatmap"].append(output["heatmap"])
  

        res["heatmap"] = np.concatenate(res["heatmap"])
        for k in inputs.keys():
            inputs[k] = np.concatenate(inputs[k])

        keypointResults = self.postprocess(inputs, res)
    
        keypointVector = keypointResults["keypoint"]
        scoreVector = keypointResults["score"]
        keypointVector[..., 0] += np.array(records)[:, 0:1]
        keypointVector[..., 1] += np.array(records)[:, 1:2]

        keypointRes["keypoint"] = (
            [keypointVector.tolist(), scoreVector.tolist()]
            if len(keypointVector) > 0
            else [[], []]
        )
        keypointRes["bbox"] = rectVector
        return keypointRes

    def drawBox(self, imgfile, results, visualThresh=0.4, ids=None):
        """
        绘制预测结果
        绘制出人体关键点和人体骨骼
        imgfile: 输入的图像
        results: 预测结果
        visualThresh 置信度
        """

        # 获取人体关键点
        skeletons, scores = results["keypoint"]
        # 将关键点的信息转换成np的形式
        skeletons = np.array(skeletons)
        kptNums = 17
        if len(skeletons) > 0:              # 如果获取骨骼关键点则将关键点的点数保存下来，否则默认为17个关键点
            kptNums = skeletons.shape[1]
        if kptNums == 17:
            EDGES = [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 4),
                (3, 5),
                (4, 6),
                (5, 7),
                (6, 8),
                (7, 9),
                (8, 10),
                (5, 11),
                (6, 12),
                (11, 13),
                (12, 14),
                (13, 15),
                (14, 16),
                (11, 12),
            ]
        else:
            EDGES = [
                (0, 1),
                (1, 2),
                (3, 4),
                (4, 5),
                (2, 6),
                (3, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (10, 11),
                (11, 12),
                (13, 14),
                (14, 15),
                (8, 12),
                (8, 13),
            ]
        NUM_EDGES = len(EDGES)

        colors = [                  # 设置关键点的颜色
            [255, 0, 0],
            [255, 85, 0],
            [255, 170, 0],
            [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [255, 0, 170],
            [255, 0, 85],
        ]

        # 用opencv读取传进函数内部的图片
        img = cv2.imread(imgfile) if type(imgfile) == str else imgfile

        # 获取模型预测结果中关键点的颜色
        colorSet = results["colors"] if "colors" in results else None

        # 如果预测结果识别到人则用opencv进行方框绘制
        if "bbox" in results and ids is None:
            bboxs = results["bbox"]
            for j, rect in enumerate(bboxs):
                xmin, ymin, xmax, ymax = rect
                color = (
                    colors[0] if colorSet is None else colors[colorSet[j] %
                                                              len(colors)]
                )
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

        # 复制一张图用于下面操作
        canvas = img.copy()

        # 遍历人体关键点
        for i in range(kptNums):
            for j in range(len(skeletons)):
                # 如果关键点的得分小于阈值则跳过
                if skeletons[j][i, 2] < visualThresh:
                    continue

                if ids is None:
                    color = (
                        colors[i]
                        if colorSet is None
                        else colors[colorSet[j] % len(colors)]
                    )
                else:
                    # 获取rgb三原色数据
                    idx = ids[j] * 3
                    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

                # 将关键点绘制在图像上
                cv2.circle(canvas, tuple(skeletons[j][i, 0:2].astype(
                    "int32")), 2, color, thickness=-1,)

        stickwidth = 2

        # 根据关键点绘制出肢体曲线
        for i in range(NUM_EDGES):
            for j in range(len(skeletons)):
                edge = EDGES[i]
                if (
                    skeletons[j][edge[0], 2] < visualThresh
                    or skeletons[j][edge[1], 2] < visualThresh
                ):
                    continue

                # 复制一张图用于下面操作
                curCanvas = canvas.copy()

                X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
                Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

                # 讲弧度转换为角度
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

                # 利用opencv生成椭圆近似曲线
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)), (int(length / 2),
                                         stickwidth), int(angle), 0, 360, 1
                )
                if ids is None:
                    color = (
                        colors[i]
                        if colorSet is None
                        else colors[colorSet[j] % len(colors)]
                    )
                else:
                    # 获取rgb三原色数据
                    idx = ids[j] * 3
                    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        
                # 填充图形颜色
                cv2.fillConvexPoly(curCanvas, polygon, color)
                # 将画出来的肢体叠加到关键点的图片上
                canvas = cv2.addWeighted(canvas, 0.4, curCanvas, 0.6, 0)
        return canvas

class Detector(object):
    """
    人体检测器
    使用Paddle Inference加载模型并读取配置文件
    """
    def __init__(self, model_dir):
        # 模型路径
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, "model.pdmodel")
        self.params_file = os.path.join(model_dir, "model.pdiparams")

        # 读取配置文件
        with open(os.path.join(model_dir, "infer_cfg.yml"), "r") as f:
            self.inferYml = yaml.safe_load(f)

        # 解析预处理配置
        self._parse_preprocess_config()

        # 初始化预处理操作列表
        self.preprocessOps = []
        for opInfo in self.inferYml.get("Preprocess", []):
            newOpInfo = opInfo.copy()
            opType = newOpInfo.pop("type")
            self.preprocessOps.append(eval(opType)(**newOpInfo))

        # 加载Paddle Inference模型
        self.predictor = self._create_predictor()

        # 获取输入输出名称
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()

    def _parse_preprocess_config(self):
        """解析预处理配置"""
        self.target_size = (320, 320)  # 默认值
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # 从配置文件中读取参数
        for op in self.inferYml.get("Preprocess", []):
            if op.get("type") == "Resize":
                target_size = op.get("target_size", [320, 320])
                self.target_size = tuple(target_size) if isinstance(target_size, list) else (target_size, target_size)
            elif op.get("type") == "NormalizeImage":
                self.mean = op.get("mean", self.mean)
                self.std = op.get("std", self.std)

    def _create_predictor(self):
        """创建Paddle Inference预测器"""
        # 创建Config
        config = Config(self.model_file, self.params_file)

        # 启用GPU推理 (Jetson)
        config.enable_use_gpu(1024, 0)

        # 启用内存优化
        config.enable_memory_optim()

        # 创建预测器
        predictor = create_predictor(config)
        return predictor

    def preprocess(self, imFile, preprocessOps):
        """
        图像预处理
        传入图像并处理
        """
        imInfo = {
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "im_shape": None,
        }

        """
        将图像信息处理成RGB
        Args:
            imFile :输入可以是图像地址或numpy类型的图像
            imInfo :图像信息
        Returns:
            im :  解码后的图像
            imInfo: 处理后的图像信息
        """
        if isinstance(imFile, str):
            with open(imFile, 'rb') as f:
                imRead = f.read()
            data = np.frombuffer(imRead, dtype='uint8')
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = imFile
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imInfo['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        imInfo['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        for operator in preprocessOps:
            im, imInfo = operator(im, imInfo)
        res = {**imInfo, "image": im.astype("float32")}
        return {i: np.expand_dims(res[i], axis=0) for i in res.keys()}

    def preprocess_keypoint(self, imFile):
        """
        关键点检测的预处理 (HRNet专用)
        使用TopDownEvalAffine进行仿射变换
        """
        imInfo = {
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "im_shape": None,
        }

        # BGR转RGB
        if isinstance(imFile, str):
            im = cv2.imread(imFile)
        else:
            im = imFile
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # 记录原始尺寸
        imInfo['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        imInfo['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        # 执行预处理操作
        for operator in self.preprocessOps:
            im, imInfo = operator(im, imInfo)

        res = {**imInfo, "image": im.astype("float32")}
        return {i: np.expand_dims(res[i], axis=0) for i in res.keys()}

    def postprocess(self, result):
        """
        图像后处理
        传入预测结果并处理
        """
        npBoxesNum = result["boxesNum"]
        assert isinstance(
            npBoxesNum, np.ndarray
        ), "`npBoxesNum` should be a `numpy.ndarray`"
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def render(self, img):
        """
        模型预测并绘制
        返回预测结果信息
        """
        # 预处理
        preprocessed = self.preprocess(img, self.preprocessOps)

        # 获取输入tensor
        input_tensor = self.predictor.get_input_handle(self.input_names[0])
        input_tensor.copy_from_cpu(preprocessed["image"])

        # 设置scale_factor (第二个输入)
        if len(self.input_names) > 1:
            scale_tensor = self.predictor.get_input_handle(self.input_names[1])
            scale_tensor.copy_from_cpu(preprocessed["scale_factor"])

        # 设置im_shape (第三个输入)
        if len(self.input_names) > 2:
            im_shape_tensor = self.predictor.get_input_handle(self.input_names[2])
            im_shape_tensor.copy_from_cpu(preprocessed["im_shape"])

        # 执行推理
        self.predictor.run()

        # 获取输出
        boxes_tensor = self.predictor.get_output_handle(self.output_names[0])
        boxes = boxes_tensor.copy_to_cpu()

        # 获取boxes数量 (根据输出格式可能需要从boxes中提取)
        boxesNum = boxes.shape[0] if len(boxes.shape) > 0 else 0
        if boxesNum > 0:
            boxesNum = np.array([boxes.shape[0]], dtype=np.int32)
        else:
            boxesNum = np.array([0], dtype=np.int32)

        return self.postprocess({"boxes": boxes, "boxesNum": boxesNum})
    
    def drawBox(self, result, threshold):
        """
        绘制预测结果
        threshold 行人检测的置信度阈值
        """
        npBoxesNum = result["boxesNum"]
        boxes = result["boxes"]
        startIdx = 0
        filterBoxes = []
        filterNum = []
        for i in range(len(npBoxesNum)):
            boxesNum = npBoxesNum[i]
            boxesI = boxes[startIdx: startIdx + boxesNum, :]
            idx = boxesI[:, 1] > threshold
            filterBoxesI = boxesI[idx, :]
            filterBoxes.append(filterBoxesI)
            filterNum.append(filterBoxesI.shape[0])
            startIdx += boxesNum
        boxes = np.concatenate(filterBoxes)
        filterNum = np.array(filterNum)
        filterRes = {"boxes": boxes, "boxesNum": filterNum}
        return filterRes

class TopDownEvalAffine(object):
    """
    对图像和坐标进行仿射变换
    """
    def __init__(self, trainsize, useUdp=False):
        self.trainsize = trainsize
        self.useUdp = useUdp

    def __call__(self, image, imInfo):
        rot = 0
        imshape = imInfo["im_shape"][::-1]
        center = imInfo["center"] if "center" in imInfo else imshape / 2.0
        scale = imInfo["scale"] if "scale" in imInfo else imshape
        if self.useUdp:
            trans = self.getWarpMatrix(
                rot,
                center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0],
                scale,
            )
            image = cv2.warpAffine(
                image,
                trans,
                (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR,
            )
        else:
            trans = self.getAffineTransform(center, scale, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans,
                (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR,
            )

        return image, imInfo

    """
    This code is based on
        https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py

    计算无偏约束下的变换矩阵。
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): 以度为单位的旋转角度。
        sizeInput (np.ndarray): 输入的图像大小 [w, h].
        sizeDst (np.ndarray): 输出的图像大小 [w, h].
        sizeTarget (np.ndarray): 输入平面ROI的大小 [w, h].

    Returns:
        变换矩阵
    """

    def getWarpMatrix(theta, sizeInput, sizeDst, sizeTarget):
        theta = np.deg2rad(theta)
        matrix = np.zeros((2, 3), dtype=np.float32)
        scaleX = sizeDst[0] / sizeTarget[0]
        scaleY = sizeDst[1] / sizeTarget[1]
        matrix[0, 0] = np.cos(theta) * scaleX
        matrix[0, 1] = -np.sin(theta) * scaleX
        matrix[0, 2] = scaleX * (
            -0.5 * sizeInput[0] * np.cos(theta)
            + 0.5 * sizeInput[1] * np.sin(theta)
            + 0.5 * sizeTarget[0]
        )
        matrix[1, 0] = np.sin(theta) * scaleY
        matrix[1, 1] = np.cos(theta) * scaleY
        matrix[1, 2] = scaleY * (
            -0.5 * sizeInput[0] * np.sin(theta)
            - 0.5 * sizeInput[1] * np.cos(theta)
            + 0.5 * sizeTarget[1]
        )
        return matrix
    def getAffineTransform(self,
        center, inputSize, rot, outputSize, shift=(0.0, 0.0), inv=False
    ):
        assert len(center) == 2
        assert len(outputSize) == 2
        assert len(shift) == 2
        if not isinstance(inputSize, (np.ndarray, list)):
            inputSize = np.array([inputSize, inputSize], dtype=np.float32)
        scaleTmp = inputSize

        shift = np.array(shift)
        srcW = scaleTmp[0]
        dstW = outputSize[0]
        dstH = outputSize[1]

        rotRad = np.pi * rot / 180
        srcDir = self.rotatePoint([0.0, srcW * -0.5], rotRad)
        dstDir = np.array([0.0, dstW * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scaleTmp * shift
        src[1, :] = center + srcDir + scaleTmp * shift
        src[2, :] = self.get3rdPoint(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dstW * 0.5, dstH * 0.5]
        dst[1, :] = np.array([dstW * 0.5, dstH * 0.5]) + dstDir
        dst[2, :] = self.get3rdPoint(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    
    def rotatePoint(self,pt, angleRad):
        assert len(pt) == 2
        sn, cs = np.sin(angleRad), np.cos(angleRad)
        newX = pt[0] * cs - pt[1] * sn
        newY = pt[0] * sn + pt[1] * cs
        rotatedPt = [newX, newY]

        return rotatedPt
    
    def get3rdPoint(self,a, b):
        assert len(a) == 2
        assert len(b) == 2
        direction = a - b
        thirdPt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

        return thirdPt

class Resize(object):
    """
    调整图像进入模型时的大小
    """
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, imInfo):
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        imScaleY, imScaleX = self.generateScale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=imScaleX,
            fy=imScaleY,
            interpolation=self.interp)
        imInfo['im_shape'] = np.array(im.shape[:2]).astype('float32')
        imInfo['scale_factor'] = np.array(
            [imScaleY, imScaleX]).astype('float32')
        return im, imInfo
    
    def generateScale(self, im):
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            imSizeMin = np.min(origin_shape)
            imSizeMax = np.max(origin_shape)
            targeSizeMin = np.min(self.target_size)
            targeSizeMaz = np.max(self.target_size)
            imScale = float(targeSizeMin) / float(imSizeMin)
            if np.round(imScale * imSizeMax) > targeSizeMaz:
                imScale = float(targeSizeMaz) / float(imSizeMax)
            imScaleX = imScale
            imScaleY = imScale
        else:
            resizeH, resizeW = self.target_size
            imScaleY = resizeH / float(origin_shape[0])
            imScaleX = resizeW / float(origin_shape[1])
        return imScaleY, imScaleX
    
class NormalizeImage(object):
    """
    将图像归一化, 方便计算特征值
    """
    def __init__(self, mean, std, is_scale=True, normType="mean_std"):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.normType = normType

    def __call__(self, im, imInfo):
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.normType == "mean_std":
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, imInfo

class Permute(object):
    """
    重新排列图像, 调整图像维度(HWC-->CHW)
    """
    def __init__(
        self,
    ):
        super(Permute, self).__init__()

    def __call__(self, im, imInfo):
        im = im.transpose((2, 0, 1)).copy()
        return im, imInfo
    
if __name__ == "__main__":

    # 加载模型文件, 需要根据情况修改路径
    # #### 代码填空开始处 #### #              
    pose = Tinypose("../../res/models/pedestrianDetection", "../../res/models/poseEstimation")
    # #### 代码填空结束处 #### #  

    # 设置摄像头
    capture = cv2.VideoCapture("/dev/deepCamera")   # 可以设置摄像头
    capture.set(cv2.CAP_PROP_FPS, 30)           # 设置视频的读取速率以及分辨率
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 创建窗口
    cv2.namedWindow("Pose Estimation - Real-time FPS", cv2.WINDOW_NORMAL)

    # FPS 计算变量
    frame_count = 0
    fps = 0.0
    start_time = time.time()
    display_fps = 0.0

    while True:

        ret, frame = capture.read()      # 读取视频

        # 检查是否成功读取视频帧
        if not ret:
            break

        # 开启模型进行人体检测
        res = pose.detector.render(frame)
        res = pose.detector.drawBox(res, 0.5)

        # 开启模型进行人体关键点检测
        res = pose.render(frame, res)

        # 绘制人体关键点和肢体曲线
        frame = pose.drawBox(frame, res, visualThresh=0.4)

        # 计算FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time

        # 每秒更新一次FPS显示
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            display_fps = fps
            frame_count = 0
            start_time = current_time

        # 在画面上显示FPS
        fps_text = f"FPS: {display_fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),  # 绿色
            2,
        )

        cv2.imshow("Pose Estimation - Real-time FPS", frame)

        if cv2.waitKey(1) == 27:                   # 按ESC退出程序
            break
    capture.release()
    cv2.destroyAllWindows()