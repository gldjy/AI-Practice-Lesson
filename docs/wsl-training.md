# WSL 上配置 PaddleDetection 训练环境

本文介绍在 Windows + WSL2 中安装 PaddleDetection 依赖、准备数据、训练、导出与部署推理模型的流程，适用于 NVIDIA 显卡与自定义小数据集。文中 `$REPO` 表示仓库根目录，`$VENV` 表示虚拟环境目录，`$DISTRO` 表示 WSL 发行版名称。

## 1 快速开始

环境配置完成后，执行：

```bash
cd $REPO
$VENV/bin/python tools/check_env.py
$VENV/bin/python vendors/paddet/tools/train.py -c configs/<dataset>/<model>.yml --eval
```

使用 `uv run` 时必须显式指定依赖组，例如 `uv run --group train python ...`；未指定时 uv 会按默认组同步环境并移除训练依赖。首次配置从第 3 节开始。

## 2 仓库结构

`configs/` 存放数据集与模型配置，`tools/` 存放数据转换、环境检查与推理预览脚本，`vendors/paddet/` 是 PaddleDetection 子模块，`data/` 存放数据集且不纳入版本控制。仓库可以位于 Windows 文件系统，通过 `/mnt/<drive>/...` 在 WSL 内访问；虚拟环境建议放在 Linux 文件系统，例如 `~/.venvs/<project>`。PaddlePaddle 2.6.x 的 GPU wheel 不包含 CUDA 与 cuDNN，因此训练在 WSL 中完成，CUDA 默认安装到 `/usr/local/cuda-11.8`。

## 3 环境准备

### 3.1 前提条件

- Windows 已安装 NVIDIA 驱动，且 `nvidia-smi` 正常输出。
- WSL2 与 Ubuntu 发行版已就绪，`wsl -l -v` 中版本号为 2。

WSL 无需单独安装驱动，`nvidia-smi` 会透传 Windows 驱动。使用 `wsl -d $DISTRO -- nvidia-smi` 验证；若失败，问题通常位于 Windows 驱动或 WSL 版本。

### 3.2 安装 CUDA 11.8 与 cuDNN 8.6

PaddlePaddle 2.6.2 的 cu118 wheel 要求 CUDA 11.8 与 cuDNN 8.6；Windows 中为其他框架安装的 12.x 不适用于当前环境。以 root 进入 WSL（`wsl -d $DISTRO -u root`）后执行：

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends \
  cuda-cudart-11-8 libcublas-11-8 libcufft-11-8 libcurand-11-8 \
  libcusolver-11-8 libcusparse-11-8 cuda-nvrtc-11-8
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/*.so* /usr/local/cuda-11.8/lib64/
cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/*.h /usr/local/cuda-11.8/include/
```

### 3.3 补齐无版本号软链接

apt 安装的库只提供 `libcublas.so.11` 这类带版本号文件，而 PaddlePaddle 通过 `dlopen` 查找 `libcublas.so`。cuDNN 的 `.so` 依赖 `libnvrtc.so`，WSL 的 `libcuda.so` SONAME 为 `libcuda.so.1`，需要手动建立链接并加入 ldconfig 缓存。

```bash
cd /usr/local/cuda-11.8/lib64
for f in lib*.so.*; do
    b="${f%%.so.*}"
    [ -e "$b.so" ] || ln -s "$f" "$b.so"
done
ln -sf /usr/lib/wsl/lib/libcuda.so.1 /usr/local/cuda-11.8/lib64/libcuda.so
echo /usr/local/cuda-11.8/lib64 > /etc/ld.so.conf.d/cuda-11-8.conf
ldconfig
ldconfig -p | grep -E "libcudnn.so.8|libcublas.so |libcuda.so |libnvrtc.so "
```

最后一条命令应输出四个库，缺项时检查 3.2 与 3.3 的安装步骤。

### 3.4 安装 Python 环境

运行仓库脚本创建环境，该脚本依次打印硬件信息、拉取子模块、选择依赖组并创建环境：

```bash
cd $REPO
bash tools/setup_env.sh
```

`train` 组用于训练、评估与导出，包含可编辑安装的 `paddledet`；`deploy` 组用于设备端推理，仅包含 PaddlePaddle、OpenCV、NumPy、PyYAML 与 setuptools。脚本在 x86_64 上默认使用 `train`，其他架构默认使用 `deploy`，也可通过 `GROUP=train bash tools/setup_env.sh` 指定。

手动安装命令：

```bash
UV_PROJECT_ENVIRONMENT=$VENV uv sync --group train
UV_PROJECT_ENVIRONMENT=$VENV uv sync --group deploy
```

Python 版本由 `.python-version` 固定为 3.10。`pyproject.toml` 固定 `setuptools<81` 以避免 `pkg_resources` 缺失，并屏蔽 `sklearn==0.0` 空壳包。`paddet` 仅放入 `train` 组。仓库位于 `/mnt` 时由 9p 挂载，创建数万个小文件较慢，脚本会把虚拟环境放到 `~/.venvs/<project>`，并通过 `UV_PROJECT_ENVIRONMENT` 告知 uv。后续命令需保留该变量，或直接使用 `$VENV/bin/python`。

安装完成后执行 `$VENV/bin/python tools/check_env.py`。输出包含 `paddle self-check passed` 与 `import ppdet` 即表示通过。自检最后一步会构建网络并执行 GPU 前向，可发现缺少算子库这类仅导入无法暴露的问题。

## 4 数据集准备

PaddleDetection 使用 COCO 目录结构：

```text
data/<dataset>/{images, annotations/instance_{train,val}.json}
```

转换脚本需满足以下约定：

- 图片放入 `data/<dataset>/images/`。
- 每张图片对应一条 `images` 记录，字段为 `id`、`file_name`、`width`、`height`。
- 每个标注框对应一条 `annotations` 记录，字段为 `id`、`image_id`、`category_id`、`bbox`、`area`、`iscrowd`。
- `bbox` 使用 `[x, y, w, h]`，原点位于左上角。
- `categories` 从 1 开始连续编号。
- 训练集与验证集使用固定随机种子划分。

`tools/prepare_dataset.py` 读取 VOC 风格标注压缩包并输出上述结构。使用 `--augment` 时，每张训练原图还会生成水平翻转、垂直翻转与白平衡三张增强图，命名为 `<stem>_aux_<n>.jpg`，验证集不生成增强图：

```bash
$VENV/bin/python tools/prepare_dataset.py --zip <archive> --out data/<dataset> [--augment]
```

数据集配置位于 `configs/<dataset>/<dataset>_detection.yml`，核心项为 `num_classes`，以及 `TrainDataset`、`EvalDataset` 中的 `dataset_dir`、`anno_path`、`image_dir`。少数类别显示 `-1.000` 表示该类别没有可评估样本，并非配置错误。

## 5 训练

### 5.1 启动训练

```bash
cd $REPO
$VENV/bin/python vendors/paddet/tools/train.py -c configs/<dataset>/<model>.yml --eval
```

`-c` 指定配置文件，`_BASE_` 继承官方模型结构与数据增强配置。`--eval` 表示每轮结束后在验证集计算 COCO mAP。临时参数可通过 `-o TrainReader.batch_size=32` 覆盖，无需修改文件。

### 5.2 相对官方配置的调整

| 项 | 取值 | 说明 |
| --- | --- | --- |
| `num_classes` | 数据集真实类别数 | 写入数据集配置，检测头自动调整 |
| `epoch` | 几百张图片使用几十到两百轮 | 官方 300 轮面向 COCO 十几万张图片，直接套用过拟合 |
| `batch_size` | 以显存为上限，参考 `max_mem_reserved` | 先用小值观察显存占用，再增加至不 OOM |
| `base_lr` | 官方值乘以 batch 比例 | 线性缩放，官方 PicoDet 为 0.32 与 batch 64 |
| `warmup steps` | 几十到几百 | 小数据集上过长会浪费训练预算 |
| `snapshot_epoch` | 总 epoch 的 1/5 到 1/10 | 便于回滚，同时避免产生过多文件 |
| `save_dir` | 独立目录 | 避免 checkpoint 堆积在仓库根目录 |
| `use_ema` | 显存紧张时关闭 | 需要额外保存模型，小数据集收益有限 |

### 5.3 日志与产物

`loss` 为各项损失之和，PicoDet 包含 `vfl`、`bbox`、`dfl`；`ips` 为每秒处理图片数，首个 batch 偏低是 cuDNN 选择 kernel 的开销，应观察后续稳定值；`max_mem_reserved` 为显存池水位，调整 batch size 时以该值为准。前几十轮 mAP 为 0 属于正常现象：检测头重新随机初始化，仅 backbone 加载预训练权重。

训练目录包含 `model_final.pdparams`、`best_model.pdparams` 与 `*.pdopt`。`best_model.pdparams` 用于推理与导出，`*.pdopt` 仅用于断点续训。验证集推理使用 `vendors/paddet/tools/infer.py`，传入 `-c` 与 `weights=<save_dir>/best_model.pdparams`，结果图写入 `output/`；批量推理可用 `--infer_dir` 替代 `--infer_img`。

## 6 导出与实时预览

### 6.1 导出推理模型

训练产物是动态图权重 `.pdparams`，部署需要包含后处理的静态图推理模型。

```bash
$VENV/bin/python vendors/paddet/tools/export_model.py \
  -c configs/<dataset>/<model>.yml \
  -o weights=<save_dir>/best_model.pdparams \
  --output_dir <save_dir>/inference
```

导出目录包含 `model.pdmodel`、`model.pdiparams` 与 `infer_cfg.yml`。`infer_cfg.yml` 保存预处理参数、类别名与 NMS 阈值。更换数据集或修改模型结构后必须重新导出，`label_list` 由配置生成。

### 6.2 实时预览

`tools/preview_detection.py` 仅依赖 Paddle Inference、OpenCV 与 PyYAML，不导入 `paddet`，因此可在 `deploy` 组中运行。

```bash
$VENV/bin/python tools/preview_detection.py --model-dir <save_dir>/inference --source 0
$VENV/bin/python tools/preview_detection.py --model-dir <save_dir>/inference \
  --source clip.mp4 --save out.mp4 --no-show
```

主要参数包括 `--device CPU|GPU`、`--threads`、`--threshold`、`--max-frames`、`--no-show` 与 `--save`。`--threshold` 默认取 `infer_cfg.yml` 中的 `draw_threshold`。运行时按 `q` 或 `Esc` 退出。无可用 X11 或 Wayland 时 `cv2.imshow` 会报错，此时使用 `--no-show --save` 输出文件。

接入其他模型或应用时需注意以下三点：

1. 训练使用 RGB，OpenCV 读取的是 BGR，必须转换；否则红蓝通道颠倒，且 mAP 无法暴露该问题。
2. 静态 shape 导出会将输入尺寸固定在模型中，NMS 输出位于输入尺寸坐标系，需要按原图尺寸与输入尺寸的比例还原。
3. 类别名应从 `infer_cfg.yml` 读取，不应硬编码，否则更换数据集时需要修改代码。

## 7 排障指南

报错文本保持原文，便于直接搜索。

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `nvidia-smi: command not found` | 驱动异常或 WSL 为版本 1 | 用 `wsl -l -v` 确认版本为 2，并检查 Windows 侧命令 |
| `dynamic_loader.cc:314 ... (libcudnn.so)` | cuDNN 未安装 | 按 3.2 安装 |
| `Could not load library libcudnn_ops_infer.so.8. Error: libnvrtc.so` | 缺少无版本号 nvrtc 软链接 | 执行 `ln -s libnvrtc.so.11.8.89 /usr/local/cuda-11.8/lib64/libnvrtc.so` |
| `... (libcublas.so)` 或 `... (libcuda.so)` | 缺少无版本号软链接 | 执行 3.3 中的循环，并建立 `libcuda.so` 链接 |
| `cudnnGetVersion` 处 SIGABRT | 库版本混装或安装未完成 | 清理后重装 cuDNN，执行 `ldconfig` 并重启进程 |
| `No module named 'pkg_resources'` | setuptools 81 起移除该模块 | 保持 `setuptools<81` |
| `uv sync` 后 `import ppdet` 失败 | `paddet` 仅存在于 `train` 组 | 训练机使用 `--group train` |
| 后台训练日志始终为 0 字节 | 输出接入 `tail` 后需等待进程结束 | 重定向到文件，另开窗口执行 `tail -f` |
| `OSError: [Errno 12] Cannot allocate memory` 与 `Blocking queue is killed` | 同卡存在多个训练进程或 WSL 内存不足 | 用 `pgrep -af '[t]rain.py'` 确认唯一进程，清理时用 `pkill -9 -f '[t]rain.py'` |
| `CUDA out of memory` | batch 过大 | 降低 `batch_size`，参考 `max_mem_reserved` |
| 数据加载缓慢 | 图片位于 `/mnt`，跨文件系统读取 | 数据量大时将 `data/` 复制到 Linux 文件系统 |

通用原则是按缺失项补齐依赖。补齐库后执行 `ldconfig` 并启动新进程，已运行的进程不会加载新库。

WSL 默认分配物理内存的一半。需要更多内存时，在 `.wslconfig` 的 `[wsl2]` 段写入 `memory=<size>` 后执行 `wsl --shutdown`。`/mnt` 默认不能创建符号链接，可在 `/etc/wsl.conf` 中启用 `metadata` 选项并重启 WSL。虚拟环境约占 2.5 GB，CUDA 约占 3 GB；vhdx 文件只增不减，清理后需用 `Optimize-VHD` 或 `wsl --manage <distro> --set-sparse true` 回收空间。

## 8 aarch64 设备

aarch64 平台会自动切换为 CPU 版 PaddlePaddle，官方未发布 arm64 GPU wheel。`setup_env.sh` 在该平台默认选择 `deploy` 组。不建议在 aarch64 设备上训练，`tools/preview_detection.py` 的实时检测是此类设备的主要使用场景。
