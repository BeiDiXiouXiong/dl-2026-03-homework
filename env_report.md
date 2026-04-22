# YOLO 环境配置报告

## 1. 仓库来源
GitHub 地址：https://github.com/ultralytics/yolov5

## 2. 训练入口
训练脚本：train.py
训练命令：
python train.py --data neu_det.yaml --cfg yolov5s.yaml --weights yolov5s.pt --epochs 100

## 3. 环境安装命令
conda create -n yolov5 python=3.9
conda activate yolov5
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## 4. 版本信息
python: 3.9
torch: 2.3.0+cu118
cuda: 11.8
os: Windows 10

## 5. 常见报错与修复
1. 报错：No module named 'torch'
   修复：重新安装 torch

2. 报错：Dataset not found
   修复：检查 neu_det.yaml 中的 path 路径

3. 报错：CUDA out of memory
   修复：减小 batch size 或使用更小模型 yolov5s

4. 报错：label format error
   修复：确保标签为 YOLO 格式（class xc yc w h）