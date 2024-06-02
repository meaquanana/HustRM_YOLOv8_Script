# 目前可支持的训练选项
1、大部分主流的主干(MobileNet1-4、ConvNext、VIT应该有几百种~)

2、部分分类损失函数Focal_loss等 (ultralytics\utils\loss.py)

3、部分定位损失，GIOU、EIOU等 (ultralytics\utils\loss.py)

4、YOLOv10的检测头NMS-FREE(后处理时间基本为0)

5、大部分注意力机制，怎么修改后续会录视频，实在是不好写文档说明
# 环境配置
本项目正常的测试环境 : CUDA11.6 Python3.8 torch1.13.1

**首先必须确保当前环境中不存在ultralytics库，建议pip uninstall ultralytics**
    
    pip install  -r requirements.txt

    pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

    # mmcv可选，建议安装，命令如下，安装失败建议换源
    pip install -U openmim
    mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
    mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
    
# 如何查看可选的主干网络并修改
目前修改主干网络的实现方式主要是timm库，只要是timm库支持的主干网络，均可以修改

使用test_timm.py可查看目前支持的主干网络，也可以参考[https://github.com/huggingface/pytorch-image-models]

修改方式以MobileNetv3为例，运行test_timm.py后，会出现目前timm支持的主干网络，**注意timm中主干网络的命名可能并不是原论文中的命名**，timm库中Mobilenetv3有'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100'，对应不同的模型大小。

我们选择tf_mobilenetv3_large_100，将ultralytics\cfg\models\v8\yolov8-timm.yaml修改如下即可
~~~
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# 0-P1/2
# 1-P2/4
# 2-P3/8
# 3-P4/16
# 4-P5/32

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, tf_mobilenetv3_large_100, [False]]  # 4
  - [-1, 1, SPPF, [1024, 5]]  # 5

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 6
  - [[-1, 3], 1, Concat, [1]]  # 7 cat backbone P4
  - [-1, 3, C2f, [512]]  # 8

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 9
  - [[-1, 2], 1, Concat, [1]]  # 10 cat backbone P3
  - [-1, 3, C2f, [256]]  # 11 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]] # 12
  - [[-1, 8], 1, Concat, [1]]  # 13 cat head P4
  - [-1, 3, C2f, [512]]  # 14 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]] # 15
  - [[-1, 5], 1, Concat, [1]]  # 16 cat head P5
  - [-1, 3, C2f, [1024]]  # 17 (P5/32-large)

  - [[11, 14, 17], 1, Detect, [nc]]  # Detect(P3, P4, P5)
~~~

当然，每个模型的特征层不一样，怎么选取取决于自己，主干网络的特征层同样可以通过运行test_timm.py得到
**配置文件backbone的第一层参数为Ture，则加载预训练权重，False则随机初始化，但是目前由于种种原因，预训练权重没办法通过timm直接下载，所以建议False**

# 修改/添加损失函数
ultralytics\utils\loss.py中可以自己定义，以Detect为例，找到class v8DetectionLoss，修改self.bce即可

# NMSFREE
训练的时候直接调用对应的配置文件即可，目前只支持检测网络

# 自己DIY配置文件
由于模型解析函数已经修改完成，所以可以自由选择组合方式，例如nmsfree+mobilenetv3，OBB，Segment，Pose均可修改，注意层数的匹配即可。

# Attention的添加和以及配置文件的修改