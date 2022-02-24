# Retinaface /Arcface 

基于retinaface/arcface在边缘设备实现实时人脸检测与识别



### Requirement

jetson nano/tx2/xavierAGX

jetpack( Tensorrt 8.x)

Codes are based on Python 3

### Performance

| Arcface       | AGX  | Nano |
| :------------ | ---- | ---- |
| r50(pth->trt) | 10ms | 70ms |

| Retinaface(pth) | AGX  | Nano  |
| --------------- | ---- | ----- |
| r50(pth->trt)   | 17ms | 190ms |

| Retinaface (onnx) | AGX  | Nano |
| ----------------- | ---- | ---- |
| r50(onnx->trt)    | 9ms  | 65ms |



## Retinaface Trail

### Model

#### pth model、wts model、test imgs 

链接：https://pan.baidu.com/s/19ZXRB6WQGW2NuXuCycePxA 
提取码：24dt

pth model is based on RetinaFace in PyTorch:

[wang-xinyu/Pytorch_Retinaface: Retinaface get 80.99% in widerface hard val using mobilenet0.25. (github.com)](https://github.com/wang-xinyu/Pytorch_Retinaface)



####  convert to trt

convert tool

url:[[tensorrtx/retinaface at master · wang-xinyu/tensorrtx (github.com)](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)](https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/retina_r50.cpp)

convert command line

```
git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinaface
// put retinaface.wts here
mkdir build
cd build
cmake ..
make
sudo ./retina_r50 -s  // build and serialize model to file i.e. 'retina_r50.engine'
```

### Infer

infer code:

​	retina_r50.cpp



```
cd tensorrtx/retinaface/build
sudo ./retina_r50 -d  // deserialize model file and run inference.
```

### 



## Arcface Trail

### Model

#### pth model、wts model、test imgs 

链接：https://pan.baidu.com/s/1QVqUO3VTFsWimgUzBLUgGA 
提取码：fta5

pth model is based on mxnet implementation of pretrained model:

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**

####  convert to trt

convert tool

url:[tensorrtx/arcface at master · wang-xinyu/tensorrtx (github.com)](https://github.com/wang-xinyu/tensorrtx/tree/master/arcface)

convert command line

```
cd tensorrtx/arcface
// download joey0.ppm and joey1.ppm, and put here(tensorrtx/arcface)
mkdir build
cd build
cmake ..
make

sudo ./arcface-r50 -s    // serialize model to plan file i.e. 'arcface-r50.engine'

or

sudo ./arcface-r100 -s   // serialize model to plan file i.e. 'arcface-r100.engine'

or

sudo ./arcface-mobilefacenet -s   // serialize model to plan file i.e. 'arcface-mobilefacenet.engine'
```

### Infer

infer code:

​	arcface-r50.cpp(r100,mobilefacenet)



```
cd tensorrtx/arcface/build
sudo ./arcface-r50 -d    // deserialize plan file and run inference

or

sudo ./arcface-r100 -d   // deserialize plan file and run inference

or

sudo ./arcface-mobilefacenet -d   // deserialize plan file and run inference
```

### 





## Onnx dynamic model to Engine Trail

### Model

#### onnx models

链接：https://pan.baidu.com/s/1bL4Dqb85Y9Mg6SDxPd1V3w 
提取码：enkm

onnx models are based on Insightface Pretrained Models

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**



#### convert to trt



```
1.cd /usr/src/tensorrt/samples/trtexec 
	make
	trtexec(有打印相关信息安装完成)
2.cd 模型路径
	trtexec onnx2engine code:

    Dynamic shapes	
    trtexec 
    --onnx=./res18.onnx 
    --explicitBatch 
    --minShapes="input":1x3x224x224 
    --optShapes="input":16x3x224x224 
    --maxShapes="input":32x3x224x224 
    --shapes="input":1x3x224x224 
    --saveEngine=./resnet50_dynamic.engine

    nondynamic	
    trtexec 
    --onnx=./res18.onnx 
    --saveEngine=test.trt 
    --explicitBatch 
    --fp16
    --workspace=4096 
    --buildonly


```



### Trtexec

Trtexec：

[Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](#work_dynamic_shapes)

[TensorRT/samples/trtexec at main · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

Insightface:

[GitHub - deepinsight/insightface: State-of-the-art 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface) 



## Retinaface+Arcface Trail 

(jetson AGX xavier实现)

本程序将company_faces中人脸图片作为源数据，与rtsp流中检测到的人脸对比，若相似度大于阈值，会在人脸上标注名称（英文缩写）。

### Source code

链接：https://pan.baidu.com/s/1M2GTxXNbg_-vODOa2HYajg 
提取码：fjc7

### Model

arcface-mobilefacenet.engine 由 arcface-m.wts(mobilenet) 转换

retina_mnet.engine 由 retina-m.wts(mobilenet) 转换

### Run

1. download baidu cloud link files and unzip "b2b"

2. 

   ```
   cd b2b 
   mkdir build && cd build
   mkdir company_faces (将本地人脸图片放入该路径下)
   make
   (company_faces目录写死，目录中人脸图片请以qys0.jpg,yx_0.jpg格式命名，仅支持0-9) 
   (rtsp已经写死，需要到b2b.cpp里video_cap函数中修改)
   ```

3. 将arcface-mobilefacenet.engine及retina_mnet.engine放到build目录下

4. ./b2b -d

