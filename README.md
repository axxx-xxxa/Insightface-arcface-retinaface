# Insightface/Arcface/Retinaface

### Requirement

Tensorrt 8.x

Codes are based on Python 3

### Performance

| Arcface       | AGX  | Nano |
| :------------ | ---- | ---- |
| r50(pth->trt) | 10ms | 70ms |

| Retinaface    | AGX  | Nano  |
| ------------- | ---- | ----- |
| r50(pth->trt) | 17ms | 190ms |

| insightface(detect) | AGX  | Nano |
| ------------------- | ---- | ---- |
| r50(onnx->trt)      | 9ms  | 65ms |



## Retinaface

### Model

#### pth model、wts model、test imgs 

链接：https://pan.baidu.com/s/19ZXRB6WQGW2NuXuCycePxA 
提取码：24dt

####  convert to trt

convert tool

url:[[tensorrtx/retinaface at master · wang-xinyu/tensorrtx (github.com)](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)]

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



## Arcface

### Model

#### pth model、wts model、test imgs 

链接：https://pan.baidu.com/s/1QVqUO3VTFsWimgUzBLUgGA 
提取码：fta5

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





## Insightface

### Model

#### onnx models

链接：https://pan.baidu.com/s/1bL4Dqb85Y9Mg6SDxPd1V3w 
提取码：enkm




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


3.Copy .engine files to {Insightface}/{models} fold
```

### Infer

```
cd {Insightface}
mkdir build && cd build
make
./my_trt_infer -d
```

## Trtexec

Trtexec：

[Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](#work_dynamic_shapes)

[TensorRT/samples/trtexec at main · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

Insightface:

[GitHub - deepinsight/insightface: State-of-the-art 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface) 
