# Retinaface /Arcface 

基于retinaface/arcface在边缘设备实现实时人脸检测与识别



### Requirement

jetson nano/tx2/xavierAGX

jetpack( Tensorrt 8.x)

Codes are based on Python 3

### Performance

| Arcface（pth） | AGX  | Nano |
| :------------- | ---- | ---- |
| r50(pth->trt)  | 10ms | 70ms |

| Retinaface（pth） | AGX  | Nano  |
| ----------------- | ---- | ----- |
| r50(pth->trt)     | 17ms | 190ms |

| Arcface（onnx） | AGX  | Nano |
| --------------- | ---- | ---- |
| r50(onnx->trt)  | 9ms  | 65ms |



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

### Run（参考b2b.cpp：串行retinaface+arcface）

1. 新开一个线程（为多路检测做准备）读取摄像头或视频流（opencv等工具），可将通道号等信息与视频帧用结构体打包，用于解析和显示，在主线程获取子线程读到的帧（结构体）进行推理。

2. 主线程串行retinaface+arcface

   2.1 构建engine和执行上下文

   ```
   IRuntime* runtime = createInferRuntime(gLogger);
   ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream_D, size_D);
   IExecutionContext* context = engine->createExecutionContext();
   ```

   2.2 视频帧转为输入data格式

   ```
   static float data[BATCH_SIZE * 3 * INPUT_H_D * INPUT_W_D];
   for (int b = 0; b < BATCH_SIZE; b++) {
        float *p_data = &data[b * 3 * INPUT_H_D * INPUT_W_D];
        for (int i = 0; i < INPUT_H_D * INPUT_W_D; i++) {
            p_data[i] = pr_img.at<cv::Vec3b>(i)[0] - 104.0;
            p_data[i + INPUT_H_D * INPUT_W_D] = pr_img.at<cv::Vec3b>(i)[1] - 117.0;
            p_data[i + 2 * INPUT_H_D * INPUT_W_D] = pr_img.at<cv::Vec3b>(i)[2] - 123.0;
            }
   	}
   ```

   2.3 inference函数

   通用

   ```
   const ICudaEngine& engine = context.getEngine();
   IExecutionContext* context = engine->createExecutionContext();
   context->destroy();
   //call TensorRT’s enqueue method to start inference asynchronously using a CUDA stream
   context.enqueue(batchSize, buffers, stream, nullptr);
   ```

   本文

   ```
   void doInference_D(IExecutionContext& context, float* input, float* output, int batchSize) {
       const ICudaEngine& engine = context.getEngine();
       assert(engine.getNbBindings() == 2);
       void* buffers[2];
       const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME_D);
       const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME_D);
   
       // Create GPU buffers on device
       CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H_D * INPUT_W_D * sizeof(float)));
       CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE_D * sizeof(float)));
   
       // Create stream
       cudaStream_t stream;
       CHECK(cudaStreamCreate(&stream));
   
       // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
       CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H_D * INPUT_W_D * sizeof(float), cudaMemcpyHostToDevice, stream));
       context.enqueue(batchSize, buffers, stream, nullptr);
       CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE_D * sizeof(float), cudaMemcpyDeviceToHost, stream));
       cudaStreamSynchronize(stream);
   
       // Release stream and buffers
       cudaStreamDestroy(stream);
       CHECK(cudaFree(buffers[inputIndex]));
       CHECK(cudaFree(buffers[outputIndex]));
   }
   
   ```

   2.4 推理封装 通过向context(由engine反序列化的执行上下文)输入data(处理好的数据)返回prob(网络输出)

   ```
   static float prob[BATCH_SIZE * OUTPUT_SIZE_D;
   doInference_D(*context, data, prob, BATCH_SIZE);
   ```

3. 通过解析prob对原视频帧进行可视化处理

   ```
   cv::circle/cv::rectangle/cv::putText
   ```

4. 显示及保存

   ```
   cv::imshow ~;
   cv::VideoWriter ~;
   ```

   

### Run（source code ：本文代码）

1. download baidu cloud link files and unzip "b2b"（下载source code）

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
