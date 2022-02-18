# Insightface/Arcface/Retinaface in nvidia edge device(this implement agx/nano)




## Easy implement here

build a folder to contain Insightface/Arcface/Retinaface folder (in nvidia device)

![1645151724(1)](https://user-images.githubusercontent.com/79968824/154606835-0e845cca-f2f3-41c9-99c1-5f4615d430cf.png)


### Download(my implement)

baidu cloud download : https://pan.baidu.com/s/1-nCO3Yy02vnUICQ1IA5oUQ
code : ibh2

```
1. find the build folder in subfolder
2. delete build first (delete my exe and build yours)
3. mkdir build && cd build
4. ./{exe} -d

tips:  Following examples described how to get engine file by onnx or pytorch(wts)
```





## Arcface (refer to Tensorrtx-wangxingyu)

[tensorrtx/retina_r50.cpp at master · wang-xinyu/tensorrtx · GitHub](https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/retina_r50.cpp)

### Download(Arcface)

![image](https://user-images.githubusercontent.com/79968824/154606932-1a47f1ec-0663-45d2-91b5-e07a6b354af5.png)

### Run(Arcface)

1.Generate .wts file from mxnet implementation of pretrained model. The following example described how to generate arcface-r50.wts from mxnet implementation of LResNet50E-IR,ArcFace@ms1m-refine-v1.

```
git clone https://github.com/deepinsight/insightface
cd insightface
git checkout 3866cd77a6896c934b51ed39e9651b791d78bb57
cd deploy
// copy tensorrtx/arcface/gen_wts.py to here(insightface/deploy)
// download model-r50-am-lfw.zip and unzip here(insightface/deploy)
python gen_wts.py
// a file 'arcface-r50.wts' will be generated.
// the master branch of insightface should work, if not, you can checkout 94ad870abb3203d6f31b049b70dd080dc8f33fca
// arcface-r100.wts/arcface-mobilefacenet.wts can be generated in similar way from mxnet implementation of LResNet100E-IR,ArcFace@ms1m-refine-v1/MobileFaceNet,ArcFace@ms1m-refine-v1 pretrained model.
```

2.Put .wts file into tensorrtx/arcface, build and run

```
cd tensorrtx/arcface
// download joey0.ppm and joey1.ppm, and put here(tensorrtx/arcface)
mkdir build
cd build
cmake ..
make
sudo ./arcface-r50 -s    // serialize model to plan file i.e. 'arcface-r50.engine'
sudo ./arcface-r50 -d    // deserialize plan file and run inference

or

sudo ./arcface-r100 -s   // serialize model to plan file i.e. 'arcface-r100.engine'
sudo ./arcface-r100 -d   // deserialize plan file and run inference


or

sudo ./arcface-mobilefacenet -s   // serialize model to plan file i.e. 'arcface-mobilefacenet.engine'
sudo ./arcface-mobilefacenet -d   // deserialize plan file and run inference
```



3.Check the output log, latency and similarity score.





## Retinaface (refer to Tensorrtx-wangxingyu)

[tensorrtx/retina_r50.cpp at master · wang-xinyu/tensorrtx · GitHub](https://github.com/wang-xinyu/tensorrtx/blob/master/retinaface/retina_r50.cpp)

### Download(Retinaface)


![image](https://user-images.githubusercontent.com/79968824/154606952-1715b3a6-55a7-4c4f-a755-475fa11ef3c4.png)

### Run(Retinaface)

The following described how to run `retina_r50`. While `retina_mnet` is nearly the same, just generate `retinaface.wts` with `mobilenet0.25_Final.pth` and run `retina_mnet`.

1.generate retinaface.wts from pytorch implementation https://github.com/wang-xinyu/Pytorch_Retinaface

```
git clone https://github.com/wang-xinyu/Pytorch_Retinaface.git
// download its weights 'Resnet50_Final.pth', put it in Pytorch_Retinaface/weights
cd Pytorch_Retinaface
python detect.py --save_model
python genwts.py
// a file 'retinaface.wts' will be generated.
```

2.put retinaface.wts into tensorrtx/retinaface, build and run

```
git clone https://github.com/wang-xinyu/tensorrtx.git
cd tensorrtx/retinaface
// put retinaface.wts here
mkdir build
cd build
cmake ..
make
sudo ./retina_r50 -s  // build and serialize model to file i.e. 'retina_r50.engine'
wget https://github.com/Tencent/FaceDetection-DSFD/raw/master/data/worlds-largest-selfie.jpg
sudo ./retina_r50 -d  // deserialize model file and run inference.
```

3.check the images generated, as follows. 0_result.jpg



## Insightface (refer to trtexec/Tensorrt onnx samples)

Trtexec：

[Developer Guide :: NVIDIA Deep Learning TensorRT Documentation](#work_dynamic_shapes)

[TensorRT/samples/trtexec at main · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec)

Insightface:

[GitHub - deepinsight/insightface: State-of-the-art 2D and 3D Face Analysis Project](https://github.com/deepinsight/insightface)

### Download(Insightface)

onnx models

![image](https://user-images.githubusercontent.com/79968824/154607001-6a75c02c-d346-4667-9f19-96efea8b73ca.png)

### Run(Insightface)

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


3.Copy .engine files to {trt_infer}/{models} fold

4.cd {trt_infer}
	mkdir build && cd build
	make
	./my_trt_infer -d
```

