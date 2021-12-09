# TensorRT部署

**Tensorrt版本**：TensorRT-7.2.3.4.Windows10.x86_64.cuda-11.1.cudnn8.1

## Win10 Tensorrt  安装

<!--配置前保证CUDA11.1、cudnn8.1、opencv3.x安装成功并配置好环境变量-->

1. 去这个地方下载对应的版本 https://developer.nvidia.com/nvidia-tensorrt-7x-download
2. 下载完成后，解压。
3. 将 TensorRT-7.2.3.4\include中头文件 copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include
4. 将TensorRT-7.2.3.4\lib 中所有lib文件 copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64
5. 将TensorRT-7.2.3.4\lib 中所有dll文件copy 到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin
6. 用VS2019 打开 TensorRT-7.2.3.4\samples\sampleMNIST\sample_mnist.sln
7. 实测Release版本可直接用，Debug版本配置有些问题。
8. 用anaconda虚拟环境 进入TensorRT-7.2.3.4\data\mnist 目录，执行python download_pgms.py
9. 进入TensorRT-7.2.3.4\bin，用cmd执行，sample_mnist.exe --datadir = d:\path\to\TensorRT-7.2.3.4\data\mnist\
10. 执行成功则说明tensorRT 配置成功 

参考链接：https://arleyzhang.github.io/articles/7f4b25ce/

## onnx模型的转换

**测试模型** 

`torchvision`中的`resnet18`，输入`[1,3,224,224]`输出FC层换成`[1,5]`，五个结果值便于直观对比输出结果的一致性。

**主要代码** 

```python
torch.onnx.export(model, input, ONNX_FILE_PATH,input_names=["input"], output_names=["output"], export_params=True)
```

## onnx模型的调用

**pytorch转onnx**

```powershell
cd torch_to_onnx
python torch_to_onnx.py
```

**TensorRT中的模式：**

**INT8** 和 **fp16**模式

INT8推理仅在具有6.1或7.x计算能力的GPU上可用，并支持在诸如ResNet-50、VGG19和MobileNet等NX模型上进行图像分类。 

**DLA**模式 

DLA是NVIDIA推出的用于专做视觉的部件，一般用于开发板 Jetson AGX Xavier ，Xavier板子上有两个DLA，定位是专做常用计算(Conv+激活函数+Pooling+Normalization+Reshape)，然后复杂的计算交给Volta GPU做。DLA功耗很低，性能很好。参考https://zhuanlan.zhihu.com/p/71984335

## VS2019工程的配置

首推Release版本，可直接用，Debug版本配置有些问题。

属性->调试->命令参数->--fp16(根据需求选择--int8模式还是--fp16)

属性->VC++目录->包含目录->

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include;
D:\TensorRT-7.2.3.4\include;
D:\TensorRT-7.2.3.4\samples\common;
D:\TensorRT-7.2.3.4\samples\common\windows;
D:\opencv\build\include;
D:\opencv\build\include\opencv;
D:\opencv\build\include\opencv2;$(IncludePath)
```

属性->VC++目录->库目录->

```powershell
D:\opencv\build\x64\vc15\lib;
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64;$(LibraryPath)
```

属性->C/C++->附加包含目录->

```powershell
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\include;%(AdditionalIncludeDirectories)
```

属性->链接器->输入->(自行删减)

```powershell
opencv_world342.lib;OpenCL.lib;cudnn_adv_infer64_8.lib;cudnn_ops_train64_8.lib;nppicc.lib;nvinfer.lib;cublas.lib;cudnn_adv_train.lib;cufft.lib;nppidei.lib;nvinfer_plugin.lib;cublasLt.lib;cudnn_adv_train64_8.lib;cufftw.lib;nppif.lib;nvjpeg.lib;cuda.lib;cudnn_cnn_infer.lib;curand.lib;nppig.lib;nvml.lib;cudadevrt.lib;cudnn_cnn_infer64_8.lib;cusolver.lib;nppim.lib;nvonnxparser.lib;cudart.lib;cudnn_cnn_train.lib;cusolverMg.lib;nppist.lib;nvparsers.lib;cudart_static.lib;cudnn_cnn_train64_8.lib;cusparse.lib;nppisu.lib;nvptxcompiler_static.lib;cudnn.lib;cudnn_ops_infer.lib;myelin64_1.lib;nppitc.lib;nvrtc.lib;cudnn64_8.lib;cudnn_ops_infer64_8.lib;nppc.lib;npps.lib;cudnn_adv_infer.lib;cudnn_ops_train.lib;nppial.lib;nvblas.lib;%(AdditionalDependencies)
```
