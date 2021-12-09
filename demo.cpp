/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 //!
 //! demo.cpp
 //! This file contains the implementation of the ONNX Model sample. It creates the network using
 //! the resnet18.onnx model.
 //! It can be run with the following command line:
 //! Command: ./OnnxTrtDemo [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
 //! [--useDLACore=<int>] [--int8 or --fp16]
 //!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
#include <opencv2/opencv.hpp>
#include<iostream>

//#include "logger.cpp"

#include "image.hpp"

#include "NvInfer.h"
#include "test_time.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>


namespace sample
{
    Logger gLogger{ Logger::Severity::kINFO };
    LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
    LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
    LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
    LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
    LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

    void setReportableSeverity(Logger::Severity severity)
    {
        gLogger.setReportableSeverity(severity);
        gLogVerbose.setReportableSeverity(severity);
        gLogInfo.setReportableSeverity(severity);
        gLogWarning.setReportableSeverity(severity);
        gLogError.setReportableSeverity(severity);
        gLogFatal.setReportableSeverity(severity);
    }
} // namespace sample



const std::string DemoName = "TensorRT_Onnx.demo";

class OnnxTrtClassify
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
    int INPUT_H = 320;
    int INPUT_W = 320;
    int INPUT_C = 3;
    int OUTPUT_SIZE = 320 * 320;
    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "output";

public:
    OnnxTrtClassify(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //! \brief Function builds the network engine
    bool build();

    //! \brief Runs the TensorRT inference engine for this sample
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.
    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model  and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //! \brief Reads the input  and stores the result in a managed buffer
    bool processInput(const samplesCommon::BufferManager& buffers);

    //! \brief Classifies digits and verify result
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};


//!创建网络:配置构建器并创建网络引擎,模型序列化存储
// 如果成功创建了引擎，则返回true；否则返回false 
bool OnnxTrtClassify::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    //Build  Engine and save as plan file
    //The engine needs to be built for the first run, and the if statement is turned on. 
    //After the engine is built, the if statement can be closed and the code block that loads the plan file can be opened.
    if (true)
    {
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

        if (!mEngine)
        {
            return false;
        }
        /*Save Serialize File*/
        const char filename[] = "data/common/u2net_student.plan";
        nvinfer1::IHostMemory* trtModelStream = mEngine->serialize();
        std::ofstream file;
        file.open(filename, std::ios::binary | std::ios::out);
        cout << "writing engine file..." << endl;
        file.write((const char*)trtModelStream->data(), trtModelStream->size());
        cout << "save engine file done" << endl;
        file.close();

        /*Verify that the engine is stored correctly */
        std::fstream file_verify;
        const std::string engineFile = "data/common/u2net_student.plan";
        file_verify.open(engineFile, std::ios::binary | std::ios::in);
        file_verify.seekg(0, std::ios::end);
        int length = file_verify.tellg();

        std::cout << "length:" << length << std::endl;
        file_verify.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file_verify.read(data.get(), length);

        assert(trtModelStream->data == data);
        assert(trtModelStream->size() == length);

        /*Destroy modelstream*/
        trtModelStream->destroy();
    }

    // After the engine is builtand serializedand stored, the plan file can be directly loadedand deserialized
    if (false)
    {
        /*Read plan File and deserializeCudaEngine*/
        const std::string engineFile = "data/common/u2net_student.plan";
        std::fstream file;

        std::cout << "Loading Filename From:" << engineFile << std::endl;

        nvinfer1::IRuntime* trtRuntime;
        file.open(engineFile, std::ios::binary | std::ios::in);
        file.seekg(0, std::ios::end);
        int length = file.tellg();

        std::cout << "Length:" << length << std::endl;

        file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);
        file.close();

        std::cout << "Load Engine Done" << std::endl;
        std::cout << "Deserializing" << std::endl;
        trtRuntime = createInferRuntime(sample::gLogger.getTRTLogger());
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(trtRuntime->deserializeCudaEngine(data.get(), length, nullptr)
            , samplesCommon::InferDeleter());
        std::cout << "Deserialize Done" << std::endl;
        assert(mEngine != nullptr);
        std::cout << "The engine in TensorRT.cpp is not nullptr" << std::endl;
        //DO NOT DESTORY! when verifying
        trtRuntime->destroy();

        ///*Verify the correctness of serialization and deserialization*/
        //nvinfer1::IHostMemory* trtModelStream{ nullptr };
        //trtModelStream = mEngine->serialize();
        //mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(trtRuntime->deserializeCudaEngine(trtModelStream->data(),
        //trtModelStream->size(), nullptr), samplesCommon::InferDeleter());
        //assert(mEngine != nullptr);
        //trtModelStream->destroy();
        //trtRuntime->destroy();
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the  Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the  network
//!
//! \param builder Pointer to the engine builder
//!
bool OnnxTrtClassify::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    //setMaxWorkspaceSize
    config->setMaxWorkspaceSize(2048_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! 运行TensorRT推理引擎
//! 分配缓冲区，设置输入并执行引擎。
//! 
bool OnnxTrtClassify::infer()
{

    // 创建RAII缓冲区管理器对象
    samplesCommon::BufferManager buffers(mEngine, 100);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // 将输入数据读入托管缓冲区
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // 从主机输入缓冲区到设备输入缓冲区的Memcpy
    buffers.copyInputToDevice();

    // 测试推断时间。 循环一千次
    CSpendTime time;
    time.Start();
    for (int i = 0; i < 1000; i++)
    {
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status)
        {
            return false;
        }
    }
    double dTime = time.End();
    sample::gLogInfo << " Time Used " << dTime / 1000 << " ms " << std::endl;;

    // 从设备输出缓冲区到主机输出缓冲区的Memcpy
    buffers.copyOutputToHost();

    // 验证结果
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;

}

//!
//! \brief 读取输入并将结果存储在托管缓冲区中
//!
bool OnnxTrtClassify::processInput(const samplesCommon::BufferManager& buffers)
{
    std::cout << "Loading..." << std::endl;
    cv::Mat image = cv::imread(locateFile("01.jpg", mParams.dataDirs), cv::IMREAD_COLOR);//读取图片
    INPUT_W = image.cols;//读取图片宽度
    INPUT_H = image.rows;//读取图片高度
    if (image.empty()) {
        std::cout << "The input image is empty. Please check....." << std::endl;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);//BGR通道转RGB
   // cout << " INPUT_H: " << INPUT_H << " INPUT_W: " << INPUT_W << endl;//查看图片尺寸大小
    cv::resize(image, image, cv::Size(320, 320), 0, 0, 0);//将原图image大小resize成320*320大小
    //cout << "image_size: " << image.size << endl;//查看resize后大小
    auto* data = normal(image);
    auto* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    for (int i = 0; i < 320 * 320 * INPUT_C; i++)//320*320*3 -> H*W*C
    {
        hostDataBuffer[i] = data[i];//hostDataBuffer里面存的是什么？
    }

    delete data;
    image.release();//释放存储空间
    /*
    std::cout << "Loading..." << std::endl;
    cv::Mat image = cv::imread(locateFile("turkish_coffee.jpg", mParams.dataDirs), cv::IMREAD_COLOR);//读取图片
    if (image.empty()) {
        std::cout << "The input image is empty. Please check....." << std::endl;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);//BGR通道转RGB
    cv::Mat dst = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3); //创建一张黑色的图，每个像素的每个通道都为0
    cv::resize(image, dst, dst.size());//将原图image大小resize成dst大小
    float* data = normal(dst);
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    for (int i = 0; i < INPUT_H * INPUT_W * INPUT_C; i++)
    {
        hostDataBuffer[i] = data[i];
    }

    delete data;
    image.release();
    dst.release();
    */

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool OnnxTrtClassify::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    // BxCxHxW         S0 ===> S5  small ===> large       已经进过sigmoid了
    auto* output = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int stride = 1;
    const int num_kernels = 1;
    const int height = (INPUT_H + stride - 1) / stride;
    const int width = (INPUT_W + stride - 1) / stride;
    const int length = height * width;
    //cout << "实际 height: " << height << " 实际 width: " << width << endl;
    // Mat存储kernel 小的kernel和最大的kernel做logical_and运算
    std::vector<cv::Mat> kernels(num_kernels);
    cv::Mat kernel = cv::Mat(320, 320, CV_32F, (void*)(output + 0 * length), 0);//存储时还是320
    kernels[0] = kernel;
    kernels[0] *= 255;
    cv::resize(kernels[0], kernels[0], cv::Size(INPUT_W, INPUT_H), 0, 0, 0);//将kernel resize成原图大小
    cv::imwrite("test2.png", kernels[0]);//现在已经能够有灰度感，但是背景是（1，1，1）。
    cv::Mat image = cv::imread("test2.png");
    //将（1，1，1）的地方变为（0，0，0）
    cv::Mat img = cv::imread("test2.png");//先读取上一步黑色背景为(1,1,1)的
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            if (img.at<cv::Vec3b>(row, col)[0] == 1 && img.at<cv::Vec3b>(row, col)[1] == 1 && img.at<cv::Vec3b>(row, col)[2] == 1)
                img.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
        }
    }
    cv::imwrite("test2_mask.png", img);//最终结果，黑色变为（0，0，0）
     
    //通过test 和 test2 ，输出最终透明背景的主体：
    cv::Mat mask = cv::imread("test2_mask.png");//读取mask
    cv::Mat input_bgra = cv::imread("01.jpg", cv::IMREAD_COLOR);//读取原图
    cv::cvtColor(input_bgra,input_bgra,CV_BGR2BGRA);//将原图的通道数改成4
    
    for(int y=0;y<input_bgra.rows;++y)
        for (int x = 0; x < input_bgra.cols; ++x)
        {
            cv::Vec3b& pixel_m = mask.at<cv::Vec3b>(y, x);
            cv::Vec4b& pixel_r = input_bgra.at<cv::Vec4b>(y, x);
            pixel_r.val[3] = pixel_m.val[0];
        }
    cv::imwrite("test2_result.png", input_bgra);
 
    //cv::threshold(kernel, kernel, 0 , 255, cv::THRESH_BINARY);//二值化处理
    //kernel.convertTo(kernel, CV_8U);//格式转换
    //cv::bitwise_and(kernel, kernel, kernel);//掩膜mask
    //kernels[0] = kernel;
    //cv::imwrite("mask0.png", kernels[0]);


    /*
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};
     // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
       sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
       if (val == output[i])
        {
            idx = i;
        }
        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] *10)), '*')
                         << std::endl;
    }
    sample::gLogInfo <<" Result "<< idx<<" "<< output[idx] << std::endl;;

    return idx;

    */
    return true;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/common/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "u2net_student.onnx";
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./demo_onnx [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
        "multiple times to add multiple directories. If no data directories are given, the default is to use "
        "(data/common/)"
        << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
        "where n is the number of DLA engines on the platform."
        << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(DemoName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    OnnxTrtClassify sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx Network" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    system("pause");

    return sample::gLogger.reportPass(sampleTest);

}
