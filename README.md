# RT-CGH

![捕获4](https://user-images.githubusercontent.com/57349703/175239422-729880a5-2592-4437-8f24-c06616675299.PNG)

Code for real-time interactive CGH using ray tracing: dragon, mirror, refraction.

This is under updating.

## 0.Contents

OptiX 6.5 code for dragon(single mesh), mirror and refraction. 

Additional OptiX 7.4 code for dragon. 

The OptiX 6.5 codes are based on NVIDIA OptiX 6.5 samples. THe Optix 7.4 codes are based on Siggraph  OptiX 7 Course Tutorial Code 
(https://github.com/ingowald/optix7course).

Ptx file is used currently for as a transition before CUDA file is released. CUDA files with GPU code will be released in the next update.

Change it this way, if ptx file is used.

```
const std::string &ptx = "pinhole_camera.cu.ptx";
Program ray_gen_program = context->createProgramFromPTXFile(ptx, "pinhole_camera");
```

## 1. environment setup

We use Visual Studio 2019,CUDA 11.4, OptiX 6.5, OpenCV 3.4.7.

Download CUDA 11.4 from https://developer.nvidia.com/cuda-toolkit

Download Optix 6.5 (NVIDIA developer account required) from https://developer.nvidia.com/rtx/ray-tracing/optix

Download Opencv 3.4.7 (only used to save CGH to png, not essentional for real-time display)from https://opencv.org/releases/

Install. Build OptiX 6.5 samples following official guide entitle "INSTALL-WIN.txt" , it can be found where you install OptiX 6.5

Then there are two ways to run our code:(1) replaced the samples project (2) build your own project. Details will be added in the next update.
