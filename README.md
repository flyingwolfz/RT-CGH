# RT-CGH

![捕获4](https://user-images.githubusercontent.com/57349703/175239422-729880a5-2592-4437-8f24-c06616675299.PNG)


**<p align="center">
Real-time interactive CGH using ray tracing**
</p>

We firstly achieve real-time CGH with accurate depth precision and large depth range at the same time. (proposed in 2021, finished in 2021.10, repository created in 2022.6, latest update in 2022.7). Refer to our paper for detailed introduction.

**Paper: now under review in Optics Express. (If it's useful, consider cite our paper!)**

## 0.Contents

OptiX 6.5 code for dragon(single mesh), mirror and refraction. 

Additional OptiX 7.4 code for dragon. 

The OptiX 6.5 codes are based on NVIDIA OptiX 6.5 samples. The Optix 7.4 codes are based on Siggraph  OptiX 7 Course Tutorial Code 
(https://github.com/ingowald/optix7course).

**Ptx file is used currently for as a transition before CUDA file is released. CUDA files with GPU code will be released in the next update.**

Change it this way, if ptx file is used.

```
const std::string &ptx = "pinhole_camera.cu.ptx";
Program ray_gen_program = context->createProgramFromPTXFile(ptx, "pinhole_camera");
```

## 1. Environment setup

Download CUDA (10 for OptiX 6, 11 for OptiX 7) from https://developer.nvidia.com/cuda-toolkit

Download OptiX 6.5 (NVIDIA developer account required) from https://developer.nvidia.com/rtx/ray-tracing/optix

Download Opencv 3.4.7 (only used to save CGH to png, not essentional for real-time display)from https://opencv.org/releases/

Install. Build OptiX 6.5 samples following official guide entitle "INSTALL-WIN.txt" , it can be found where you install OptiX 6.5.

For OptiX 7 version, refer to Siggraph  OptiX 7 Course Tutorial Code.

## 2. Run the code

There are two ways to run our code: 

- 1, Replace the OptiX sample project by our code. Then configure the properties of CUDA and opencv refer to the provided PropertySheet.

- 2, Build your own cuda project. Refer to the provided PropertySheet for OptiX, CUDA ,opencv.

For both methods, **remember set the correct SAMPLE_NAME** (your project name or sample project name), and change the model file path.

## 3.Tips

- 1, under updating
