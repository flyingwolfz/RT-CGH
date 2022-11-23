# RT-CGH

![捕获4](https://user-images.githubusercontent.com/57349703/175239422-729880a5-2592-4437-8f24-c06616675299.PNG)


**<p align="center">
Real-time interactive CGH with ray tracing**
</p>

This repository reproduces the above three experiments. Refer to our paper for detailed introduction.

**Paper: https://doi.org/10.1364/OE.474644   (If it's useful, consider cite our paper!)**

our visualization 3 is selected as IMAGE OF THE WEEK by Optica Publishing Group (Formerly OSA)!

<p align="center">
 <img src="https://user-images.githubusercontent.com/57349703/203493304-9645e8a7-98c8-4b45-88dc-42a1cb68e9a9.PNG" alt="1920" width='20%' height='20%'/>
</p>


## 0.Contents

OptiX 6.5 code for dragon (single mesh), mirror and refraction. 

Additional OptiX 7.4 code for dragon (single mesh). 

The OptiX 6.5 codes are based on NVIDIA OptiX 6.5 samples. The Optix 7.4 codes are based on Siggraph  OptiX 7 Course Tutorial Code 
(https://github.com/ingowald/optix7course).

## 1. Environment setup

We run our code in Windows 10, VS2019/2017.

Download CUDA (10 for OptiX 6, 11 for OptiX 7) from https://developer.nvidia.com/cuda-toolkit

Download OptiX 6.5 (NVIDIA developer account required) from https://developer.nvidia.com/rtx/ray-tracing/optix

Download Opencv 3.4.7 (only used to save CGH to png, not essentional for real-time display)from https://opencv.org/releases/

Install. Build OptiX 6.5 samples following official guide entitle "INSTALL-WIN.txt" , it can be found where you install OptiX 6.5.

For OptiX 7 version, refer to Siggraph  OptiX 7 Course Tutorial Code.

Here are recommended tutorials：https://www.cnblogs.com/chen9510/p/11737941.html  https://blog.csdn.net/novanova2009/article/details/88917889

## 2. Run the code

There are two ways to run our code: 

- 1, Replace the OptiX sample project by our code. Then configure the properties of CUDA and opencv referring to the provided PropertySheet.

- 2, Build your own cuda project. Refer to the provided PropertySheet for OptiX, CUDA ,opencv.

For both methods, **remember set the correct SAMPLE_NAME** (your project name or sample project name), and change the model file path.

## 3.Tips

- 1, Speed tests run using the dragon program. Comment out these lines of code:

```
        float time_GPU;  //from this
	cudaEvent_t start_GPU, stop_GPU;
	cudaEventRecord(start_GPU, 0);
	.............
	for (int i = 0; i < 20; i++)
	{	
       //keep the CGH pipeline
	}
         ...............
	printf("\nThe time for GPU:\t%f(ms)\n", time_GPU/20.0);
	cudaEventDestroy(start_GPU);   
	cudaEventDestroy(stop_GPU);// to the end
```

- 2, In some situations when using OptiX 7, debug mode is much slower, swith to release mode in IDE if it happens. 

- 3, Part of our code is not well organized. Maybe I will organize them and provide more updates in the future.
