# RT-CGH

![捕获4](https://user-images.githubusercontent.com/57349703/175239422-729880a5-2592-4437-8f24-c06616675299.PNG)

Code for real-time interactive CGH: dragon, mirror, refraction.

This is under updating.

Current contents:  dragon visualization, mirror visualization based on Optix 6.5 and ptx file.

Next update: cuda files for ray tracing kernels. refraction visualization based on Optix 6.5, additional dragon visualization based on Optix 7.4.

Cuda files or ptx files are required. Change it this way, if ptx file is used.

```
const std::string &ptx = "pinhole_camera.cu.ptx";
Program ray_gen_program = context->createProgramFromPTXFile(ptx, "pinhole_camera");
```
