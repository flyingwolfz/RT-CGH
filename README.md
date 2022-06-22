# RT-CGH

under updating


Current contents:  dragon visualization, mirror visualization based on Optix 6.5 and ptx file.

Next update: refraction visualization based on Optix 6.5, dragon visualization based on Optix 7.4, cuda files for ray tracing kernels.

change it this way, if ptx file is used

```
const std::string &ptx = "pinhole_camera.cu.ptx";
Program ray_gen_program = context->createProgramFromPTXFile(ptx, "pinhole_camera");
```
