# RT-CGH

under updating

CGH using Optix 6.5 ray tracing engine for three visualizations. 

Additional Optix 7.4 code for dragon visualization.


change it this way, if ptx file is used

```
const std::string &ptx = "pinhole_camera.cu.ptx";
Program ray_gen_program = context->createProgramFromPTXFile(ptx, "pinhole_camera");
```
