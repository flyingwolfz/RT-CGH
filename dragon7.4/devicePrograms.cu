// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <optix_device.h>
#include "random.h"
#include "LaunchParams.h"
#define PI 3.1415926
using namespace osc;

namespace osc {

    struct rayload
    {
        vec3f color;
        float length;
    };
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  // for this simple example, we have a single ray type
  enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

  static __forceinline__ __device__
      void* unpackPointer(uint32_t i0, uint32_t i1)
  {
      const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
      void* ptr = reinterpret_cast<void*>(uptr);
      return ptr;
  }

  static __forceinline__ __device__
      void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
  {
      const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
      i0 = uptr >> 32;
      i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T* getPRD()
  {
      const uint32_t u0 = optixGetPayload_0();
      const uint32_t u1 = optixGetPayload_1();
      return reinterpret_cast<T*>(unpackPointer(u0, u1));
  }

  static __forceinline__ __device__ rayload* getPRD()
  {
      const unsigned int u0 = optixGetPayload_0();
      const unsigned int u1 = optixGetPayload_1();
      return reinterpret_cast<rayload*>(unpackPointer(u0, u1));
  }
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__radiance()
  { 
     

      const TriangleMeshSBTData& sbtData
          = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

      const int   primID = optixGetPrimitiveIndex();
      const vec3i index = sbtData.index[primID];

      const float u = optixGetTriangleBarycentrics().x;
      const float v = optixGetTriangleBarycentrics().y;

      vec3f N;
      if (sbtData.normal) {
          N = (1.f - u - v) * sbtData.normal[index.x]
              + u * sbtData.normal[index.y]
              + v * sbtData.normal[index.z];
      }
      else {
          const vec3f& A = sbtData.vertex[index.x];
          const vec3f& B = sbtData.vertex[index.y];
          const vec3f& C = sbtData.vertex[index.z];
          N = normalize(cross(B - A, C - A));
      }
      N = normalize(N);

      vec3f diffuseColor = sbtData.color;
      if (sbtData.hasTexture && sbtData.texcoord) {
          const vec2f tc
              = (1.f - u - v) * sbtData.texcoord[index.x]
              + u * sbtData.texcoord[index.y]
              + v * sbtData.texcoord[index.z];

          vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
          diffuseColor *= (vec3f)fromTexture;
      }

      const vec3f rayDir = optixGetWorldRayDirection();
      const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, N));

      //vec4f& prd = *(vec4f*)getPRD<vec4f>();
      rayload& prd = *(rayload*)getPRD<rayload>();
      //rayload* prd = getPRD();
      const float thit = optixGetRayTmax();
      prd.color = cosDN * diffuseColor;
      prd.length = thit;

      //prd.x = (cosDN * diffuseColor).x;
     
      //prd = vec4f(cosDN * diffuseColor, thit);
      //prd.w = thit;
    
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }


  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
      //vec4f& prd = *(vec4f*)getPRD<vec4f>();
      // set to constant white as background color
      //prd = vec4f(0.f);
  }



  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
  
    const auto& camera = optixLaunchParams.camera;
    //vec3f pixelColorPRD = vec3f(0.f);
    rayload  raydata;
    raydata.color = vec3f(0.f);
    raydata.length = 0.0f;


    vec_t<float, 8> a;
    uint32_t u0, u1;

    packPointer(&raydata, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix + .5f, iy + .5f)
        / vec2f(optixLaunchParams.frame.size));

    // generate ray direction
    vec3f rayDir = normalize(camera.direction
        + (screen.x - 0.5f) * camera.horizontal
        + (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable,
        camera.position,
        rayDir,
        0.f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1);
    float extent = camera.extent;

    float depth = 0.0f;
    float normdepth = 0.0f;

    vec3f zdirection= camera.lookat-camera.position;
    const float leng = length(zdirection);
    zdirection = normalize(zdirection);
    depth=dot(raydata.length* rayDir, zdirection);
  

   

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    uint32_t fullIndex = 0;
    for (int i = 0; i < optixLaunchParams.holoparams.layernum; i++)
      {
        fullIndex = i * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + fbIndex;
        optixLaunchParams.complex_prt[fullIndex] = make_float2(0.0f, 0.0f);
      }

    if (depth > 0)
    {
       // normdepth = 1.0f - (depth + 0.5f * extent - leng) / extent;
        float cengshu = floor((depth + 0.5f * extent - leng) / extent * optixLaunchParams.holoparams.layernum);
        float f = optixLaunchParams.holoparams.f + optixLaunchParams.holoparams.juli * cengshu;

        float rr = float(raydata.color.x);

        float deltax = optixLaunchParams.holoparams.lambda * f / optixGetLaunchDimensions().x / optixLaunchParams.holoparams.pitch;
        float deltay = optixLaunchParams.holoparams.lambda * f / optixGetLaunchDimensions().y / optixLaunchParams.holoparams.pitch;;

        int x = ix - optixGetLaunchDimensions().x / 2;
        int y = iy - optixGetLaunchDimensions().y / 2;

        const int    subframe_index = optixLaunchParams.frame_index;
        //unsigned int seed = tea<4>(ix + iy * optixLaunchParams.frame.size.x, subframe_index);
        unsigned int seed = tea<4>(ix + iy * optixLaunchParams.frame.size.x, 1);
        float rand = rnd(seed);
        //float rand = 1;

        float xiangwei = PI / optixLaunchParams.holoparams.lambda / optixLaunchParams.holoparams.f * (x * deltax * x * deltax + y * deltay * y * deltay + 2 * PI * rand);
        fullIndex = cengshu * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + fbIndex;
        optixLaunchParams.complex_prt[fullIndex].x = rr * cos(xiangwei);
        optixLaunchParams.complex_prt[fullIndex].y = rr * sin(xiangwei);

    }
   
    //int r = 0, g = 0, b = 0;
    /*  if (normdepth > 0.5f)
      {*/
    //r = int(255.99f * normdepth);
    //g = int(255.99f * normdepth);
   // b = int(255.99f * normdepth);
    /* }
     else
     {
         r = int(255.99f * raydata.color.x);
         g = int(255.99f * raydata.color.y);
         b = int(255.99f * raydata.color.z);
     }
  */
  }
  extern "C" __global__ void __raygen__holo()
  {

      const int ix = optixGetLaunchIndex().x;
      const int iy = optixGetLaunchIndex().y;
      int r2 = 255, g2 =255, b2 = 255;

     
      // and write to frame buffer ...
      const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

      int x = ix - optixGetLaunchDimensions().x / 2;
      int y = iy - optixGetLaunchDimensions().y / 2;
      float Pitch = optixLaunchParams.holoparams.pitch;
      float f = 0;
      float c, s, real=0.0f, image=0.0f, xiang;

      uint32_t fullIndex = 0;
      for (int i = 0; i < optixLaunchParams.holoparams.layernum; i++)
      {
          fullIndex = i * optixLaunchParams.frame.size.x * optixLaunchParams.frame.size.y + fbIndex;
          f = optixLaunchParams.holoparams.f + optixLaunchParams.holoparams.juli * i;
          xiang = PI / optixLaunchParams.holoparams.lambda / f * (x * x * Pitch * Pitch + y * Pitch * y * Pitch);
          c = cos(xiang);
          s = sin(xiang);
          real += c * optixLaunchParams.complex_prt[fullIndex].x - s * optixLaunchParams.complex_prt[fullIndex].y;
          image += s * optixLaunchParams.complex_prt[fullIndex].x + c * optixLaunchParams.complex_prt[fullIndex].y;
      }  

      float jiao = atan2(image, real);
      if (jiao < 0)
      {
          jiao = jiao + 2.0 * 3.1415926;
      }
      float grayy = jiao / 2.0 / 3.1415926;

      r2 = int(255.99f * grayy);
      g2 = int(255.99f * grayy);
      b2 = int(255.99f * grayy);
      const uint32_t rgba2 = 0xff000000
          | (r2 << 0) | (g2 << 8) | (b2 << 16);

      optixLaunchParams.frame.colorBuffer[fbIndex] = rgba2;

  }
  
} // ::osc
