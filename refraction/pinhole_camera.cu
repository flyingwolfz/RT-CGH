/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix_world.h>
//#include <common.h>
#include "helpers.h"
#include "tutorial.h"
using namespace optix;



rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float3, bg_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(float,z, , );
rtDeclareVariable(float, juli, , );
rtBuffer<float, 2>              output_buffer;
//rtBuffer<float2, 2>              xiang2;
//rtBuffer<float, 3>              xiang;
//rtBuffer<float, 3>              xiang1;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );


rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
//rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, l1, , );
rtDeclareVariable(float, fencengshu, , );
rtDeclareVariable(float, maxextent, , );
rtDeclareVariable(float, len, , );


rtBuffer<float, 2>              random;

rtBuffer<float2, 3>     complex;

#define Pitch  0.008
#define Lambda 639e-6
#define PI 3.1415926

RT_PROGRAM void pinhole_camera()
{
	
	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x*U + d.y*V + W);

	optix::Ray rayy = optix::make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.dep = 0.0f;
	prd.depth = 0;
	prd.fandep = 0.0f;
	prd.zhedep = 0.0f;
	prd.type = 0;

	rtTrace(top_object, rayy, prd);
	uint3 index;
	index.x = (uint)launch_index.x;
	index.y = (uint)launch_index.y;
	index.z = (uint)0;

	for (int i = 0; i < fencengshu; i++)
	{
		index.z = (uint)i;
		complex[index] = make_float2(0.0f, 0.0f);
	}
	float ggray = 0.0f;
	int x = launch_index.x - launch_dim.x / 2;
	int y = launch_index.y - launch_dim.y / 2;
	if (prd.dep > 0.1f)
	{
	  float cengshu = floor((prd.dep - len + maxextent / 2.0f) / maxextent * fencengshu);

	   float f = z + juli * cengshu;
	   float deltax = Lambda * f / launch_dim.x / Pitch;
	   float deltay = Lambda * f / launch_dim.y / Pitch;
	   float gray = prd.result.x * 0.3 + prd.result.y *0.59 + prd.result.z * 0.11;
	   float xiangwei = PI / Lambda / f * (x*deltax*x*deltax + y*deltay*y*deltay+random[launch_index]);
	   index.z = cengshu;
	   if(cengshu<fencengshu)
	   complex[index] = make_float2(gray * cos(xiangwei), gray * sin(xiangwei));
	   ggray += gray;

    }
	if (prd.fandep > 0.1f)
	{
		float cengshu = floor((prd.fandep - len + maxextent / 2.0f) / maxextent * fencengshu);

		float f = z + juli * cengshu;
		float deltax = Lambda * f / launch_dim.x / Pitch;
		float deltay = Lambda * f / launch_dim.y / Pitch;
		float gray = prd.fanresult.x * 0.3 + prd.fanresult.y *0.59 + prd.fanresult.z * 0.11 + 0.2f;
		float xiangwei = PI / Lambda / f * (x*deltax*x*deltax + y * deltay*y*deltay + random[launch_index]);
		index.z = cengshu;
		if (cengshu < fencengshu)
		{
			complex[index] = make_float2(gray * cos(xiangwei), gray * sin(xiangwei));
			ggray += gray;
		}

	}
	if (prd.zhedep > 0.1f)
	{
		float cengshu = floor((prd.zhedep - len + maxextent / 2.0f) / maxextent * fencengshu);

		float f = z + juli * cengshu;
		float deltax = Lambda * f / launch_dim.x / Pitch;
		float deltay = Lambda * f / launch_dim.y / Pitch;
		float gray = prd.zheresult.x * 0.3 + prd.zheresult.y *0.59 + prd.zheresult.z * 0.11 + 0.2f;
		float xiangwei = PI / Lambda / f * (x*deltax*x*deltax + y * deltay*y*deltay + random[launch_index]);
		index.z = cengshu;
		if (cengshu < fencengshu)
		{
			complex[index] = make_float2(gray * cos(xiangwei), gray * sin(xiangwei));
			ggray += gray;
		}

	}
	//ggray = prd.result.x * 0.3 + prd.result.y *0.59 + prd.result.z * 0.11;
	// output_buffer[launch_index] = ggray;
 
}

RT_PROGRAM void exception()
{
  rtPrintExceptionDetails();
 // output_buffer[launch_index] = make_color( bg_color );
  output_buffer[launch_index] = 0.6f;
}

rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, ambient_light_color, , );
rtBuffer<BasicLight> lights;

RT_PROGRAM void closest_hit()
{
    float3 world_shade_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 ffnormal = faceforward(world_shade_normal, -ray.direction, world_shade_normal);
	//首先将颜色设定为 系数*环境光
	float3 color = Ka * ambient_light_color;
    float3 hit_point = ray.origin + t_hit * ray.direction;
	//每有一个光源就循环一次，计算对应的散射光
	for (int i = 0; i < lights.size(); ++i) {
		BasicLight light = lights[i];
		float3 L = normalize(light.pos - hit_point);
		float nDl = dot(ffnormal, L);
		if (nDl > 0)
		 //如果cos大于0，散射光由半兰伯特定律计算
		 //环境光+散射系数*散射光
		 color += Kd * nDl * light.color;
	}
	
	prd_radiance.result = color;
	
	prd_radiance.dep = dot(t_hit*ray.direction,l1);
	
}

RT_PROGRAM void miss()
{
	prd_radiance.result = bg_color;
	prd_radiance.depth = -1.0f;

}


RT_PROGRAM void holo()
{
	float real = 0.0;
	float image = 0.0;
	float c = 0.0;
	float s = 0.0;
	uint3 index;
	index.x = (uint)launch_index.x;
	index.y = (uint)launch_index.y;
	index.z = (uint)0;
	int x = launch_index.x - launch_dim.x/2;
	int y = launch_index.y - launch_dim.y / 2;
	//c = 1;
	//s = 1;
	for (int i = 0; i < fencengshu; i++)
	{
		float xiangg = PI / Lambda / (z + i * juli) *(x*x *Pitch*Pitch + y*Pitch*y*Pitch);
		index.z = (uint)i;
		//c=cos(xiang[index]);
		//s=sin(xiang[index]);
		c = cos(xiangg);
		s = sin(xiangg);
		real = real + c * complex[index].x - s * complex[index].y;
		image = image + s * complex[index].x + c * complex[index].y;
		//real = real + xiang[index].x * complex[index].x - xiang[index].y * complex[index].y;
		//image = image + xiang[index].y * complex[index].x + xiang[index].x * complex[index].y;
	}

	float jiao = atan2(image, real);
	if (jiao < 0)
	{
		jiao = jiao + 2.0 * PI;
	}
	float grayy = jiao / 2.0 / PI;
	
	output_buffer[launch_index] = grayy;
	

}



