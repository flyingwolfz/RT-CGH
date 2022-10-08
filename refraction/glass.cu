
#include "tutorial.h"
#include <optixu/optixu_aabb.h>
#include <optix_world.h>

#include "helpers.h"



rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
//
// 
// Dielectric surface shader
//
rtDeclareVariable(float3, cutoff_color, , );
rtDeclareVariable(float, fresnel_exponent, , );
rtDeclareVariable(float, fresnel_minimum, , );
rtDeclareVariable(float, fresnel_maximum, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(int, refraction_maxdepth, , );
rtDeclareVariable(int, reflection_maxdepth, , );
rtDeclareVariable(float3, refraction_color, , );
rtDeclareVariable(float3, reflection_color, , );
rtDeclareVariable(float3, extinction_constant, , );
rtDeclareVariable(float, importance_cutoff, , );
rtDeclareVariable(int, max_depth, , );

rtDeclareVariable(float3, l1, , );

/*
RT_PROGRAM void glass_closest_hit_radiance()
{
	// intersection vectors
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
	const float3 i = ray.direction;                                            // incident direction

	float reflection = 0.5f;
	float3 result = make_float3(0.0f);

	float3 beer_attenuation;
	if (dot(n, ray.direction) > 0) {
		// Beer's law attenuation
		beer_attenuation = exp(extinction_constant * t_hit);
	}
	else {
		beer_attenuation = make_float3(1);
	}

	// refraction
	if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
	{
		float3 t;                                                            // transmission direction
		if (refract(t, i, n, refraction_index))
		{

			// check for external or internal reflection
			float cos_theta = dot(i, n);
			if (cos_theta < 0.0f)
				cos_theta = -cos_theta;
			else
				cos_theta = dot(t, n);

			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

			float importance = prd_radiance.importance * (1.0f - reflection) * optix::luminance(refraction_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, t, RADIANCE_RAY_TYPE, scene_epsilon);
				PerRayData_radiance refr_prd;
				refr_prd.depth = prd_radiance.depth + 1;
				refr_prd.importance = importance;

				rtTrace(top_object, ray, refr_prd);
				result += (1.0f - reflection) * refraction_color * refr_prd.result;
			}
			else {
				result += (1.0f - reflection) * refraction_color * cutoff_color;
			}
		}
		// else TIR
	}

	// reflection
	if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
	{
		float3 r = reflect(i, n);

		float importance = prd_radiance.importance * reflection * optix::luminance(reflection_color * beer_attenuation);
		if (importance > importance_cutoff) {
			optix::Ray ray(h, r, RADIANCE_RAY_TYPE, scene_epsilon);
			PerRayData_radiance refl_prd;
			refl_prd.depth = prd_radiance.depth + 1;
			refl_prd.importance = importance;

			rtTrace(top_object, ray, refl_prd);
			result += reflection * reflection_color * refl_prd.result;
		}
		else {
			result += reflection * reflection_color * cutoff_color;
		}
	}

	result = result * beer_attenuation;
	result += make_float3(0.2f);
	prd_radiance.result = result;
	prd_radiance.thedepth = dot(t_hit*ray.direction, l1);
}
*/
RT_PROGRAM void glass_closest_hit_radiance()
{
	// intersection vectors
	const float3 h = ray.origin + t_hit * ray.direction;            // hitpoint
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
	const float3 i = ray.direction;                                            // incident direction

	float reflection = 0.9f;
	float3 result = make_float3(0.4f);


	if (prd_radiance.type == 0)
	{
		prd_radiance.dep = t_hit;
		prd_radiance.result = result;
	}
	float3 beer_attenuation;
	if (dot(n, ray.direction) > 0) {
		// Beer's law attenuation
		beer_attenuation = exp(extinction_constant * t_hit);
	}
	else {
		beer_attenuation = make_float3(1);
	}

	// refraction
	if (prd_radiance.depth < min(refraction_maxdepth, max_depth))
	{
		float3 t;                                                            // transmission direction
		if (refract(t, i, n, refraction_index))//refract函数计算折射方向t
		{
			// check for external or internal reflection
			float cos_theta = dot(i, n);
			if (cos_theta < 0.0f)
			{
				cos_theta = -cos_theta;
			}
			else
				cos_theta = dot(t, n);
			int inout = 1;//0 in 1 out
			float cos_theta1 = dot(i, n);
			if (cos_theta1 < 0.0f)
			{
				cos_theta1 = -cos_theta1;
				inout = 0;

			}

			float cos_theta2 = dot(t, n);
			if (cos_theta2 < 0.0f)
				cos_theta2 = -cos_theta2;
			float indexratio = 1.0;
			if (inout == 0)
				indexratio = 1.0 / refraction_index;
			else
				indexratio = refraction_index;
			double some = sqrt(1.0 - (indexratio*indexratio - 1.0)*(1.0 / cos_theta2 * cos_theta2 - 1.0));
			//反射系数，折射系数用1-反射系数
			reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);
			//reflection = 0.2f;
			float importance = prd_radiance.importance * (1.0f - reflection) * optix::luminance(refraction_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, t, RADIANCE_RAY_TYPE, scene_epsilon);//折射光线
				PerRayData_radiance refr_prd;
				refr_prd.depth = prd_radiance.depth + 1;
				refr_prd.importance = importance;
				refr_prd.dep = 0.0f;
				refr_prd.type = 1;
				refr_prd.fandep = 0.0f;
				refr_prd.zhedep = 0.0f;
				refr_prd.fanresult = make_float3(0.0f, 0.0f, 0.0f);
				refr_prd.zheresult = make_float3(0.0f, 0.0f, 0.0f);


				rtTrace(top_object, ray, refr_prd);
				if (refr_prd.dep > 0.1f)//击中普通物体
				{
					prd_radiance.zheresult = (1.0f - reflection) * refraction_color * refr_prd.result* beer_attenuation;
					//prd_radiance.zhedep += refr_prd.dep;
					prd_radiance.zhedep += abs(t_hit) + refr_prd.dep*cos_theta2 / cos_theta1 * indexratio * pow(some, 3.0);
					//prd_radiance.zheresult = (1.0f - reflection) * refr_prd.result;
					//prd_radiance.zhedep += abs(t_hit);
				}
				else//击中玻璃，会有反射折射
				{
					float fangray = refr_prd.fanresult.x * 0.3 + refr_prd.fanresult.y *0.59 + refr_prd.fanresult.z * 0.11;
					float zhegray = refr_prd.zheresult.x * 0.3 + refr_prd.zheresult.y *0.59 + refr_prd.zheresult.z * 0.11;

					if (fangray > 0.01f)
					{
						if (zhegray > 0.01f)
						{
							if (fangray > zhegray)
							{
								prd_radiance.zheresult = (1.0f - reflection) * refraction_color *refr_prd.fanresult* beer_attenuation;
								//prd_radiance.zhedep += refr_prd.fandep;
								prd_radiance.zhedep += abs(t_hit) + refr_prd.fandep*cos_theta2 / cos_theta1 * indexratio * pow(some, 3.0);
								//prd_radiance.zheresult = (1.0f - reflection) *refr_prd.fanresult;
								//prd_radiance.zhedep += abs(t_hit);
							}
							else
							{
								prd_radiance.zheresult = (1.0f - reflection) * refraction_color * refr_prd.zheresult* beer_attenuation;
								//prd_radiance.zhedep += refr_prd.zhedep;
								prd_radiance.zhedep += abs(t_hit) + refr_prd.zhedep*cos_theta2 / cos_theta1 * indexratio * pow(some, 3.0);
								//prd_radiance.zheresult = (1.0f - reflection) * refr_prd.zheresult;
								//prd_radiance.zhedep += abs(t_hit);
							}
						}
						else
						{
							prd_radiance.zheresult = (1.0f - reflection) * refraction_color *refr_prd.fanresult* beer_attenuation;
							//prd_radiance.zhedep += refr_prd.fandep;
							prd_radiance.zhedep += abs(t_hit) + refr_prd.fandep*cos_theta2 / cos_theta1 * indexratio * pow(some, 3.0);
							//prd_radiance.zheresult = (1.0f - reflection) *refr_prd.fanresult;
							//prd_radiance.zhedep += abs(t_hit);
						}
					}
					else if (zhegray > 0.01f)
					{
						prd_radiance.zheresult = (1.0f - reflection) * refraction_color *refr_prd.zheresult* beer_attenuation;
						//prd_radiance.zhedep += refr_prd.zhedep;
						prd_radiance.zhedep += abs(t_hit) + refr_prd.zhedep*cos_theta2 / cos_theta1 * indexratio * pow(some, 3.0);
						//prd_radiance.zheresult = (1.0f - reflection) * refr_prd.zheresult;
						//prd_radiance.zhedep += abs(t_hit);
					}
				}

				//result += (1.0f - reflection) * refraction_color * refr_prd.result;
			}
			else {
				//result += (1.0f - reflection) * refraction_color * cutoff_color;
			}
		}


	}

	int rereflect = 1;
	if (rereflect)
	{
		// reflection
		if (prd_radiance.depth < min(reflection_maxdepth, max_depth))
		{
			float3 r = reflect(i, n);

			float importance = prd_radiance.importance * reflection * optix::luminance(reflection_color * beer_attenuation);
			if (importance > importance_cutoff) {
				optix::Ray ray(h, r, RADIANCE_RAY_TYPE, scene_epsilon);
				PerRayData_radiance refl_prd;
				refl_prd.depth = prd_radiance.depth + 1;
				refl_prd.importance = importance;
				refl_prd.fandep = 0.0f;
				refl_prd.zhedep = 0.0f;
				refl_prd.dep = 0.0f;
				refl_prd.type = 1;
				refl_prd.fanresult = make_float3(0.0f, 0.0f, 0.0f);
				refl_prd.zheresult = make_float3(0.0f, 0.0f, 0.0f);

				rtTrace(top_object, ray, refl_prd);



				float fangray = refl_prd.fanresult.x * 0.3 + refl_prd.fanresult.y *0.59 + refl_prd.fanresult.z * 0.11;
				float zhegray = refl_prd.zheresult.x * 0.3 + refl_prd.zheresult.y *0.59 + refl_prd.zheresult.z * 0.11;
				if (refl_prd.dep > 0.1f)
				{
					prd_radiance.fanresult = reflection * reflection_color *refl_prd.result* beer_attenuation;
					//prd_radiance.fanresult = reflection *refl_prd.result;
					prd_radiance.fandep += abs(t_hit) + refl_prd.dep;
					//prd_radiance.fandep += abs(t_hit);
				}
				else
				{
					if (fangray > 0.01f)
					{
						if (zhegray > 0.01f)
						{
							if (fangray > zhegray)
							{
								prd_radiance.fanresult = reflection * reflection_color * refl_prd.fanresult* beer_attenuation;
								prd_radiance.fandep += abs(t_hit) + refl_prd.fandep;
								//prd_radiance.fanresult = reflection *  refl_prd.fanresult;
								//prd_radiance.fandep += abs(t_hit);
							}
							else
							{
								prd_radiance.fanresult = reflection * reflection_color *refl_prd.zheresult* beer_attenuation;
								prd_radiance.fandep += abs(t_hit) + refl_prd.zhedep;
								//prd_radiance.fanresult = reflection * refl_prd.zheresult;
								//prd_radiance.fandep += abs(t_hit);
							}
						}
						else
						{
							prd_radiance.fanresult = reflection * reflection_color *refl_prd.fanresult* beer_attenuation;
							prd_radiance.fandep += abs(t_hit) + refl_prd.fandep;
							//prd_radiance.fanresult = reflection * refl_prd.fanresult;
							//prd_radiance.fandep += abs(t_hit);
						}
					}
					else if (zhegray > 0.01f)
					{
						prd_radiance.fanresult = reflection * reflection_color * refl_prd.zheresult* beer_attenuation;
						prd_radiance.fandep += abs(t_hit) + refl_prd.zhedep;
						//prd_radiance.fanresult = reflection * refl_prd.zheresult;
						//prd_radiance.fandep += abs(t_hit);
					}
				}


				//	result += reflection * reflection_color * refl_prd.result;


			}
			else {
				//result += reflection * reflection_color * cutoff_color;
			}
		}
	}

	//result = result * beer_attenuation;

	//prd_radiance.result = result;
	//prd_radiance.dep = t_hit;
}

//
// (NEW)
// Attenuates shadow rays for shadowing transparent objects
//

rtDeclareVariable(float3, shadow_attenuation, , );

RT_PROGRAM void glass_any_hit_shadow()
{
	float3 world_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float nDi = fabs(dot(world_normal, ray.direction));

	prd_shadow.attenuation *= 1 - fresnel_schlick(nDi, 5, 1 - shadow_attenuation, make_float3(1));

	rtIgnoreIntersection();
}
