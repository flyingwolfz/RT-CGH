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


#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include<GL/freeglut.h>
#include<optixu/optixpp.h>
#include"optixu_aabb_namespace.h"
#include<sutil.h>
#include<cufft.h>
#include<cuda_runtime.h>
#include<Arcball.h>
#include<OptiXMesh.h>
#include"common.h"
#include<fstream>
#include<iostream>
#include<cfloat>
#include<cstdlib>
#include<cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  
using namespace optix;

const char* const SAMPLE_NAME = "mirror";

//---------------
//
// Globals
//
//---------------
int debugmode=0;  

float fencengshu =3.0f;
float3 l1;
float dep1;
float z = 300.0f;
float juli = 0.3f;
optix::Context        context;
uint32_t       width  = 1024u;
uint32_t       height =1024u;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;
optix::Aabb    aabb;

cv::Mat out(height, height, CV_8UC1, 1);

cufftHandle forward_plan;
// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

cufftComplex *complex_prt;
float *outbuffer_prt;
float *xiang_prt;
struct RenderBuffers
{
	Buffer complex;
	int optix_device_ordinal;
};
RenderBuffers complex;
RenderBuffers outbuffer;
RenderBuffers xiang;
//------------------------------------------------------------------------------
//
// Forward decls 
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadMesh( const std::string& filename );
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );

void calculateCameraVariables(float3 eye, float3 lookat, float3 up,
	float  fov, float  aspect_ratio,
	float3& U, float3& V, float3& W, bool fov_is_vertical);
Matrix4x4 myrotate(const float2& from, const float2& to);
//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

int initSingleDevice()
{
	std::vector<int> devices = context->getEnabledDevices();
	// Limit to single device
	if (devices.size() > 1) {
		context->setDevices(devices.begin(), devices.begin() + 1);
		char name[256];
		context->getDeviceAttribute(devices[0], RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name);
		std::cerr << "Limiting to device: " << name << std::endl;
	}
	int ordinal = -1;
	context->getDeviceAttribute(devices[0], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal);

	cudaSetDevice(ordinal);

	return devices[0]; // Return OptiX device ordinal, NOT the CUDA device ordinal
}

Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}

void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 2 );
  
	context["fencengshu"]->setFloat(fencengshu);
    context["scene_epsilon"    ]->setFloat( 1.e-4f );
	context["ambient_light_color"]->setFloat(0.7f, 0.7f, 0.7f);
	context["Ka"]->setFloat(0.8f, 0.8f, 0.8f);
	context["Kd"]->setFloat(0.9f, 0.9f, 0.9f);
	context["z"]->setFloat(z);
	context["juli"]->setFloat(juli);

	outbuffer.complex = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width, height);

    context["output_buffer"]->set(outbuffer.complex);

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString(ptx, "miss" ) );
    context["bg_color"]->setFloat( 0.0f, 0.0f, 0.0f );

	complex.complex= context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT2, width, height, fencengshu);
	context["complex"]->set(complex.complex);


	Program holo = context->createProgramFromPTXString(ptx, "holo");
	context->setRayGenerationProgram(1, holo);
	
	complex_prt = static_cast<cufftComplex*>(complex.complex->getDevicePointer(complex.optix_device_ordinal));
	outbuffer_prt = static_cast<float*>(outbuffer.complex->getDevicePointer(outbuffer.optix_device_ordinal));
	
	Buffer bufferr = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, width, height);
	context["random"]->set(bufferr);

}


void loadMesh( const std::string& filename )
{
    OptiXMesh mesh;
    mesh.context = context;
    mesh.use_tri_api = use_tri_api;
    mesh.ignore_mats = ignore_mats;

	const char *ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
	Program c= context->createProgramFromPTXString(ptx, "closest_hit");
	mesh.closest_hit = c;

    loadMesh( filename, mesh ); 

    aabb.set( mesh.bbox_min, mesh.bbox_max );
	

	Transform transform = context->createTransform();
	Transform transform2 = context->createTransform();

	float theta = 3.14159 / 2.0 /2.0;

	float rotate[16] = {
	cos(theta), 0.0f, sin(theta), 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	-sin(theta), 0.0f, cos(theta), 0.0,
	0.0f, 0.0f, 0.0f, 1.0f
	};

	transform->setMatrix(false, rotate, NULL);

    GeometryGroup geometry_group = context->createGeometryGroup();
    geometry_group->addChild( mesh.geom_instance );
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

	transform->setChild(geometry_group);




	// floor geo
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	parallelogram->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "bounds"));
	parallelogram->setIntersectionProgram(context->createProgramFromPTXString(ptx, "intersect"));
	//float3 anchor = make_float3(-0.2f, -0.05f, -0.2f);
	//float3 v1 = make_float3(0.3f, 0.0f, 0.0f);
	//float3 v2 = make_float3(0.0f, 0.3f, 0.0f);
	float3 anchor = make_float3(-0.19f, -0.04f, -0.2f);
	float3 v1 = make_float3(0.26f, 0.0f, 0.0f);
	float3 v2 = make_float3(0.0f, 0.26f, 0.0f);
	float3 normal = cross(v1, v2);
	normal = normalize(normal);
	float d = dot(normal, anchor);
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	float4 plane = make_float4(normal, d);
	parallelogram["plane"]->setFloat(plane);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);
	parallelogram["anchor"]->setFloat(anchor);


	// checker material
	ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
	Program check_ch = context->createProgramFromPTXString(ptx, "plane_closest_hit");
	//Program check_ah = context->createProgramFromPTXString(ptx, "any_hit_shadow");
	Material floor_matl = context->createMaterial();
	floor_matl->setClosestHitProgram(0, check_ch);
	//floor_matl->setAnyHitProgram(1, check_ah);

	floor_matl["Kd"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_matl["Ka"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_matl["Ks"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["Kd2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_matl["Ka2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_matl["Ks2"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["inv_checker_size"]->setFloat(10.0f, 10.0f, 1.0f);
	floor_matl["phong_exp1"]->setFloat(0.0f);
	floor_matl["phong_exp2"]->setFloat(0.0f);
	floor_matl["Kr1"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_matl["Kr2"]->setFloat(0.0f, 0.0f, 0.0f);



	// floorside geo
	Geometry parallelogram2 = context->createGeometry();
	parallelogram2->setPrimitiveCount(1u);
	ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
	parallelogram2->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "bounds"));
	parallelogram2->setIntersectionProgram(context->createProgramFromPTXString(ptx, "intersect"));
	 //anchor = make_float3(-0.21f, -0.06f, -0.201f);
	// v1 = make_float3(0.32f, 0.0f, 0.0f);
	// v2= make_float3(0.0f, 0.32f, 0.0f);
	 anchor = make_float3(-0.20f, -0.05f, -0.201f);
	 v1 = make_float3(0.28f, 0.0f, 0.0f);
	 v2 = make_float3(0.0f, 0.28f, 0.0f);
	 normal = cross(v1, v2);
	normal = normalize(normal);
	 d = dot(normal, anchor);
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	 plane = make_float4(normal, d);
	parallelogram2["plane"]->setFloat(plane);
	parallelogram2["v1"]->setFloat(v1);
	parallelogram2["v2"]->setFloat(v2);
	parallelogram2["anchor"]->setFloat(anchor);


	// checker material
	ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
	Program check_ch2 = context->createProgramFromPTXString(ptx, "plane2_closest_hit");
	//Program check_ah = context->createProgramFromPTXString(ptx, "any_hit_shadow");
	Material floor_mat2 = context->createMaterial();
	floor_mat2->setClosestHitProgram(0, check_ch2);
	//floor_matl->setAnyHitProgram(1, check_ah);

	floor_mat2["Kd"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_mat2["Ka"]->setFloat(0.8f, 0.3f, 0.15f);
	floor_mat2["Ks"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_mat2["Kd2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_mat2["Ka2"]->setFloat(0.9f, 0.85f, 0.05f);
	floor_mat2["Ks2"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_mat2["inv_checker_size"]->setFloat(10.0f, 10.0f, 1.0f);
	floor_mat2["phong_exp1"]->setFloat(0.0f);
	floor_mat2["phong_exp2"]->setFloat(0.0f);
	floor_mat2["Kr1"]->setFloat(0.0f, 0.0f, 0.0f);
	floor_mat2["Kr2"]->setFloat(0.0f, 0.0f, 0.0f);


	std::vector<GeometryInstance> gis;
	gis.push_back(context->createGeometryInstance(parallelogram, &floor_matl, &floor_matl + 1));
	gis.push_back(context->createGeometryInstance(parallelogram2, &floor_mat2, &floor_mat2 + 1));
	GeometryGroup floorgeo = context->createGeometryGroup();
	floorgeo->setChildCount(static_cast<unsigned int>(gis.size()));
	floorgeo->setChild(0, gis[0]);
	floorgeo->setChild(1, gis[1]);
	floorgeo->setAcceleration(context->createAcceleration("Trbvh"));

	Group top_group = context->createGroup();
	top_group->setAcceleration(context->createAcceleration("Trbvh"));
	top_group->setChildCount(2);

	top_group->setChild(0, transform);
	//top_group->setChild(1, transform4);
	top_group->setChild(1, floorgeo);
    context[ "top_object"   ]->set(top_group);
    context[ "top_shadower" ]->set(top_group);
}

  
void setupCamera()
{
   // const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1));
	//if (aabb.extent(2) > max_dim)
	//	max_dim = aabb.extent(2);// max of x, y components
	float max_dim = fmaxf(aabb.extent(0), aabb.extent(1));
	if (aabb.extent(2) > max_dim)
		max_dim = aabb.extent(2);
	float maxextent = length(aabb.extent());
	context["maxextent"]->setFloat(maxextent*2.0f);
    //camera_eye    = aabb.center() + make_float3( 0.0f, 0.0f, max_dim*1.5f ); 
	camera_eye = aabb.center() + make_float3(max_dim*0.7f, max_dim*0.3f, max_dim*2.0f);
    camera_lookat = aabb.center() + make_float3(max_dim*0.2f, 0.0f, -max_dim*1.1f);
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate  = Matrix4x4::identity();
	
}


void setupLights()
{
     float max_d = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y£¬z components

	/*
    BasicLight lights[] = {
        { make_float3( -0.5f,  0.25f, -1.0f ), make_float3( 0.2f, 0.2f, 0.25f ), 0, 0 },
        { make_float3( -0.5f,  0.0f ,  1.0f ), make_float3( 0.1f, 0.1f, 0.10f ), 0, 0 },
        { make_float3(  0.5f,  0.5f ,  0.5f ), make_float3( 0.7f, 0.7f, 0.65f ), 1, 0 }
    };
	*/
	BasicLight lights[] = {
		{ make_float3(-0.5f,  0.25f, -1.0f), make_float3(0.5f, 0.5f, 0.5f), 0, 0 },
		{ make_float3(-0.5f,  0.0f ,  1.0f), make_float3(0.3f, 0.3f, 0.3f), 0, 0 },
		{ make_float3(0.5f,  0.5f ,  0.5f), make_float3(0.9f, 0.9f, 0.9f), 1, 0 },
		{ make_float3(0.5f,  0.5f ,  -0.5f), make_float3(0.9f, 0.9f, 0.9f), 1, 0 }
	};
    lights[0].pos *= max_d * 10.0f; 
    lights[1].pos *= max_d * 10.0f; 
    lights[2].pos *= max_d * 10.0f; 
	lights[3].pos *= max_d * 10.0f;

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov = 50.0f;
    const float aspect_ratio = static_cast<float>(width) /
                               static_cast<float>(height);
    
    float3 camera_u, camera_v, camera_w;
     calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame = Matrix4x4::fromBasis( 
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans     = frame*camera_rotate*camera_rotate*frame_inv; 

    camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    calculateCameraVariables(
            camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
            camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

	 l1 = camera_lookat - camera_eye;
	//std::cout << "l1£º" << l1.x << "," << l1.y << "," <<l1.z << std::endl;
	 dep1 = length(l1);
	//std::cout << dep1 << std::endl;
	l1 = normalize(l1);
	context["l1"]->setFloat(l1);
	context["len"]->setFloat(dep1);
	
	
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 0, 0 );                                               
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();                                                              
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);                                                   
    glLoadIdentity();                                                              
    glOrtho(0, 1, 0, 1, -1, 1 );                                                   

    glMatrixMode(GL_MODELVIEW);                                                    
    glLoadIdentity();                                                              

    glViewport(0, 0, width, height);                                 

    glutShowWindow();                                                              
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();
	
    glutMainLoop();
	
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------
int run = 1;
int batch = fencengshu;
int nRows = width;
int nCols = height;
int n[2] = { nRows, nCols };
int idist = nRows * nCols;
int odist = nRows * nCols;
int inembed[] = { nRows, nCols };
int onembed[] = { nRows, nCols };
int istride = 1;
int ostride = 1;
int rank = 2;


void glutDisplay()
{
	if (run == 1) {
		
	updateCamera();
	
	context->launch(0, width, height);

	cufftExecC2C(forward_plan, complex_prt, complex_prt, CUFFT_FORWARD);
	
	context->launch(1, width, height);
	

	sutil::displayBufferGL(getOutputBuffer());

	{
		static unsigned frame_count = 0;
		sutil::displayFps(frame_count++);
		
	}

	glutSwapBuffers();
    }
	if (debugmode == 1)
	{
		run = 0;
		Buffer xiang = context["output_buffer"]->getBuffer();
		float* xiang_ptr = static_cast<float*>(xiang->map());
		int kk = 0;
		for (int a = height-1; a > -1; a--)
		{
			uchar* tu = out.ptr<uchar>(a);

			for (int b = 0; b < width; b++)
			{
			
				tu[b] = floor(xiang_ptr[kk] * 255.0);
				kk++;
			}
		}
		kk = 0;
		xiang->unmap();
		imwrite("1.png", out);
	}
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        camera_eye = camera_eye + (camera_lookat - camera_eye)*scale;
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_rotate = myrotate( b, a );
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
	/*
    if ( w == (int)width && h == (int)height ) return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    
    glViewport(0, 0, width, height);                                               

    glutPostRedisplay();
	*/
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------



int main( int argc, char** argv )
 {
    std::string out_file;
    //std::string mesh_file = std::string( sutil::samplesDir() ) + "/data/cow.obj";
	std::string mesh_file = std::string("dragon.ply");
    int usage_report_level = 0;
    
    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif
		
		
        createContext();
		
		Buffer Random = context["random"]->getBuffer();
		float* rand_ptr = static_cast<float*>(Random->map());
		for (int a = 0; a < height; a++)
		{
			for (int b = 0; b < width; b++)
			{
				int c = b + a * height;
				rand_ptr[c] = 2 * 3.14159*rand() / float(RAND_MAX);
				//rand_ptr[c] = rand() / float(RAND_MAX);

			}
		}
		Random->unmap();
		
		
		cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
        loadMesh( mesh_file );
        setupCamera();
        setupLights();


        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
		cufftDestroy(forward_plan);
        return 0;
    }
    SUTIL_CATCH( context->get() )
}


void calculateCameraVariables(float3 eye, float3 lookat, float3 up,
	float  fov, float  aspect_ratio,
	float3& U, float3& V, float3& W, bool fov_is_vertical)
{
	float ulen, vlen, wlen;
	W = lookat - eye; // Do not normalize W -- it implies focal length

	wlen = length(W);
	U = normalize(cross(W, up));
	V = normalize(cross(U, W));

	if (fov_is_vertical) {
		vlen = wlen * tanf(0.5f * fov * 3.14159f / 180.0f);
		V *= vlen;
		ulen = vlen * aspect_ratio;
		U *= ulen;
	}
	else {
		ulen = wlen * tanf(0.5f * fov * 3.14159f / 180.0f);
		U *= ulen;
		vlen = ulen / aspect_ratio;
		V *= vlen;
	}
}


//Ìí¼ÓÐý×ª
class Quaternion
{
public:

	Quaternion()
	{
		q[0] = q[1] = q[2] = q[3] = 0.0;
	}

	Quaternion(float w, float x, float y, float z)
	{
		q[0] = w; q[1] = x; q[2] = y; q[3] = z;
	}

	Quaternion(const float3& from, const float3& to);

	Quaternion(const Quaternion& a)
	{
		q[0] = a[0];  q[1] = a[1];  q[2] = a[2];  q[3] = a[3];
	}

	Quaternion(float angle, const float3& axis);

	// getters and setters
	void setW(float _w) { q[0] = _w; }
	void setX(float _x) { q[1] = _x; }
	void setY(float _y) { q[2] = _y; }
	void setZ(float _z) { q[3] = _z; }
	float w() const { return q[0]; }
	float x() const { return q[1]; }
	float y() const { return q[2]; }
	float z() const { return q[3]; }


	Quaternion& operator-=(const Quaternion& r)
	{
		q[0] -= r[0]; q[1] -= r[1]; q[2] -= r[2]; q[3] -= r[3]; return *this;
	}

	Quaternion& operator+=(const Quaternion& r)
	{
		q[0] += r[0]; q[1] += r[1]; q[2] += r[2]; q[3] += r[3]; return *this;
	}

	Quaternion& operator*=(const Quaternion& r);

	Quaternion& operator/=(const float a);

	Quaternion conjugate()
	{
		return Quaternion(q[0], -q[1], -q[2], -q[3]);
	}

	void rotation(float& angle, float3& axis) const;
	void rotation(float& angle, float& x, float& y, float& z) const;
	Matrix4x4 rotationMatrix() const;

	float& operator[](int i) { return q[i]; }
	float operator[](int i)const { return q[i]; }

	// l2 norm
	float norm() const
	{
		return sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	}

	float  normalize();

private:
	float q[4];
};
inline float Quaternion::normalize()
{
	float n = norm();
	float inverse = 1.0f / n;
	q[0] *= inverse;
	q[1] *= inverse;
	q[2] *= inverse;
	q[3] *= inverse;
	return n;
}
inline Matrix4x4 Quaternion::rotationMatrix() const
{
	Matrix4x4 m;

	const float qw = q[0];
	const float qx = q[1];
	const float qy = q[2];
	const float qz = q[3];

	m[0 * 4 + 0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;
	m[0 * 4 + 1] = 2.0f*qx*qy - 2.0f*qz*qw;
	m[0 * 4 + 2] = 2.0f*qx*qz + 2.0f*qy*qw;
	m[0 * 4 + 3] = 0.0f;

	m[1 * 4 + 0] = 2.0f*qx*qy + 2.0f*qz*qw;
	m[1 * 4 + 1] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;
	m[1 * 4 + 2] = 2.0f*qy*qz - 2.0f*qx*qw;
	m[1 * 4 + 3] = 0.0f;

	m[2 * 4 + 0] = 2.0f*qx*qz - 2.0f*qy*qw;
	m[2 * 4 + 1] = 2.0f*qy*qz + 2.0f*qx*qw;
	m[2 * 4 + 2] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
	m[2 * 4 + 3] = 0.0f;

	m[3 * 4 + 0] = 0.0f;
	m[3 * 4 + 1] = 0.0f;
	m[3 * 4 + 2] = 0.0f;
	m[3 * 4 + 3] = 1.0f;

	return m;
}
inline Quaternion::Quaternion(const float3& from, const float3& to)
{
	const float3 c = cross(from, to);
	q[0] = dot(from, to);
	q[1] = c.x;
	q[2] = c.y;
	q[3] = c.z;
}

float2  m_center = make_float2(0.5f);
float   m_radius = 0.45f;

float3 toSphere(const float2& v)
{
	float x = (v.x - m_center.x) / m_radius;
	float y = (1.0f - v.y - m_center.y) / m_radius;

	float z = 0.0f;
	float len2 = x * x + y * y;
	if (len2 > 1.0f) {
		// Project to closest point on edge of sphere.
		float len = sqrtf(len2);
		x /= len;
		y /= len;
	}
	else {
		z = sqrtf(1.0f - len2);
	}
	return make_float3(x, y, z);
}

Matrix4x4 myrotate(const float2& from, const float2& to)
{
	float3 a = toSphere(from);
	float3 b = toSphere(to);

	Quaternion q = Quaternion(a, b);
	q.normalize();

	return q.rotationMatrix();
}