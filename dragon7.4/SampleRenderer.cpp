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

#include "SampleRenderer.h"
#include<cstdlib>
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

#include <fstream>
/*! \namespace osc - Optix Siggraph Course */
namespace osc {


  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };
  
  SampleRenderer::SampleRenderer(const Model* model)
      : model(model)
  {

    initOptix();
      
    std::cout << "#osc: creating optix context ..." << std::endl;
    createContext();
      
    std::cout << "#osc: setting up module ..." << std::endl;
    createModule();

    std::cout << "#osc: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    createRaygenPrograms2();
    std::cout << "#osc: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "#osc: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    launchParams.traversable = buildAccel();

    std::cout << "#osc: setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << "#osc: building SBT ..." << std::endl;
    buildSBT();
    //buildSBT2();
    //std::cout << "#osc: building SBT2 ..." << std::endl;
    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << "#osc: context, module, pipeline, etc, all set up ..." << std::endl;

    launchParams.holoparams.lambda = 639e-6;
    launchParams.holoparams.f = 300.0f;
    launchParams.holoparams.juli = 1.0f;
    launchParams.holoparams.layernum = 300;
    launchParams.holoparams.pitch= 0.008f;
    launchParams.frame_index = 0;
    


    std::cout << GDT_TERMINAL_GREEN;
    std::cout << "#osc: Optix 7 Sample fully set up" << std::endl;
    std::cout << GDT_TERMINAL_DEFAULT;
  }

  void SampleRenderer::createTextures()
  {
      int numTextures = (int)model->textures.size();

      textureArrays.resize(numTextures);
      textureObjects.resize(numTextures);

      for (int textureID = 0; textureID < numTextures; textureID++) {
          auto texture = model->textures[textureID];

          cudaResourceDesc res_desc = {};

          cudaChannelFormatDesc channel_desc;
          int32_t width = texture->resolution.x;
          int32_t height = texture->resolution.y;
          int32_t numComponents = 4;
          int32_t pitch = width * numComponents * sizeof(uint8_t);
          channel_desc = cudaCreateChannelDesc<uchar4>();

          cudaArray_t& pixelArray = textureArrays[textureID];
          CUDA_CHECK(MallocArray(&pixelArray,
              &channel_desc,
              width, height));

          CUDA_CHECK(Memcpy2DToArray(pixelArray,
              /* offset */0, 0,
              texture->pixel,
              pitch, pitch, height,
              cudaMemcpyHostToDevice));

          res_desc.resType = cudaResourceTypeArray;
          res_desc.res.array.array = pixelArray;

          cudaTextureDesc tex_desc = {};
          tex_desc.addressMode[0] = cudaAddressModeWrap;
          tex_desc.addressMode[1] = cudaAddressModeWrap;
          tex_desc.filterMode = cudaFilterModeLinear;
          tex_desc.readMode = cudaReadModeNormalizedFloat;
          tex_desc.normalizedCoords = 1;
          tex_desc.maxAnisotropy = 1;
          tex_desc.maxMipmapLevelClamp = 99;
          tex_desc.minMipmapLevelClamp = 0;
          tex_desc.mipmapFilterMode = cudaFilterModePoint;
          tex_desc.borderColor[0] = 1.0f;
          tex_desc.sRGB = 0;

          // Create texture object
          cudaTextureObject_t cuda_tex = 0;
          CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
          textureObjects[textureID] = cuda_tex;
      }
  }

  OptixTraversableHandle SampleRenderer::buildAccel()
  {
      const int numMeshes = (int)model->meshes.size();
      vertexBuffer.resize(numMeshes);
      normalBuffer.resize(numMeshes);
      texcoordBuffer.resize(numMeshes);
      indexBuffer.resize(numMeshes);

      OptixTraversableHandle asHandle{ 0 };

      // ==================================================================
      // triangle inputs
      // ==================================================================
      std::vector<OptixBuildInput> triangleInput(model->meshes.size());
      std::vector<CUdeviceptr> d_vertices(model->meshes.size());
      std::vector<CUdeviceptr> d_indices(model->meshes.size());
      std::vector<uint32_t> triangleInputFlags(model->meshes.size());

      for (int meshID = 0; meshID < model->meshes.size(); meshID++) {
          // upload the model to the device: the builder
          TriangleMesh& mesh = *model->meshes[meshID];
          vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
          indexBuffer[meshID].alloc_and_upload(mesh.index);
          if (!mesh.normal.empty())
              normalBuffer[meshID].alloc_and_upload(mesh.normal);
          if (!mesh.texcoord.empty())
              texcoordBuffer[meshID].alloc_and_upload(mesh.texcoord);

          triangleInput[meshID] = {};
          triangleInput[meshID].type
              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

          // create local variables, because we need a *pointer* to the
          // device pointers
          d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
          d_indices[meshID] = indexBuffer[meshID].d_pointer();

          triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
          triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
          triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
          triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

          triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
          triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
          triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
          triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

          triangleInputFlags[meshID] = 0;

          // in this example we have one SBT entry, and no per-primitive
          // materials:
          triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
          triangleInput[meshID].triangleArray.numSbtRecords = 1;
          triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
          triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
          triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
      }
      // ==================================================================
      // BLAS setup
      // ==================================================================

      OptixAccelBuildOptions accelOptions = {};
      accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
          | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
          ;
      accelOptions.motionOptions.numKeys = 1;
      accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage
      (optixContext,
          &accelOptions,
          triangleInput.data(),
          (int)model->meshes.size(),  // num_build_inputs
          &blasBufferSizes
      ));

      // ==================================================================
      // prepare compaction
      // ==================================================================

      CUDABuffer compactedSizeBuffer;
      compactedSizeBuffer.alloc(sizeof(uint64_t));

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = compactedSizeBuffer.d_pointer();

      // ==================================================================
      // execute build (main stage)
      // ==================================================================

      CUDABuffer tempBuffer;
      tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

      CUDABuffer outputBuffer;
      outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

      OPTIX_CHECK(optixAccelBuild(optixContext,
          /* stream */0,
          &accelOptions,
          triangleInput.data(),
          (int)model->meshes.size(),
          tempBuffer.d_pointer(),
          tempBuffer.sizeInBytes,

          outputBuffer.d_pointer(),
          outputBuffer.sizeInBytes,

          &asHandle,

          &emitDesc, 1
      ));
      CUDA_SYNC_CHECK();

      // ==================================================================
      // perform compaction
      // ==================================================================
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize, 1);

      asBuffer.alloc(compactedSize);
      OPTIX_CHECK(optixAccelCompact(optixContext,
          /*stream:*/0,
          asHandle,
          asBuffer.d_pointer(),
          asBuffer.sizeInBytes,
          &asHandle));
      CUDA_SYNC_CHECK();

      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
      tempBuffer.free();
      compactedSizeBuffer.free();

      return asHandle;
  }

  /*! helper function that initializes optix and checks for errors */
  void SampleRenderer::initOptix()
  {
    std::cout << "#osc: initializing optix..." << std::endl;
      
    // -------------------------------------------------------
    // check for available optix7 capable devices
    // -------------------------------------------------------
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw std::runtime_error("#osc: no CUDA capable devices found!");
    std::cout << "#osc: found " << numDevices << " CUDA devices" << std::endl;

    // -------------------------------------------------------
    // initialize optix
    // -------------------------------------------------------
    OPTIX_CHECK( optixInit() );
    std::cout << GDT_TERMINAL_GREEN
              << "#osc: successfully initialized optix... yay!"
              << GDT_TERMINAL_DEFAULT << std::endl;
  }

  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
  void SampleRenderer::createContext()
  {
    // for this sample, do everything on one device
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));
    CUDA_CHECK(StreamCreate(&stream2));
      
    cudaGetDeviceProperties(&deviceProps, deviceID);
    std::cout << "#osc: running on device: " << deviceProps.name << std::endl;
      
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if( cuRes != CUDA_SUCCESS ) 
      fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
      
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,nullptr,4));
  }


  static void getInputDataFromFile(std::string& ptx)
  {
      const std::string sourceFilePath = "D:/zcl/vs2019/optixtu/optixtu3/optixtu3/x64/Debug/devicePrograms.cu.ptx";   
      std::ifstream file(sourceFilePath.c_str(), std::ios::binary);
      std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
      ptx.assign(buffer.begin(), buffer.end());
  }


  /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
  void SampleRenderer::createModule()
  {
    moduleCompileOptions.maxRegisterCount  = 50;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
    pipelineLinkOptions.maxTraceDepth          = 2;
    std::string* ptx;
    ptx = new std::string();


    getInputDataFromFile(*ptx);
    //const std::string ptxCode = embedded_ptx_code;
    const std::string ptxCode = *ptx;


    //const std::string sourceFilePath = "D:/zcl/vs2019/optixtu/optixtu/x64/Debug/optixPathTracer.cu.ptx";
   // std::ifstream file(sourceFilePath.c_str(), std::ios::binary);
   // std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
   // ptx.assign(buffer.begin(), buffer.end());
    
 
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                         &moduleCompileOptions,
                                         &pipelineCompileOptions,
                                         ptxCode.c_str(),
                                         ptxCode.size(),
                                         log,&sizeof_log,
                                         &module
                                         ));
    if (sizeof_log > 1) PRINT(log);
  }
    


  /*! does all setup for the raygen program(s) we are going to use */
  void SampleRenderer::createRaygenPrograms()
  {
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = module;           
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &raygenPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
  void SampleRenderer::createRaygenPrograms2()
  {
      // we do a single ray gen program in this example:
      raygenPGs2.resize(1);

      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc pgDesc = {};
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module = module;
      pgDesc.raygen.entryFunctionName = "__raygen__holo";

      // OptixProgramGroup raypg;
      char log[2048];
      size_t sizeof_log = sizeof(log);
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
          &pgDesc,
          1,
          &pgOptions,
          log, &sizeof_log,
          &raygenPGs2[0]
      ));
      if (sizeof_log > 1) PRINT(log);
  }
  /*! does all setup for the miss program(s) we are going to use */
  void SampleRenderer::createMissPrograms()
  {
    // we do a single ray gen program in this example:
    missPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = module;           
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &missPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    
  /*! does all setup for the hitgroup program(s) we are going to use */
  void SampleRenderer::createHitgroupPrograms()
  {
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);
      
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc    = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = module;           
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH            = module;           
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,&sizeof_log,
                                        &hitgroupPGs[0]
                                        ));
    if (sizeof_log > 1) PRINT(log);
  }
    

  /*! assembles the full pipeline of all programs */
  void SampleRenderer::createPipeline()
  {
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
      programGroups.push_back(pg);
    for (auto pg : missPGs)
      programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
      programGroups.push_back(pg);

    for (auto pg : raygenPGs2)
        programGroups.push_back(pg);
      
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log,&sizeof_log,
                                    &pipeline
                                    ));
    if (sizeof_log > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                (/* [in] The pipeline to configure the stack size for */
                 pipeline, 
                 /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */                 
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 2*1024,
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 1));
    if (sizeof_log > 1) PRINT(log);
  }


  /*! constructs the shader binding table */
  void SampleRenderer::buildSBT()
  {
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i=0;i<raygenPGs.size();i++) {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    std::vector<RaygenRecord> raygenRecords2;
    for (int i = 0; i < raygenPGs2.size(); i++) {
        RaygenRecord rec2;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs2[i], &rec2));
        rec2.data = nullptr; /* for now ... */
        raygenRecords2.push_back(rec2);
    }
    raygenRecordsBuffer2.alloc_and_upload(raygenRecords2);
    sbt2.raygenRecord = raygenRecordsBuffer2.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i=0;i<missPGs.size();i++) {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase          = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = (int)missRecords.size();
 

    sbt2.missRecordBase = missRecordsBuffer.d_pointer();
    sbt2.missRecordStrideInBytes = sizeof(MissRecord);
    sbt2.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::cout << "#osc: building SBT ..."<< numObjects << std::endl;
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++) {
        auto mesh = model->meshes[meshID];
        //std::cout << "#osc: building SBT ..." << std::endl;

        HitgroupRecord rec;
        // all meshes use the same code, so all same hit group
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
        rec.data.color = mesh->diffuse;
        if (mesh->diffuseTextureID >= 0) {
            rec.data.hasTexture = true;
            rec.data.texture = textureObjects[mesh->diffuseTextureID];
        }
        else {
            rec.data.hasTexture = false;
        }
        rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
        rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
        rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
        rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
        hitgroupRecords.push_back(rec);
    }
    
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();

    sbt2.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt2.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt2.hitgroupRecordCount = (int)hitgroupRecords.size();
  }

  void SampleRenderer::buildSBT2()
  {
      // ------------------------------------------------------------------
      // build raygen records
      // ------------------------------------------------------------------
      std::vector<RaygenRecord> raygenRecords;
      for (int i = 0; i < raygenPGs2.size(); i++) {
          RaygenRecord rec;
          OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs2[i], &rec));
          rec.data = nullptr; /* for now ... */
          raygenRecords.push_back(rec);
      }
      raygenRecordsBuffer2.alloc_and_upload(raygenRecords);
      sbt2.raygenRecord = raygenRecordsBuffer2.d_pointer();

      // ------------------------------------------------------------------
      // build miss records
      // ------------------------------------------------------------------
      std::vector<MissRecord> missRecords;
      for (int i = 0; i < missPGs.size(); i++) {
          MissRecord rec;
          OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
          rec.data = nullptr; /* for now ... */
          missRecords.push_back(rec);
      }
      missRecordsBuffer.alloc_and_upload(missRecords);
      sbt2.missRecordBase = missRecordsBuffer.d_pointer();
      sbt2.missRecordStrideInBytes = sizeof(MissRecord);
      sbt2.missRecordCount = (int)missRecords.size();

      // ------------------------------------------------------------------
      // build hitgroup records
      // ------------------------------------------------------------------
      int numObjects = (int)model->meshes.size();
      std::vector<HitgroupRecord> hitgroupRecords;
      for (int meshID = 0; meshID < numObjects; meshID++) {
          auto mesh = model->meshes[meshID];
          //std::cout << "#osc: building SBT ..." << std::endl;

          HitgroupRecord rec;
          // all meshes use the same code, so all same hit group
          OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
          rec.data.color = mesh->diffuse;
          if (mesh->diffuseTextureID >= 0) {
              rec.data.hasTexture = true;
              rec.data.texture = textureObjects[mesh->diffuseTextureID];
          }
          else {
              rec.data.hasTexture = false;
          }
          rec.data.index = (vec3i*)indexBuffer[meshID].d_pointer();
          rec.data.vertex = (vec3f*)vertexBuffer[meshID].d_pointer();
          rec.data.normal = (vec3f*)normalBuffer[meshID].d_pointer();
          rec.data.texcoord = (vec2f*)texcoordBuffer[meshID].d_pointer();
          hitgroupRecords.push_back(rec);
      }

      hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
      sbt2.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
      sbt2.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
      sbt2.hitgroupRecordCount = (int)hitgroupRecords.size();
  }

  /*! render one frame */
  void SampleRenderer::render()
  {
      // sanity check: make sure we launch only after first resize is
     // already done:
      if (launchParams.frame.size.x == 0) return;

      launchParamsBuffer.upload(&launchParams, 1);

      float time_GPU;
      cudaEvent_t start_GPU, stop_GPU;
      //����Event
      cudaEventCreate(&start_GPU);
      cudaEventCreate(&stop_GPU);
      //��¼��ǰʱ��
      cudaEventRecord(start_GPU, 0);
      for (int i = 0; i < 20; i++)
      {
          OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
              pipeline, stream,
              /*! parameters and SBT */
              launchParamsBuffer.d_pointer(),
              launchParamsBuffer.sizeInBytes,
              &sbt,
              /*! dimensions of the launch: */
              launchParams.frame.size.x,
              launchParams.frame.size.y,
              1
          ));


          //CUDA_SYNC_CHECK();
          cufftExecC2C(forward_plan, launchParams.complex_prt, launchParams.complex_prt, CUFFT_FORWARD);
          //CUDA_SYNC_CHECK();
          OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
              pipeline, stream,
              /*! parameters and SBT */
              launchParamsBuffer.d_pointer(),
              launchParamsBuffer.sizeInBytes,
              &sbt2,
              /*! dimensions of the launch: */
              launchParams.frame.size.x,
              launchParams.frame.size.y,
              1
          ));
          launchParams.frame_index++;
      }
      cudaEventRecord(stop_GPU, 0);
      cudaEventSynchronize(start_GPU);    //�ȴ��¼���ɡ�
      cudaEventSynchronize(stop_GPU);    //�ȴ��¼���ɡ���¼֮ǰ������
      cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU);    //����ʱ���
      printf("\nThe time for GPU:\t%f(ms)\n", time_GPU / 20.0);
      cudaEventDestroy(start_GPU);    //����Event
      cudaEventDestroy(stop_GPU);
    
      // sync - make sure the frame is rendered before we download and
      // display (obviously, for a high-performance application you
      // want to use streams and double-buffering, but for this simple
      // example, this will have to do)
      CUDA_SYNC_CHECK();
  }
  void SampleRenderer::setCamera(const Camera& camera)
  {
      lastSetCamera = camera;
      launchParams.camera.lookat = camera.at;
      launchParams.camera.extent = camera.modelextent;

      //std::cout << "lllllllllllllllllllll:" << camera.modelextent << std::endl;
      //std::cout << "lllllllllllllllllllll:" << launchParams.camera.extent << std::endl;

      launchParams.camera.position = camera.from;
      launchParams.camera.direction = normalize(camera.at - camera.from);
      const float cosFovy = 0.66f;
      const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
      launchParams.camera.horizontal
          = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
              camera.up));
      launchParams.camera.vertical
          = cosFovy * normalize(cross(launchParams.camera.horizontal,
              launchParams.camera.direction));
  }
  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {

    // if window minimized
    if (newSize.x == 0 | newSize.y == 0) return;
    
    // resize our cuda frame buffer
    colorBuffer.resize(newSize.x* newSize.y*sizeof(uint32_t));
    complexbuffer.resize(newSize.x * newSize.y * launchParams.holoparams.layernum *sizeof(cufftComplex));


   
    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.frame.size = newSize;
    launchParams.frame.colorBuffer = (uint32_t*)colorBuffer.d_pointer();
    launchParams.complex_prt = (cufftComplex*)complexbuffer.d_pointer();
   
    
    int batch = launchParams.holoparams.layernum;
    int nRows = newSize.y;
    int nCols = newSize.x;
    int n[2] = { nRows, nCols };
    int idist = nRows * nCols;
    int odist = nRows * nCols;
    int inembed[] = { newSize.y,newSize.x };
    int onembed[] = { nRows, nCols };
    int istride = 1;
    int ostride = 1;
    int rank = 2;

    cufftPlanMany(&forward_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

  }

  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    colorBuffer.download(h_pixels,
        launchParams.frame.size.x * launchParams.frame.size.y);
  }
  
} // ::osc
