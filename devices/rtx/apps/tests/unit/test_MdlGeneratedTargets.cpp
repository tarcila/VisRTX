// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <anari/ext/visrtx/makeVisRTXDevice.h>
#include <dlfcn.h>
#include <link.h>
#include <mi/neuraylib/factory.h>
#include <mi/neuraylib/iversion.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "libmdl/ptx.h"
#include "mdlTargets/probe.h"
#include "renderer/Renderer.h"
#include "world/World.h"

using namespace visrtx;
using namespace mdl_target_test;
namespace fs = std::filesystem;
static int errors = 0;
static void status(const void *,
    ANARIDevice,
    ANARIObject,
    ANARIDataType,
    ANARIStatusSeverity severity,
    ANARIStatusCode,
    const char *message)
{
  if (severity <= ANARI_SEVERITY_ERROR) {
    ++errors;
    std::fprintf(stderr, "ANARI: %s\n", message);
  }
}
static void require(bool ok, const char *message)
{
  if (!ok)
    throw std::runtime_error(message);
}
static void checked(cudaError_t e)
{
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
static void ox(OptixResult r)
{
  if (r != OPTIX_SUCCESS)
    throw std::runtime_error(optixGetErrorName(r));
}
static std::string read(const fs::path &p)
{
  std::ifstream f(p, std::ios::binary);
  require(bool(f), "missing qualification input");
  return {std::istreambuf_iterator<char>(f), {}};
}
static void write(const fs::path &p, const std::string &s)
{
  std::ofstream f(p, std::ios::binary);
  f << s;
  require(bool(f), "cannot record qualification artifact");
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record
{
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
struct Probe
{
  explicit Probe(DeviceGlobalState &s) : state(s) {}
  ~Probe();
  void initialize(
      const std::vector<char> &ptx, const mi::neuraylib::ITarget_code &target);
  std::vector<Result> run(unsigned mode);
  DeviceGlobalState &state;
  OptixModule module{};
  OptixPipeline pipeline{};
  std::array<OptixProgramGroup, 5> groups{};
  OptixShaderBindingTable sbt{};
  DeviceBuffer rg, dc, params, frame, argument, output;
};
Probe::~Probe()
{
  cudaStreamSynchronize(state.stream);
  if (pipeline)
    optixPipelineDestroy(pipeline);
  for (auto g : groups)
    if (g)
      optixProgramGroupDestroy(g);
  if (module)
    optixModuleDestroy(module);
}
void Probe::initialize(
    const std::vector<char> &ptx, const mi::neuraylib::ITarget_code &target)
{
  auto options = makeVisRTXOptixPipelineCompileOptions();
  options.pipelineLaunchParamsVariableName = "params";
  OptixModuleCompileOptions mo{};
  mo.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  mo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  char log[16384];
  size_t size = sizeof(log);
  auto r = optixModuleCreate(state.optixContext,
      &mo,
      &options,
      ptx.data(),
      ptx.size(),
      log,
      &size,
      &module);
  if (r != OPTIX_SUCCESS)
    std::fprintf(stderr, "MDL module: %s\n", log);
  ox(r);
  std::array<OptixProgramGroupDesc, 5> descriptions{};
  descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  descriptions[0].raygen.module = module;
  descriptions[0].raygen.entryFunctionName = "__raygen__mdlSurfaceProbe";
  const char *names[4]{"__direct_callable__init",
      "__direct_callable__nextRay",
      "__direct_callable__evalBsdf",
      "__direct_callable__evaluatePdf"};
  for (int i = 1; i < 5; ++i) {
    descriptions[i].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    descriptions[i].callables.moduleDC = module;
    descriptions[i].callables.entryFunctionNameDC = names[i - 1];
  }
  OptixProgramGroupOptions go{};
  size = sizeof(log);
  r = optixProgramGroupCreate(state.optixContext,
      descriptions.data(),
      5,
      &go,
      log,
      &size,
      groups.data());
  if (r != OPTIX_SUCCESS)
    std::fprintf(stderr, "MDL groups: %s\n", log);
  ox(r);
  OptixPipelineLinkOptions lo{};
  lo.maxTraceDepth = 1;
  size = sizeof(log);
  r = optixPipelineCreate(state.optixContext,
      &options,
      &lo,
      groups.data(),
      5,
      log,
      &size,
      &pipeline);
  if (r != OPTIX_SUCCESS)
    std::fprintf(stderr, "MDL pipeline: %s\n", log);
  ox(r);
  OptixStackSizes stack{};
  for (auto g : groups)
    ox(optixUtilAccumulateStackSizes(g, &stack, pipeline));
  unsigned a, b, c;
  ox(optixUtilComputeStackSizes(&stack, 1, 0, 2, &a, &b, &c));
  ox(optixPipelineSetStackSize(pipeline, a, b, c, 2));
  Record record{};
  ox(optixSbtRecordPackHeader(groups[0], &record));
  rg.upload(&record);
  sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(rg.ptr());
  std::vector<Record> records(4);
  for (int i = 0; i < 4; ++i)
    ox(optixSbtRecordPackHeader(groups[i + 1], &records[i]));
  dc.upload(records);
  sbt.callablesRecordBase = reinterpret_cast<CUdeviceptr>(dc.ptr());
  sbt.callablesRecordStrideInBytes = sizeof(Record);
  sbt.callablesRecordCount = 4;
  if (target.get_argument_block_count()) {
    mi::base::Handle<const mi::neuraylib::ITarget_argument_block> block(
        target.get_argument_block(0));
    if (block && block->get_size())
      argument.upload(block->get_data(), block->get_size());
  }
  FrameGPUData fd{};
  frame.upload(&fd);
  output.reserve(sizeof(Result) * SAMPLES);
}
std::vector<Result> Probe::run(unsigned mode)
{
  Params p{frame.ptrAs<FrameGPUData>(),
      argument.ptrAs<char>(),
      output.ptrAs<Result>(),
      mode};
  params.upload(&p);
  checked(cudaStreamSynchronize(nullptr));
  ox(optixLaunch(pipeline,
      state.stream,
      reinterpret_cast<CUdeviceptr>(params.ptr()),
      sizeof(p),
      &sbt,
      SAMPLES,
      1,
      1));
  checked(cudaStreamSynchronize(state.stream));
  std::vector<Result> result(SAMPLES);
  output.download(result.data(), result.size());
  return result;
}
static bool close(float a, float b)
{
  return std::isfinite(a) && std::isfinite(b)
      && std::abs(double(a) - b) <= 2e-6 + 5e-5 * std::abs(double(b));
}
static void check(const std::string &name,
    unsigned mode,
    const std::vector<Result> &results,
    std::ofstream &report)
{
  using namespace mi::neuraylib;
  size_t absorbed = 0, stale = 0, singular = 0, continuous = 0;
  for (const auto &r : results) {
    require(r.unchangedRandom, "MDL wrapper changed supplied xi sequence");
    if (r.event == BSDF_EVENT_ABSORB) {
      ++absorbed;
      bool positive = false, nonnegativeFinite = true;
      for (int c = 0; c < 3; ++c) {
        positive |= r.weight[c] > 0.f;
        nonnegativeFinite &= std::isfinite(r.weight[c]) && r.weight[c] >= 0.f;
      }
      stale += positive && nonnegativeFinite;
      require(isDead(r.consumed) && r.consumed.direction == vec3(0)
              && r.consumed.contributionWeight == vec3(0) && r.consumed.pdf == 0
              && r.consumed.flags == 0,
          "generated absorption not consumed as dead");
      continue;
    }
    require(r.consumed.mdlEventMask == r.event
            && r.consumed.component == NextRayComponent::MDL,
        "generated event bits lost");
    require(r.consumed.direction == r.direction
            && r.consumed.contributionWeight == r.weight,
        "wrapper changed live generated sample");
    for (int c = 0; c < 3; ++c)
      require(close(r.wrapperEvaluated[c], r.evaluated[c]),
          "wrapper/target evaluation mismatch");
    require(close(r.wrapperPdf, r.queriedPdf), "wrapper/target pdf mismatch");
    if (r.event & BSDF_EVENT_SPECULAR) {
      ++singular;
      require(r.consumed.event == NextRayEvent::Singular
              && std::isinf(r.consumed.pdf),
          "live delta lost");
    } else {
      ++continuous;
      require(r.consumed.event == NextRayEvent::Continuous,
          "continuous event lost");
      require(close(r.pdf, r.queriedPdf), "sample/pdf target mismatch");
      for (int c = 0; c < 3; ++c)
        require(close(r.weight[c] * r.pdf, r.evaluated[c]),
            "sample/evaluate target mismatch");
    }
    if (mode < 2
        && (name == "diffuse" || name == "mirror" || name == "mixed")) {
      const vec3 tint(.6f, .4f, .2f);
      for (int c = 0; c < 3; ++c)
        require(close(r.weight[c], tint[c]), "analytic tint compensation");
      if (!(r.event & BSDF_EVENT_SPECULAR)) {
        const float p = (name == "mixed" ? .5f : 1.f) * r.direction.z
            / float(std::acos(-1.));
        require(close(r.pdf, p), "analytic Lambertian pdf");
      }
    }
  }
  if (name == "black")
    for (const auto &r : results)
      require(r.weight == vec3(0) && r.consumed.contributionWeight == vec3(0),
          "black target has nonzero contribution");
  if (mode < 2 && name == "diffuse")
    require(continuous == SAMPLES, "diffuse target event mass");
  if (mode < 2 && (name == "mirror" || name == "transmit"))
    require(singular == SAMPLES, "specular target event mass");
  if (mode < 2 && name == "mixed")
    require(std::abs(double(singular) / SAMPLES - .5)
            <= 6 * std::sqrt(.25 / SAMPLES) + 1e-4,
        "mixed singular mass");
  if (mode == 2 && name == "mirror")
    require(absorbed == SAMPLES && stale == SAMPLES,
        "real stale-ABSORB fixture not reproduced");
  report << name << "," << mode << "," << absorbed << "," << stale << ","
         << singular << "," << continuous << "\n";
  report.flush();
  std::printf("%s state %u: absorb=%zu stale=%zu singular=%zu continuous=%zu\n",
      name.c_str(),
      mode,
      absorbed,
      stale,
      singular,
      continuous);
}
static int loaded(dl_phdr_info *info, size_t, void *data)
{
  if (info->dlpi_name && std::strstr(info->dlpi_name, "libmdl_sdk.so"))
    static_cast<std::vector<std::string> *>(data)->push_back(info->dlpi_name);
  return 0;
}
static int loadedDevice(dl_phdr_info *info, size_t, void *data)
{
  if (info->dlpi_name
      && std::strstr(info->dlpi_name, "libanari_library_visrtx"))
    static_cast<std::vector<std::string> *>(data)->push_back(info->dlpi_name);
  return 0;
}
int main()
try {
  const fs::path out = MDL_TARGET_OUTPUT;
  fs::create_directories(out);
  auto d = makeVisRTXDevice(status);
  anariCommitParameters(d, d);
  auto world = anariNewWorld(d);
  anariCommitParameters(d, world);
  struct Release
  {
    ANARIDevice d;
    ANARIWorld w;
    ~Release()
    {
      anariRelease(d, w);
      anariRelease(d, d);
    }
  } release{d, world};
  box3 bounds;
  require(anariGetProperty(d,
              world,
              "bounds",
              ANARI_FLOAT32_BOX3,
              &bounds,
              sizeof(bounds),
              ANARI_WAIT)
          != 0,
      "device initialization");
  auto &state = *static_cast<DeviceGlobalState *>(
      reinterpret_cast<World *>(world)->helium::BaseObject::deviceState());
  require(bool(state.mdl), "MDL backend unavailable");
  ox(optixInit());
  std::vector<std::string> images;
  dl_iterate_phdr(loaded, &images);
  require(images.size() == 1, "ambiguous loaded SDK image");
  std::unique_ptr<void, decltype(&dlclose)> image(
      dlopen(images[0].c_str(), RTLD_LAZY | RTLD_NOLOAD), &dlclose);
  require(bool(image), "SDK image not loaded");
  auto version = mi::base::Handle<mi::neuraylib::IVersion>(
      mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(
          dlsym(image.get(), "mi_factory")));
  require(bool(version), "runtime SDK version unavailable");
  std::string identity =
      "sdk=" + images[0] + "\nversion=" + version->get_string() + "\n";
  std::vector<std::string> devices;
  dl_iterate_phdr(loadedDevice, &devices);
  require(devices.size() == 1, "ambiguous loaded VisRTX image");
  identity += "device_library=" + devices[0] + "\n";
  identity += "device=" + std::string(state.deviceProps.name) + "\n";
  int driver = 0, runtime = 0;
  checked(cudaDriverGetVersion(&driver));
  checked(cudaRuntimeGetVersion(&runtime));
  identity += "cuda_driver=" + std::to_string(driver)
      + "\ncuda_runtime=" + std::to_string(runtime)
      + "\noptix=" + std::to_string(OPTIX_VERSION) + "\n";
  write(out / "identity.txt", identity);
  std::printf("%s", identity.c_str());
  version.reset();
  image.reset();
  const auto source = read(MDL_TARGET_FIXTURES),
             texture = read(MDL_TEXTURE_PTX), wrapper = read(MDL_SHADER_PTX),
             probe = read(MDL_TARGET_PROBE_PTX);
  write(out / "fixtures.mdl", source);
  write(out / "wrapper.ptx", wrapper);
  write(out / "resources.ptx", texture);
  write(out / "probe.ptx", probe);
  std::ofstream report(out / "results.csv");
  report << "material,state,absorbed,stale,singular,continuous\n";
  auto &core = state.mdl->core;
  mi::base::Handle<mi::neuraylib::ITransaction> transaction(
      core.createTransaction());
  mi::base::Handle<const mi::neuraylib::IModule> module(
      core.loadModuleFromString(
          "::visrtx_surface_qualification", source, transaction.get()));
  require(bool(module), "fixture module compilation failed");
  for (const char *name :
      {"black", "diffuse", "mirror", "transmit", "mixed", "glossy"}) {
    mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
        core.getFunctionDefinition(module.get(), name, transaction.get()));
    require(bool(definition), "fixture definition missing");
    mi::base::Handle<mi::neuraylib::ICompiled_material> compiled(
        core.getCompiledMaterial(definition.get()));
    require(bool(compiled), "fixture material compilation failed");
    mi::base::Handle<const mi::neuraylib::ITarget_code> target(
        core.getPtxTargetCode(compiled.get(), transaction.get()));
    require(bool(target), "fixture target generation failed");
    write(out / (std::string(name) + ".ptx"),
        std::string(target->get_code(), target->get_code_size()));
    const auto ptx = libmdl::stitchPTXs(
        std::vector{nonstd::span{texture.data(), texture.size()},
            nonstd::span{target->get_code(), target->get_code_size()},
            nonstd::span{wrapper.data(), wrapper.size()},
            nonstd::span{probe.data(), probe.size()}});
    Probe executable(state);
    executable.initialize(ptx, *target);
    for (unsigned mode = 0; mode < 3; ++mode) {
      const auto results = executable.run(mode);
      std::ofstream samples(out
          / (std::string(name) + "-" + std::to_string(mode) + "-samples.csv"));
      samples
          << "event,pdf,k2x,k2y,k2z,wx,wy,wz,consumed_event,consumed_weight_x,query_pdf\n";
      for (size_t i = 0; i < 8; ++i) {
        const auto &r = results[i];
        samples << r.event << "," << r.pdf << "," << r.direction.x << ","
                << r.direction.y << "," << r.direction.z << "," << r.weight.x
                << "," << r.weight.y << "," << r.weight.z << ","
                << uint32_t(r.consumed.event) << ","
                << r.consumed.contributionWeight.x << "," << r.queriedPdf
                << "\n";
      }
      check(name, mode, results, report);
    }
  }
  module.reset();
  transaction->commit();
  require(errors == 0, "unexpected backend errors");
  std::puts(
      "Generated MDL surface targets: bounded leaf/event qualification passed");
  return 0;
} catch (const std::exception &e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what());
  return 1;
}
