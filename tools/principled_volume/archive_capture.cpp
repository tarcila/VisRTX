// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Linux measurement interposer for Vela's public ANARI calls. Performance mode
// leaves frame format unchanged. Optional HDR mode requests float output, saves
// raw linear samples, then supplies an sRGB8 copy to Vela's display pipeline.
#include <anari/anari.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
namespace {
struct Capture
{
  unsigned index = 0;
  std::unordered_map<ANARIFrame, unsigned> frameIndex;
  std::unordered_map<ANARIObject, std::array<float, 3>> positions;
  std::unordered_map<ANARIFrame, ANARICamera> cameras;
  void *display = nullptr;
  size_t capacity = 0;
  ~Capture()
  {
    if (display)
      cudaFree(display);
  }
};
Capture capture;
void checked(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::fprintf(
        stderr, "capture CUDA failure: %s\n", cudaGetErrorString(error));
    std::abort();
  }
}
} // namespace
extern "C" void anariSetParameter(ANARIDevice d,
    ANARIObject o,
    const char *name,
    ANARIDataType type,
    const void *value)
{
  static auto fn = reinterpret_cast<decltype(&anariSetParameter)>(
      dlsym(RTLD_NEXT, "anariSetParameter"));
  if (std::strcmp(name, "position") == 0 && type == ANARI_FLOAT32_VEC3) {
    const float *p = static_cast<const float *>(value);
    capture.positions[o] = {p[0], p[1], p[2]};
  }
  if (std::strcmp(name, "camera") == 0 && type == ANARI_CAMERA)
    capture.cameras[reinterpret_cast<ANARIFrame>(o)] =
        *static_cast<const ANARICamera *>(value);
  // Lua integer renderer parameters are signed; randomSeed's public contract
  // is uint32. Preserve the requested bits when crossing that binding seam.
  if (std::strcmp(name, "randomSeed") == 0 && type == ANARI_INT32) {
    const auto seed = uint32_t(*static_cast<const int32_t *>(value));
    fn(d, o, name, ANARI_UINT32, &seed);
    std::fprintf(stderr, "SEED,%u\n", seed);
    return;
  }
  if (std::getenv("VISRTX_CAPTURE_HDR")
      && std::strcmp(name, "channel.color") == 0 && type == ANARI_DATA_TYPE) {
    const ANARIDataType format = ANARI_FLOAT32_VEC4;
    fn(d, o, name, type, &format);
  } else
    fn(d, o, name, type, value);
}
extern "C" void anariRenderFrame(ANARIDevice d, ANARIFrame f)
{
  static auto fn = reinterpret_cast<decltype(&anariRenderFrame)>(
      dlsym(RTLD_NEXT, "anariRenderFrame"));
  const auto start = std::chrono::steady_clock::now();
  if (const char *amplitude = std::getenv("VISRTX_CAPTURE_CAMERA")) {
    const auto camera = capture.cameras.at(f);
    auto p = capture.positions.at(reinterpret_cast<ANARIObject>(camera));
    p[0] += float(std::atof(amplitude)) * std::sin(.2f * float(capture.index));
    static auto set = reinterpret_cast<decltype(&anariSetParameter)>(
        dlsym(RTLD_NEXT, "anariSetParameter"));
    set(d, camera, "position", ANARI_FLOAT32_VEC3, p.data());
    anariCommitParameters(d, camera);
  }
  fn(d, f);
  anariFrameReady(d, f, ANARI_WAIT);
  float duration = 0;
  int samples = 0;
  if (!anariGetProperty(d,
          f,
          "duration",
          ANARI_FLOAT32,
          &duration,
          sizeof(duration),
          ANARI_WAIT)
      || !anariGetProperty(d,
          f,
          "numSamples",
          ANARI_INT32,
          &samples,
          sizeof(samples),
          ANARI_WAIT))
    std::abort();
  const double wall = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start)
                          .count();
  size_t free = 0, total = 0;
  checked(cudaMemGetInfo(&free, &total));
  const auto i = capture.index++;
  capture.frameIndex[f] = i;
  std::fprintf(stderr,
      "CAPTURE,%u,%d,%.9g,%.9g,%zu\n",
      i,
      samples,
      duration * 1000.,
      wall,
      total - free);
}
extern "C" const void *anariMapFrame(ANARIDevice d,
    ANARIFrame f,
    const char *channel,
    uint32_t *w,
    uint32_t *h,
    ANARIDataType *type)
{
  static auto fn = reinterpret_cast<decltype(&anariMapFrame)>(
      dlsym(RTLD_NEXT, "anariMapFrame"));
  const void *p = fn(d, f, channel, w, h, type);
  const char *out = std::getenv("VISRTX_CAPTURE_HDR");
  if (!out || !p || std::strcmp(channel, "channel.colorCUDA") != 0)
    return p;
  if (*type != ANARI_FLOAT32_VEC4) {
    std::fprintf(stderr, "capture expected float4, got %d\n", int(*type));
    std::abort();
  }
  const size_t count = size_t(*w) * (*h) * 4;
  std::vector<float> linear(count);
  checked(cudaMemcpy(
      linear.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost));
  const std::string path = std::string(out) + "/frame-"
      + std::to_string(capture.frameIndex.at(f)) + ".rgba";
  FILE *file = std::fopen(path.c_str(), "wb");
  if (!file)
    std::abort();
  if (std::fwrite(linear.data(), sizeof(float), count, file) != count)
    std::abort();
  std::fclose(file);
  std::vector<unsigned char> pixels(count);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(linear[i])) {
      std::fprintf(stderr, "nonfinite captured HDR\n");
      std::abort();
    }
    float v = std::clamp(linear[i], 0.f, 1.f);
    if (i % 4 != 3)
      v = v <= .0031308f ? 12.92f * v
                         : 1.055f * std::pow(v, 1.f / 2.4f) - .055f;
    pixels[i] = static_cast<unsigned char>(std::lround(v * 255.f));
  }
  if (capture.capacity < count) {
    if (capture.display)
      checked(cudaFree(capture.display));
    checked(cudaMalloc(&capture.display, count));
    capture.capacity = count;
  }
  checked(cudaMemcpy(
      capture.display, pixels.data(), count, cudaMemcpyHostToDevice));
  *type = ANARI_UFIXED8_RGBA_SRGB;
  return capture.display;
}
