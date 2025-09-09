# Screen-Space Ambient Occlusion (SSAO) in VisRTX

## Overview

VisRTX now includes a Screen-Space Ambient Occlusion (SSAO) postprocessing feature that can be added to rendering pipelines. This complements the existing ray-traced ambient occlusion renderer by providing a fast postprocessing alternative.

## Key Differences

### Ray-Traced AO (Existing)
- **Location**: Built into the AmbientOcclusion renderer (`devices/rtx/renderer/AmbientOcclusion.cpp`)
- **Method**: Uses OptiX ray tracing for physically accurate AO
- **Quality**: High quality, physically based
- **Performance**: More expensive, especially at high sample counts
- **Usage**: Set renderer type to "ao" and configure `ambientSamples` parameter

### Screen-Space AO (New)
- **Location**: Postprocessing pass (`tsd/src/tsd/rendering/pipeline/passes/SSAOPass.cpp`)
- **Method**: Screen-space technique using depth buffer
- **Quality**: Good quality approximation
- **Performance**: Fast, suitable for real-time applications
- **Usage**: Add as a postprocessing step in the rendering pipeline

## Using SSAO in the TSD Viewer

### Via UI (MultiDeviceViewport)
1. Open the viewport menu bar
2. Navigate to **Viewport > Postprocessing**
3. Check **"Enable SSAO"**
4. Adjust parameters:
   - **Radius**: Controls the sampling radius (0.1 - 2.0)
   - **Bias**: Prevents self-occlusion artifacts (0.0 - 0.1)
   - **Intensity**: Controls AO strength (0.0 - 3.0)
   - **Samples**: Number of samples per pixel (8 - 128)

### Programmatically in Rendering Pipeline

```cpp
#include <tsd/rendering/pipeline/passes/SSAOPass.h>

// Create pipeline
tsd::rendering::RenderPipeline pipeline(width, height);

// Add scene rendering
auto *scenePass = pipeline.emplace_back<tsd::rendering::AnariSceneRenderPass>(device);
// ... configure scene pass ...

// Add SSAO postprocessing
auto *ssaoPass = pipeline.emplace_back<tsd::rendering::SSAOPass>();
ssaoPass->setRadius(0.5f);        // AO sampling radius
ssaoPass->setBias(0.025f);        // Depth bias
ssaoPass->setIntensity(1.2f);     // AO intensity
ssaoPass->setSampleCount(64);     // Samples per pixel

// Render
pipeline.render();
```

## Parameters

### Radius
- **Range**: 0.1 - 2.0
- **Default**: 0.5
- **Description**: Controls how far the AO sampling extends from each pixel. Larger values create softer, more diffuse occlusion but may introduce artifacts.

### Bias
- **Range**: 0.0 - 0.1
- **Default**: 0.025
- **Description**: Depth threshold to prevent self-occlusion artifacts. Increase if you see excessive darkening on flat surfaces.

### Intensity
- **Range**: 0.0 - 3.0
- **Default**: 1.0
- **Description**: Multiplier for the final AO effect. Values > 1.0 create stronger darkening.

### Sample Count
- **Range**: 8 - 128
- **Default**: 64
- **Description**: Number of samples taken per pixel. Higher values improve quality but reduce performance.

## Implementation Details

The SSAO implementation uses a hemisphere sampling approach:

1. **Kernel Generation**: Random sample points are generated in a hemisphere
2. **Depth Sampling**: For each pixel, nearby depth values are sampled
3. **Occlusion Calculation**: Depth differences determine occlusion contribution
4. **Bilateral Blur**: Results are smoothed to reduce noise
5. **Color Application**: AO factor is applied to the final color buffer

The implementation is CPU-based for simplicity but could be optimized with GPU compute shaders for better performance.

## Performance Considerations

- SSAO is applied as a postprocessing step, so it adds minimal overhead to the main rendering
- The CPU implementation may be a bottleneck for very high resolutions
- Consider reducing sample count for real-time applications
- The blur step helps reduce noise from lower sample counts

## Integration with Existing Features

- SSAO works with any VisRTX renderer (default, directLight, pathTracer, etc.)
- Compatible with denoising and other postprocessing effects
- Requires depth buffer information, so ensure depth channel is enabled
- Can be combined with ray-traced AO for different artistic effects

## Future Enhancements

Potential improvements for the SSAO implementation:

1. **GPU Implementation**: Move computation to CUDA kernels for better performance
2. **Temporal Filtering**: Use previous frame information to improve quality
3. **Hierarchical Sampling**: Multi-scale sampling for better large-scale occlusion
4. **Interleaved Sampling**: Reduce per-pixel cost with spatially distributed sampling
5. **Integration with Denoiser**: Leverage OptiX denoiser for SSAO results