# Measurement Overlay PoC — Design

Integrate the vector2d CUDA rasterizer device into tsdViewer to render
depth-composited overlays. First use case: a measurement tool that picks two
3D points and displays a connecting line with a distance label.

## Architecture

Viewport owns a second ANARI device (vector2d) alongside the main rendering
device. A new `OverlayRenderPass` in the pipeline renders the overlay frame and
composites it into the shared `RenderBuffers`. A `MeasureTool` class manages
the user interaction state machine and the vector2d ANARI object graph.

### Data flow

```
AnariSceneRenderPass (VisRTX)
  → writes color + depth to RenderBuffers
OverlayRenderPass (vector2d)
  → renders its own frame (color + depth)
  → per-pixel: if overlay.depth <= scene.depth, blend overlay over scene
  → writes merged result back to RenderBuffers
... remaining passes (Pick, AOV, Outline, Axes, CopyToSDL) ...
```

### Compositing

Depth-aware, per-pixel on CPU. The overlay frame produces premultiplied-alpha
color and linearized depth. For each pixel where `overlay.depth <= scene.depth`:

    out.color = overlay.color + scene.color * (1 - overlay.alpha)
    out.depth = overlay.depth

Where the overlay has no coverage (alpha == 0) or fails the depth test, the
scene pixel passes through unchanged.

### Camera sync

OverlayRenderPass reads the same eye/direction/up/fov/aspect from the
Viewport's camera manipulator and sets them on the vector2d perspective camera
each frame. Both devices see the same view, so depth values are comparable.

### Frame size sync

The overlay frame resizes when the viewport resizes via `updateSize()`,
matching the same `resolutionScale`.

## MeasureTool

### State machine

```
IDLE  --[click]--> PICKED_A  --[click]--> MEASURED
  ^                   |                      |
  +---[Esc/toggle]----+------[Esc/toggle]----+
  ^                                          |
  +----------[click (new A)]-----------------+
```

- **IDLE**: Measure mode active, waiting for first click.
- **PICKED_A**: Point A set. Overlay shows a marker at A.
- **MEASURED**: Both points set. Overlay shows segment A→B + distance label at
  midpoint. Clicking again starts a new measurement (replaces old).

### Vector2d object graph

```
World
 └─ Instance (identity transform)
     └─ Group
         ├─ Raster → Segment geometry
         │    vertex.position = [A, B]
         │    width = 2.0
         │    vertex.color = measurement color
         └─ Raster → Text geometry
              vertex.position = [midpoint(A, B)]
              text = ["3.42"]
              height = 16.0
              offset = (5.0, 5.0)
```

Objects are created once at device init and reused. Only parameters (positions,
text content) are updated and recommitted when measurement points change.

## UI integration

- **Toggle**: Toolbar button or keybind activates/deactivates measure mode.
- **Pick reuse**: When measure mode is active, single-clicks trigger the
  existing depth-based pick (3D world position from depth reconstruction) and
  route the result to MeasureTool instead of the arcball recenter.
- **Info display**: ImGui panel or status text shows coordinates and distance.
- **Clear**: Escape or toggling measure mode off clears the measurement.

## Device loading

The vector2d device is loaded dynamically via `anariLoadLibrary("vector2d")`.
No direct linking required — TSD only needs the ANARI loader (`anari::anari`),
which it already links. The `anari/ext/vector2d/` headers are needed at compile
time for `ANARI_RASTER` type and `anariNewRaster()`.
