# Measurement Overlay PoC — Implementation Plan

Reference: `2026-03-14-measurement-overlay-design.md`

Each step ends with `just build` and a `jj commit`.

## Step 1: OverlayRenderPass scaffold

Add an empty `OverlayRenderPass` that loads the vector2d device, creates a
frame/camera/renderer/world, and renders a blank frame each cycle. No
compositing yet — just prove the device loads and renders without crashing.

**Files:**
- `tsd/src/tsd/rendering/pipeline/passes/OverlayRenderPass.h` — class decl
- `tsd/src/tsd/rendering/pipeline/passes/OverlayRenderPass.cpp` — impl
- `tsd/src/tsd/rendering/CMakeLists.txt` — add source

**Pass responsibilities:**
- Constructor: `anariLoadLibrary("vector2d")`, create device, renderer
  (background transparent `{0,0,0,0}`), perspective camera, empty world, frame
- `updateSize()`: resize overlay frame to match pipeline dimensions
- `render()`: sync camera params from a setter, call `anariRenderFrame`,
  `anariFrameReady`, `anariMapFrame` — but don't write to `RenderBuffers` yet
- Destructor: release all ANARI handles

**Validate:** `just build` succeeds. Commit.

## Step 2: Wire OverlayRenderPass into Viewport pipeline

Insert the pass into `Viewport::imagePipeline_populate()` after
`AnariSceneRenderPass` and before `SaveToFilePass`. Feed it camera parameters
from `BaseViewport`'s arcball each frame.

**Files:**
- `tsd/src/tsd/ui/imgui/windows/Viewport.h` — add `OverlayRenderPass *m_overlayPass`
- `tsd/src/tsd/ui/imgui/windows/Viewport.cpp` — populate + camera sync in `buildUI`/`updateFrame`

**Camera sync point:** After `BaseViewport::camera_update()` runs in
`buildUI()`, read `m_camera.arcball` position/direction/up and
`m_camera.current` fov/aspect, set them on the overlay pass via a setter
(e.g., `setCamera(pos, dir, up, fov, aspect)`).

**Validate:** `just build`, run tsdViewer, confirm no crash/regression. Commit.

## Step 3: Depth-aware compositing

Implement the per-pixel composite in `OverlayRenderPass::render()`. Map the
overlay frame's color and depth channels, iterate all pixels, merge into
`RenderBuffers`.

**Compositing logic (per pixel):**
```cpp
float overlayDepth = overlayDepthBuf[i];
auto overlayColor = overlayColorBuf[i]; // FLOAT32_VEC4, premultiplied
if (overlayColor.w > 0.f && overlayDepth <= sceneDepth[i]) {
  // alpha-over
  float invA = 1.f - overlayColor.w;
  uint32_t sc = b.color[i];
  // unpack scene RGBA, blend, repack
  ...
  b.color[i] = blended;
  b.depth[i] = overlayDepth;
}
```

**Validate:** `just build`. To test visually, temporarily hardcode a segment in
the overlay world (e.g., a diagonal line). Confirm it renders atop the scene
with correct depth occlusion. Commit.

## Step 4: MeasureTool class

Standalone class managing the state machine and vector2d object graph. No UI
wiring yet — just the data model.

**Files:**
- `tsd/src/tsd/ui/imgui/tools/MeasureTool.h` — class decl
- `tsd/src/tsd/ui/imgui/tools/MeasureTool.cpp` — impl
- `tsd/src/tsd/ui/imgui/CMakeLists.txt` — add source

**Interface:**
```cpp
class MeasureTool {
public:
  enum class State { IDLE, PICKED_A, MEASURED };

  MeasureTool(anari::Device overlayDevice);
  ~MeasureTool();

  void setPointA(tsd::math::float3 pos);
  void setPointB(tsd::math::float3 pos);
  void clear();
  State state() const;
  float distance() const;
  anari::World world() const; // the overlay world to render
};
```

**ANARI objects created in constructor:**
- World, Instance, Group
- Segment geometry (initially empty vertex.position)
- Text geometry (initially empty)
- Two Raster objects wrapping the geometries

**`setPointA`/`setPointB`:** Update vertex.position arrays, recompute text
content (`snprintf` the distance), recommit geometries/rasters/group/world.

**Validate:** `just build`. Commit.

## Step 5: Wire MeasureTool into Viewport

Connect the MeasureTool to the pick system and overlay pass.

**Files:**
- `tsd/src/tsd/ui/imgui/windows/Viewport.h` — add `std::unique_ptr<MeasureTool> m_measureTool`, `bool m_measureModeActive`
- `tsd/src/tsd/ui/imgui/windows/Viewport.cpp` — creation, pick routing, world hand-off

**Pick routing:** In `ui_picking()`, when `m_measureModeActive` is true,
single-clicks trigger a pick. The pick callback reconstructs the 3D position
(reusing the existing center-pick math) and calls
`m_measureTool->setPointA()` or `setPointB()` based on the tool's current
state.

**World hand-off:** After `MeasureTool` updates, set its world on the overlay
pass: `m_overlayPass->setWorld(m_measureTool->world())`.

**Validate:** `just build`, run tsdViewer. Click two points on a scene — the
measurement line and label should appear, depth-composited. Commit.

## Step 6: UI — mode toggle and info display

**Files:**
- `tsd/src/tsd/ui/imgui/windows/Viewport.cpp` — toolbar button and info panel

**Toolbar toggle:** Add a button or keybind (`M`) in the viewport toolbar area.
When active, change the cursor hint and route clicks to MeasureTool.

**Info display:** When in `MEASURED` state, show an ImGui overlay or status
line with:
- Point A coordinates
- Point B coordinates
- Distance

**Clear:** Escape key or toggling measure mode off calls
`m_measureTool->clear()` and sets overlay world back to empty.

**Validate:** `just build`, run tsdViewer. Full round-trip: toggle measure mode,
pick two points, see line + label + info panel, clear, re-measure. Commit.
