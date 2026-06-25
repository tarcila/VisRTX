# tsdFlow: IsosurfaceExtract Compute Node — Design

**Date:** 2026-06-25
**Status:** Approved (pending user spec review)
**Phase:** Spec B (isosurface compute node)

## Problem

The graph has display-only consumers of a volume `Field` (DisplayVolume, BoundingBox→DisplaySurface). There is no node that derives new geometry from the field's *values*. Add a volume-based compute node that extracts an isosurface mesh from a scalar `Field` and emits `SurfaceData`, so it flows into the existing `DisplaySurface` path.

## Backend Decision

Use the **Viskores** contour filter (`viskores::filter::contour::Contour`), not a hand-rolled marching cubes and not a render-time isosurface.

- Viskores 1.1.0 is installed in this build (`Viskores_DIR=/nix/store/...viskores-1.1.0/lib/cmake/viskores`); `viskores::filter_contour` and `viskores/filter/contour/Contour.h` are present.
- The existing `tsdDemoViskores` app already demonstrates the marshaling pattern: `viskores::cont::DataSetBuilderUniform::Create(...)` + `dataset.AddField(name, Association::Points, handle)`.
- Render-time isosurface is rejected: it would be a display mode that emits no `SurfaceData` and does not flow into `DisplaySurface` — a different feature, out of scope.

Trade-off accepted: Viskores becomes a dependency of `tsd_graph_nodes`. It is made **optional** (auto-detected) so builds without Viskores are unaffected.

## Component: `IsosurfaceExtract` node

### Build gating (optional dependency)

In `tsd/src/tsd/graph_nodes/CMakeLists.txt`:

- `find_package(Viskores 1.1.0 QUIET)`.
- `if(Viskores_FOUND)`: add `IsosurfaceExtract.cpp` to sources; `project_link_libraries(PUBLIC viskores::filter_contour viskores::cont)`; `target_compile_definitions(tsd_graph_nodes PUBLIC TSD_GRAPH_NODES_HAVE_VISKORES)`.
- When not found: the node source, link, and macro are all omitted; nothing else changes.

`registerBuiltinNodes` (in `BuiltinNodes.cpp`) calls `registerIsosurfaceExtract(reg)` inside `#ifdef TSD_GRAPH_NODES_HAVE_VISKORES`. The declaration in `BuiltinNodes.hpp` is likewise guarded. In this environment Viskores is found, so the node compiles, registers, and its test runs.

Never run clang-format on `CMakeLists.txt`.

### Node interface (mirrors `BoundingBox`)

- name: `IsosurfaceExtract`; category: `processor`.
- input: `in` = `portField()` (a `Field`), required.
- output: `out` = `portSurface()` (a `SurfaceData`).
- params (both on the `ParameterList`, so editing them changes the eval hash and re-extracts — correct, the geometry depends on them):
  - `isovalue`: `float`, default `0.5`.
  - `computeNormals`: `bool`, default `true`.

### evaluate() data flow

1. `auto f = std::static_pointer_cast<Field>(ctx.input(Token("in"), hostResidency()).payload);` — `ctx.fail("IsosurfaceExtract: missing field input")` and return if null.
2. Validate: `dims.x/y/z > 0`; `f->data.size() == dims.x*dims.y*dims.z`; `f->data.type() == ANARI_FLOAT32`. Any failure → `ctx.fail(...)` with a specific message, return.
3. One-time, thread-safe `viskores::cont::Initialize()` guarded by a function-local `std::once_flag`. Use the default (Serial host) device — do **not** force CUDA. (Field data is host-resident; determinism on the Evaluator worker thread is preferred over GPU.)
4. Build a uniform dataset:
   - `viskores::cont::DataSetBuilderUniform builder;`
   - `auto ds = builder.Create(viskores::Id3(dims.x, dims.y, dims.z), viskores::Vec3f(origin...), viskores::Vec3f(spacing...));`
   - Create a `viskores::cont::ArrayHandle<viskores::Float32>` of `dims` values copied from `f->data` (x-fastest, matching `GenerateNoiseVolume`), and `ds.AddField(viskores::cont::Field("scalars", Association::Points, handle));`
5. Contour:
   - `viskores::filter::contour::Contour c;`
   - `c.SetActiveField("scalars"); c.SetIsoValue(isovalue); c.SetGenerateNormals(computeNormals); c.SetMergeDuplicatePoints(true);`
   - `auto result = c.Execute(ds);`
6. Extract into a `SurfaceData s` (`s.geomSubtype = Token("triangle")`):
   - Coordinates → `vertex.position`: read `result.GetCoordinateSystem().GetDataAsMultiplexer()` as `ArrayHandle<Vec3f>`; copy into an `AnyArray(ANARI_FLOAT32_VEC3, N)` → `s.prim.arrays.push_back({Token("vertex.position"), pos})`.
   - Connectivity → `primitive.index`: the cell set is triangles (`CellSetSingleType`); read its connectivity `ArrayHandle<viskores::Id>` (length `3*M`), convert each `viskores::Id` to `uint32`, pack as `AnyArray(ANARI_UINT32_VEC3, M)` → `s.prim.arrays.push_back({Token("primitive.index"), idx})`.
   - Normals (only if `computeNormals`): point field `"normals"` as `ArrayHandle<Vec3f>` → `AnyArray(ANARI_FLOAT32_VEC3, N)` → `s.prim.arrays.push_back({Token("vertex.normal"), nrm})`.
   - Appearance: `s.appearance.scalars.push_back({Token("color"), Any(params.getOr<float3>(Token("color"), float3(0.8f)))})` (same as `BoundingBox`).
7. Emit: `Value out{type=PortType{portSurface()}, residency=hostResidency(), payload=s}`; `ctx.setOutput(Token("out"), out)`.

### Edge cases

- Missing / wrong-type / zero-dim field → `ctx.fail(...)`, loud.
- Isovalue outside the data range → `Contour` yields an empty mesh → a valid `SurfaceData` with zero-length arrays. This is **not** a failure; it renders nothing. The bridge's `buildSurface` must tolerate empty arrays (it builds an ANARI geometry from whatever arrays are present).

## Demo graph

Extend `buildVolumeSurfaceDemo` (`DemoGraph.cpp`), inside `#ifdef TSD_GRAPH_NODES_HAVE_VISKORES`:

- `const NodeId iso = add("IsosurfaceExtract");`
- `const NodeId dsIso = add("DisplaySurface");`
- `link(src, "out", iso, "in"); link(iso, "out", dsIso, "in");`

`DemoDisplays` is unchanged: the app's generic display collection (`collectDisplayMasks` / `collectDisplayTransforms`, which iterate all nodes) picks up the extra `DisplaySurface` automatically, so the new branch needs no app-side wiring. The extra display gets the default viewport mask like any other.

## Testing

`tsd/tests/test_nodes_Isosurface.cpp`, added to the CTest suite only when Viskores is found (CMake-gated), following the existing `test_nodes_Surface` harness (construct a node, evaluate it in isolation, inspect the output `SurfaceData`).

- **Extraction case:** build a small `structuredRegular` `Field` (e.g. 16³) whose values are a sphere distance function `v = R - |p - center|`; set `isovalue` to a value inside the range; run the node; assert:
  - `geomSubtype == Token("triangle")`;
  - `vertex.position` present and non-empty;
  - `primitive.index` present, non-empty, every index `< vertexCount`;
  - all positions within `[origin, origin + spacing*dims]`;
  - `vertex.normal` present and same length as `vertex.position` when `computeNormals == true`.
- **Empty case:** `isovalue` beyond the field's value range → zero triangles, no crash, valid empty `SurfaceData`.

No new test for the demo-graph wiring (covered by the node test plus the existing graph build assertions). The full suite must stay green.

## Out of Scope (v1)

- Explicit CUDA / device selection (Serial host only).
- Multiple isovalues / nested surfaces.
- Flying-edges vs marching-cells selection (use `Contour`'s default).
- Value-range-driven isovalue UI / slider bounds from the field.
- Persistence of node state (Phase 5).
