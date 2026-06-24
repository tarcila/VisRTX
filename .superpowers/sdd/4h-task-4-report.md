## Task 4 Report: Inspector TRS Transform Section

### What changed

File: `tsd/src/tsd/ui/imgui/windows/Inspector.cpp`

1. Added includes:
   - `#include "tsd/graph_nodes/TransformableNode.hpp"`
   - `#include "ImGuizmo.h"`

2. Added additive transform block **after** the `if (ITransferFunctionNode) {...} else { drawParameters(*m_selected); }` block in `buildUI()`. The block dynamic_casts `gn->impl.get()` to `ITransformableNode *`; if the node implements it, renders Separator + "Transform" label + DragFloat3 for T/R/S + Reset button.

### Additive / no markDirty

- Additive: the new block is appended after the existing if/else, not inside it. A display node still calls `drawParameters` and renders its `viewportMask`.
- No `m_graph->markDirty(...)` call in the transform block — only `*m_graphDirty = true`.

### Build result

Target: `tsd_ui_imgui` (OBJECT library — compile-only, no link step)
Command: `cmake --build _out/_cmake --config RelWithDebInfo --target tsd_ui_imgui`
Result: compiled 2 TUs (Inspector.cpp + GraphViewport.cpp) with **zero warnings, zero errors**.

### Commit

SHA: `0caccb71`
Subject: `feat(ui): Inspector transform TRS section for ITransformableNode`

### Deviations

None.

### Self-review

- `&m[0].x` is correct for column-major `float4x4` — 16 contiguous floats.
- `m` is a reference into the node so writes are direct.
- clang-format reformatted the long comment on the `*m_graphDirty = true` line (split across two lines) — functionally identical.

### Concerns

None.
