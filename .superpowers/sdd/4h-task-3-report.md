## Task 3 Report: `GraphRenderBridge::setDisplayTransform`

### Implemented

- `Display::transform{tsd::math::IDENTITY_MAT4}` field added to the private `Display` struct in `GraphRenderBridge.hpp`.
- `void setDisplayTransform(NodeId, const tsd::math::mat4&)` declared in `public:` section after `removeDisplay`.
- Setter implemented in `GraphRenderBridge.cpp`: finds the display by node ID and stores the matrix.
- `update()` loop extended: after `rebuildLayer`, applies `(*d.layer->root())->setAsTransform(d.transform)` guarded by `if (d.layer)`.

### API spelling (verified against source)

| API | Actual spelling | Location |
|---|---|---|
| `Layer::root()` | `LayerNodeRef root() const` | `Layer.hpp:46` |
| Set transform | `LayerNodeData::setAsTransform(const math::mat4&)` | `LayerNodeData.hpp:63` |
| Get transform | `LayerNodeData::getTransform() const` | `LayerNodeData.hpp:73` |

### Const-ness / indirection adjustment

The brief's `root->getTransform()` and `d.layer->root()->setAsTransform(m)` do **not** compile as written. `root()` returns `LayerNodeRef = ObjectPoolRef<ForestNode<LayerNodeData>>`. `ObjectPoolRef::operator->()` yields `ForestNode<LayerNodeData>*` (not `LayerNodeData*`), so a single `->` hits `ForestNode` which has no `getTransform()`/`setAsTransform()`. The correct double-dereference idiom (confirmed from `TransformBinding.cpp:105` and `serialization_datatree.cpp:506`) is `(*nodeRef)->method()`:

- Bridge: `(*d.layer->root())->setAsTransform(d.transform)`
- Test: `(*root)->getTransform()[3].x`

### TDD RED/GREEN

**RED** (before implementing):
```
error: 'class tsd::rendering::GraphRenderBridge' has no member named 'setDisplayTransform'
error: 'const struct tsd::core::ForestNode<tsd::scene::LayerNodeData>' has no member named 'getTransform'
```

**GREEN** (after implementing + fixing the double-dereference in the test):
```
1/1 Test #46: tsd::rendering::BridgeTransform ...   Passed    1.39 sec
100% tests passed, 0 tests failed out of 1
```

### Files changed

- `tsd/src/tsd/rendering/bridge/GraphRenderBridge.hpp` — `Display::transform` field + `setDisplayTransform` declaration
- `tsd/src/tsd/rendering/bridge/GraphRenderBridge.cpp` — `setDisplayTransform` implementation + `update()` transform application
- `tsd/tests/test_bridge_Transform.cpp` — new test (adjusted dereference from brief)
- `tsd/tests/CMakeLists.txt` — source + `add_test` + `TIMEOUT 300`

### Commit

`d72fa418` — `feat(rendering): GraphRenderBridge::setDisplayTransform applies to the display layer-root`

### Self-review / concerns

- The transform is applied unconditionally every `update()` (even when `rebuildLayer` early-returns on version match). That's intentional and cheap — it avoids a dirty-flag and keeps the root in sync across calls.
- `setDisplayTransform` silently no-ops if the node isn't registered; consistent with `setDisplay` behavior.
- No concerns.
