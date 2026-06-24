# Task 2 Report: ITransformableNode + collectDisplayTransforms

## Status: DONE

## TDD RED / GREEN

**RED** (Step 3):
```
FAILED: tsd/tests/CMakeFiles/tsdTests.dir/RelWithDebInfo/test_nodes_DisplayTransform.cpp.o
fatal error: tsd/graph_nodes/DisplayTransform.hpp: No such file or directory
```

**GREEN** (Step 9):
```
Start 38: tsd::nodes::DisplayTransform
1/1 Test #38: tsd::nodes::DisplayTransform .....   Passed    0.92 sec
100% tests passed, 0 tests failed out of 1
```

## Commit

SHA: `1dfe120e`  
Subject: `feat(graph_nodes): ITransformableNode on display nodes + collectDisplayTransforms`

## Files Changed

| File | Action |
|------|--------|
| `tsd/src/tsd/graph_nodes/TransformableNode.hpp` | Created — `ITransformableNode` interface |
| `tsd/src/tsd/graph_nodes/DisplayTransform.hpp` | Created — `DisplayTransform` struct + `collectDisplayTransforms` decl |
| `tsd/src/tsd/graph_nodes/DisplayTransform.cpp` | Created — `collectDisplayTransforms` iterates via `dynamic_cast<ITransformableNode*>` |
| `tsd/src/tsd/graph_nodes/DisplayVolume.cpp` | Modified — added `ITransformableNode` base, `m_transform` member, `transform()` override |
| `tsd/src/tsd/graph_nodes/DisplaySurface.cpp` | Modified — same as DisplayVolume |
| `tsd/src/tsd/graph_nodes/CMakeLists.txt` | Modified — added `DisplayTransform.cpp` to sources |
| `tsd/tests/test_nodes_DisplayTransform.cpp` | Created — device-free unit test |
| `tsd/tests/CMakeLists.txt` | Modified — added source + `add_test` for `[display-transform]` |

## Self-Review

- `evaluate()` in both display nodes is untouched — `m_transform` is not read there.
- `m_transform` is not in `ParameterList`, so it won't enter `ParameterList::hash()`.
- `dynamic_cast` works because `Node` has a virtual destructor (confirmed from task brief).
- `collectDisplayTransforms` mirrors `collectDisplayMasks` structure exactly, selecting via `ITransformableNode` instead of typeInfo name.
- No `.envrc` committed.

## Concerns

- **mat4 operator==**: linalg's element-wise `==` worked correctly for both identity vs identity and identity vs modified-translation assertions. No flakiness — the test compares exact float values with no arithmetic transformations.
- No other concerns.
