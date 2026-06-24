## Task 1: wireframe `BoundingBox`

**Files:** Modify `tsd/src/tsd/graph_nodes/BoundingBox.cpp`; Test `tsd/tests/test_nodes_Surface.cpp` (extend).

**Interfaces:** Produces: `BoundingBox` `SurfaceData` with `geomSubtype == Token("cylinder")`, `prim.arrays` `vertex.position` (24 float3 = 12 edges × 2), `prim.scalars` `radius` (>0). Downstream `DisplaySurface` → `Renderable` `primSubtype == "cylinder"` (unchanged mapping).

- [ ] **Step 1: Update the existing failing test** — in `tsd/tests/test_nodes_Surface.cpp`, the `[nodes-surface]` BoundingBox scenario currently asserts `geomSubtype == Token("triangle")` and 36 vertices. Change those assertions to:

```cpp
    THEN("it is a cylinder wireframe with 24 vertex positions and a radius")
    {
      REQUIRE(s != nullptr);
      REQUIRE(s->geomSubtype == Token("cylinder"));
      // 12 box edges x 2 endpoints
      bool foundPos = false, foundRadius = false;
      for (const auto &a : s->prim.arrays)
        if (a.first == Token("vertex.position")) {
          REQUIRE(a.second.size() == 24);
          foundPos = true;
        }
      for (const auto &sc : s->prim.scalars)
        if (sc.first == Token("radius")) {
          REQUIRE(sc.second.get<float>() > 0.f);
          foundRadius = true;
        }
      REQUIRE(foundPos);
      REQUIRE(foundRadius);
    }
```
And update the downstream `DisplaySurface` assertion in the same scenario from `primSubtype == Token("triangle")` to `primSubtype == Token("cylinder")`. (Keep the rest of the scenario; `AnyArray::size()` and `Any::get<float>()` are the accessors already used elsewhere in the suite.)

- [ ] **Step 2: Build + confirm FAIL**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '\[nodes-surface\]' --output-on-failure 2>&1 | tail -15`
Expected: FAIL (still emits triangle/36).

- [ ] **Step 3: Rewrite the geometry in `BoundingBox::evaluate`.** Replace the 36-index triangle table + the `pos(...,36)` build + `geomSubtype = Token("triangle")` block (everything from the `const int tri[36] = {...}` through the `s->prim.arrays.push_back({Token("vertex.position"), pos});` line) with the 12-edge cylinder build. The 8 corners `c[8]` and `lo`/`hi` are computed above and stay. Use:

```cpp
    // 12 box edges as cylinder segments (consecutive vertex.position pairs).
    static const int edge[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}}; // verticals
    tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, 24);
    for (int e = 0; e < 12; ++e) {
      pos.get<float3>(size_t(2 * e)) = c[edge[e][0]];
      pos.get<float3>(size_t(2 * e + 1)) = c[edge[e][1]];
    }
    const float3 d = hi - lo;
    const float radius = std::max(0.004f * std::sqrt(dot(d, d)), 1e-4f);

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("cylinder");
    s->prim.arrays.push_back({Token("vertex.position"), pos});
    s->prim.scalars.push_back({Token("radius"), tsd::core::Any(radius)});
    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});
```
(Keep the trailing `Value out; out.type = PortType{portSurface()}; out.residency = hostResidency(); out.payload = s; ctx.setOutput(...)` unchanged. Add `#include <cmath>` if not present — it is, for nothing yet; `std::sqrt`/`std::max` need `<cmath>`/`<algorithm>`, add both. `dot` is `tsd::core::math::dot` (linalg), available via the existing math include.)

- [ ] **Step 4: Build + run the test → PASS**

Run: `cmake --build _out/_cmake --config RelWithDebInfo --target tsdTests && ctest --test-dir _out/_cmake/tsd/tests -C RelWithDebInfo -R '\[nodes-surface\]' --output-on-failure 2>&1 | tail -8`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
clang-format -i tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/tests/test_nodes_Surface.cpp
jj commit tsd/src/tsd/graph_nodes/BoundingBox.cpp tsd/tests/test_nodes_Surface.cpp -m "feat(graph_nodes): BoundingBox emits a cylinder wireframe (non-occluding)"
```

---

