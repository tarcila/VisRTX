# TSD Dataflow Pipeline App — Design

**Date:** 2026-06-16 (revised 2026-06-17 after meta-review)
**Status:** Approved design, pending implementation plan
**Working app name:** `tsdFlow`

## Summary

A new interactive TSD application driven by data interaction: load one or many
datasets, present them, apply processing, and chain results into reusable
processing pipelines via a node graph. The graph is the source of truth; display
nodes produce backend-agnostic renderable values that a render bridge realizes
into one or more viewports, selected per display object by a viewport mask.

The design is TSD-native and fresh (the existing `viskores` demo and its
`viskores_graph` library are reference only, not a dependency). v1 focuses on
scientific volumes and spatial fields, TSD's scivis strength.

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Foundation | Fresh, TSD-native dataflow design; viskores demo is reference only |
| Wire data | Generic typed ports; any registered type can be a port |
| Residency/provenance | Backend-agnostic residency descriptor on every value, designed in from the start (host/CUDA now, pluggable for Vulkan/etc.). Lineage/audit provenance deferred (recorded as a stub, no UI). |
| Execution | Lazy pull + dirty propagation; per-output caching keyed by **version stamp** (not content hash) |
| Threading | Single sequential eval thread for v1; UI stays live; nodes show a `Computing` state; results published only after backend completion, swapped in atomically on the UI thread |
| Node definition | C++ registry (descriptor-driven, UI auto-generated) + Lua-scriptable node type |
| Implicit ops | Residency transfers and type/format conversions are auto-inserted and cached, never silent — every inserted op is surfaced (wire badge + per-pull "Implicit Operations" list with direction and cost) |
| Display | Display nodes are ordinary producers of a `renderable` value; viewport mask per display object; N viewports (≤64), each renders the union of display nodes masked to it |
| Non-3D inspection | Deferred for v1 (port/type/shape tooltips only) |
| v1 catalog | Scientific volumes & fields; surface path via a simple probe/bounding-box op (cutting-plane Slice deferred) |
| Editor features | Undo/redo and copy/paste in v1; cache eviction/budget deferred (documented limitation) |
| Persistence | Graph serialized via DataTree, embedded in `.tsd`; source nodes store file references |
| Structure | Approach C: graph engine is source of truth + a thin per-viewport `GraphRenderBridge` |
| Implementation | Delivered in 5 phases (see Phasing) |

## Architecture & module layout

A new UI-free core library (`tsd_graph`) plus a new interactive app. The engine is
testable in isolation with no UI and no ANARI dependency.

```
tsd/
  src/tsd/graph/                 NEW core lib: tsd_graph (no UI, no ANARI)
    PortType.hpp                 logical type descriptors + PortTypeRegistry
    Residency.hpp                backend-agnostic residency + TransferRegistry
    Value.hpp                    typed, residency-tagged value carried on wires
    Port.hpp                     input/output port specs (incl. accepted backends)
    Node.hpp                     Node interface (ports + params + evaluate())
    NodeRegistry.hpp             self-registration of node types
    Graph.hpp                    nodes + connections (stable ids), topology, dirty
    Evaluator.hpp                lazy-pull eval, version-stamped cache, scheduling
    EvalReport.hpp               record of implicit ops per pull
    History.hpp                  snapshot-based undo/redo + clipboard
    LuaNode.hpp                  Lua-backed node (guarded by TSD_USE_LUA)
    nodes/                       v1 catalog (sources, processors, sinks)
    io/GraphSerialization.*      DataTree <-> Graph

  src/tsd/ui/imgui/windows/
    NodeEditor.h/.cpp            NEW: graph canvas window (ImNodes-style)
    GraphInspector.h/.cpp        NEW: selected-node params + implicit-ops list

  src/tsd/rendering/graph/
    GraphRenderBridge.h/.cpp     NEW: owns one RenderIndex per viewport,
                                 populated from masked display nodes

  apps/interactive/tsdFlow/      NEW app
    main.cpp                     Application subclass; NodeEditor, GraphInspector,
                                 N Viewports, GraphRenderBridge
```

### Boundaries & dependencies

- `tsd_graph` depends only on `tsd_core` (Array, Any, Token, DataTree). It does
  **not** depend on ANARI, RenderIndex, or UI. Pure dataflow engine, fully
  unit-testable headless.
- The residency/backend layer is an abstraction in `tsd_graph` (enum + pluggable
  transfer functions). The CUDA implementation self-registers when `TSD_USE_CUDA`
  is on. No hard CUDA dependency in the engine core.
- `GraphRenderBridge` (in `tsd_rendering`) is the only component aware of both the
  graph and ANARI/RenderIndex. It pulls display-node `renderable` values and
  translates them into per-viewport `RenderIndex` instances.
- The app wires UI windows <-> graph <-> bridge and owns no engine logic.

Three independently-understandable units: **engine** (what computes), **bridge**
(how results reach a viewport), **app/UI** (how a human edits the graph).

### Reuse notes (verified against the codebase)

- `RenderIndex` already exposes `setExternalInstances()`
  (`rendering/index/RenderIndex.cpp`); the bridge extends it with incremental
  add/remove rather than introducing the population path from scratch.
- `Application` already owns a thread-safe single-worker `tsd::core::TaskQueue`
  (`core/TaskQueue.hpp`) with `std::future`-based completion and an existing
  cancellable-task pattern; the evaluator's eval thread builds on it.
- `computeScalarRange`, the RAW/VTI/NanoVDB/VTU importers, `DataTree`
  nested-record serialization, and the Lua/Sol framework all exist as assumed.
- **`TransferRegistry` is a genuinely new subsystem**, not light reuse: `Array`
  exposes `HOST`/`CUDA`/`PROXY` storage but no host↔device transfer primitives,
  so transfer functions (with CUDA stream management and the version/residency
  keyed cache) are net-new code.
- `generate_noiseVolume` returns a `Volume` and the Ensight importer mutates the
  Scene; both need thin adapters to return a field `Value` instead.

## Engine internals (`tsd_graph`)

### Type system & residency

A **PortType** is a logical type identified by a `Token` (e.g. `"array"`,
`"spatialField"`, `"transferFunction"`, `"range"`, `"transform"`, `"renderable"`).
Types self-register into a `PortTypeRegistry`, so Lua and future C++ nodes
reference types by token without a central enum. The authoritative v1 token list
lives in `PortType.hpp`.

A **Residency** descriptor is orthogonal to type:

```cpp
struct Residency {
  Token backend;   // "host", "cuda"  (extensible: "vulkan", ...)
  int   deviceId;  // -1 for host
};
```

A **Value** is what travels on a wire:

```cpp
struct Value {
  PortType              type;
  Residency             residency;
  std::shared_ptr<void> payload;        // backend-specific deleter (see below)
  uint64_t              producerNodeId; // lineage hook (recorded now, UI later)
  uint64_t              version;         // monotonic, bumped each (re)emit
  std::optional<uint64_t> contentTag;    // opt-in equality tag for cheap scalars
};
```

- **Version stamps, not content hashing.** A value's `version` is assigned by its
  producer and bumped whenever that producer re-emits an output. Cache
  invalidation and the "skip unchanged subgraph" short-circuit compare versions —
  O(1), no data scan. `contentTag` is an *optional* equality check populated only
  by cheap host-side producers (e.g. `range`, `transform`) where "identical output
  despite recompute" actually pays off; large array values leave it empty.
- **Backend-specific deleters.** A `payload`'s `shared_ptr` deleter captures the
  originating backend + deviceId and frees via the registered backend allocator
  (`cudaFree` on the right context vs host `free`). A `Value` payload is valid
  **only** on its own `Residency`.
- **TransferRegistry**: residency-to-residency transfer functions keyed by
  `(PortType, fromBackend, toBackend)`. CUDA module registers host<->cuda when
  `TSD_USE_CUDA`; core ships only host identity. Each returns a descriptor with an
  estimated cost (bytes moved). Net-new subsystem (see Reuse notes).
- **ConversionRegistry**: type/format conversions keyed by `(fromType, toType)`,
  with an estimated cost (elements converted).

### Ports, Nodes, Registry

```cpp
struct PortSpec {
  Token name;
  PortType type;
  bool required;
  std::vector<Token> acceptedBackends;  // empty => any (host-preferred)
};

struct NodeTypeInfo {
  Token name, category;
  std::vector<PortSpec> inputs, outputs;
  bool isCacheable = true;   // false => always recompute on pull
};

class Node {
 public:
  virtual NodeTypeInfo typeInfo() const = 0;
  virtual ParameterList &parameters() = 0;      // reuses tsd Parameter
  virtual void evaluate(EvalContext &) = 0;     // pull inputs, set outputs
};
```

- **Per-port residency.** Inputs declare `acceptedBackends`; the evaluator
  guarantees the value arrives in an accepted residency (inserting a transfer if
  needed). An output port declares the backend it produces (or "inherits input
  *i*"). A node never performs transfers itself.
- **`EvalContext`** exposes: `ctx.input<T>("name")` (transferred to an accepted
  residency), `ctx.hasInput("name")` and `ctx.inputOr("name", default)` for
  optional ports, `ctx.param<T>("name")`, `ctx.setOutput("name", value)`, a
  cancellation token, and a progress reporter.
- **Parameters.** A node owns a `ParameterList` (existing `tsd::Parameter`). The
  inspector renders it via existing Parameter UI; a param-changed callback marks
  the node dirty. The **param hash** that feeds cache validity is the hash of the
  serialized `ParameterList`; each Parameter type defines its contribution (float
  params hashed by exact bits).
- **`isCacheable`.** Pure built-ins default to `true`. Procedural/RNG nodes and
  `LuaNode` default to `false` (the engine cannot verify Lua purity); a Lua table
  may opt in with `pure=true`. Non-cacheable nodes always recompute on pull.
- **NodeRegistry**: each C++ node type self-registers a factory + descriptor at
  static init. UI is generated entirely from `typeInfo()` ports + `parameters()`.

### Lua nodes

`LuaNode` is one C++ node type whose `evaluate()` calls a Lua function via the
existing `tsd_lua` / Sol bindings. Its port/param spec is declared in a Lua table.
Guarded by `TSD_USE_LUA`; degrades to an "unavailable node type" if Lua is off.
Lua nodes operate host-side by default (CUDA inputs transferred to host on input;
flagged as an implicit op) and are non-cacheable by default.

### Graph structure

`Graph` owns nodes (in an `ObjectPool`-style store), connections, and a
topological view. **Connections carry stable ids** (`fromNode.outPort ->
toNode.inPort`) so the UI and `EvalReport` can reference a specific wire.
Connections are validated at link time:

- **Type compatibility**: exact match, or a registered conversion exists.
- **Cycle rejection** at link time; the depth-first `pull()` also carries an
  in-progress guard to fail loud rather than overflow if a cycle slips through.
- Residency mismatch is **never** a link error — resolved at eval.
- **Fan-out**: an output port may feed many consumers (1:N); an input is 1:1.
- Connections relying on an implicit conversion are flagged so the UI can badge
  them.

### Undo/redo & clipboard

`History` keeps a snapshot/command stack over graph mutations (add/remove node,
connect/disconnect, param edit, mask edit, paste). Snapshots reuse
`GraphSerialization` (cheap given DataTree). **Copy/paste** serializes a selected
node subset and remaps ids on paste, rewiring internal connections.

## Evaluation

### Dirty tracking & lazy pull

Each node holds an `evalState` (`Clean | Dirty | Computing | Error`) and a cache
of its outputs. Cache entry shape: per output port, a map
`Residency -> {payload, version}` (so fan-out to consumers on different residencies
each gets a cached, correctly-resident copy). Mutations mark dirty and propagate
**downstream only**:

- editing a param -> that node + all transitive consumers `Dirty`
- adding/removing a connection -> downstream node + its consumers `Dirty`
- a source's file ref changing -> that source + downstream `Dirty`
- **deleting a node** -> drop its connections, mark former consumers `Dirty`; a
  consumer now missing a *required* input enters a "missing input" `Error` until
  reconnected.

Nothing recomputes on edit. Evaluation is **pulled** from sinks (a display node
via the bridge, or the inspector). The evaluator walks inputs depth-first,
recomputing a node only if it is `Dirty`, non-cacheable, or any input's `version`
differs from what produced its cached output. Clean nodes return cached outputs.

### Implicit transfers & conversions (with feedback)

When handing a producer's `Value` to a consumer port, the evaluator checks
`(type, residency)` against the port's accepted set, in order:

1. **Type mismatch** -> `ConversionRegistry`; insert conversion, or fail the
   connection.
2. **Residency mismatch** -> `TransferRegistry`; insert transfer. The transferred
   copy is cached keyed by `(source producerNodeId+version, target Residency incl.
   deviceId, target type)` and invalidated when the source version bumps. Keying on
   target `deviceId` is required so transfers to device 0 and device 1 don't
   collide.

Every inserted op is appended to an **`EvalReport`** for the current pull:
`{connectionId | synthetic-bridge-id(+viewport), kind: Transfer|Convert, from, to,
estCost, actualMicros}`. A node-internal conversion is attributed to its input
port and badged on the inbound wire. The report drives two UI surfaces:

- a **badge on the wire** in the NodeEditor (up = upload, down = download,
  bidirectional = convert), and
- the **GraphInspector "Implicit Operations" list** for the last pull, with
  direction and cost.

EvalReport is accumulated per pull-session and cleared on the next pull.
Implicit for convenience, never silent.

### Scheduling, cancellation, publish (v1)

- **Single eval thread.** `pull()` is non-blocking; a single worker thread (on
  `Application`'s `TaskQueue`) walks the dirty DAG sequentially, issuing GPU work
  to one stream. Concurrent multi-node evaluation is deferred (out of scope v1).
  This satisfies "UI stays live" without a scheduler.
- **Threading contract.** The worker operates on an **immutable snapshot** of
  topology + params captured at `pull()`. UI edits mutate the live graph and bump
  a generation counter / request cancellation; they never race the worker's
  snapshot.
- **Cancellation is cooperative**, checked between node evaluations and between
  chunked transfer segments — *not* within a kernel. Already-launched GPU work
  runs to completion and its result is **discarded**. Rapid param drags are
  **debounced before launch** (coalesced, scheduled on settle / at a throttled
  cadence) so expensive kernels launch fewer times. A cancelled node returns to
  `Dirty`, not `Error`.
- **Publish after completion.** A produced `Value` is published only after its
  backend completion handle is signaled (a CUDA event for device work). Results
  cross to the UI thread via a completed-results queue drained at frame start; the
  cache slot is replaced via atomic `shared_ptr` exchange. The bridge therefore
  only ever reads finished, correctly-resident data; the viewport keeps rendering
  the last good output until the swap.
- **Errors.** `evaluate()` failure sets `Error` with a message; downstream pulls
  short-circuit; node/wire marked in UI. No partial outputs are published.

## Render bridge & viewport masks

### Display nodes

A display node is an **ordinary node** that produces a backend-agnostic
`renderable` Value (a surface/volume descriptor) via `ctx.setOutput`. It is "a
sink" only in that the bridge is its sole consumer — there is no off-graph output
channel. Display state lives alongside the node:

```cpp
struct DisplayNodeState {
  uint64_t viewportMask;   // bit i set => visible in viewport i (max 64 viewports)
  bool     enabled;        // global on/off, independent of mask
};
```

`DisplayVolume` constructs its `renderable` from `field` + `transferFunction`
inputs; `DisplaySurface` from a `surface` input. (There is no `volume` wire type;
the ANARI volume is built in the bridge from the `renderable` descriptor.)

### The bridge

`GraphRenderBridge` is the only component aware of both `tsd_graph` and
ANARI/`RenderIndex`. It owns, per viewport `i`, one `RenderIndex` (one ANARI
world). Each frame:

```
for each viewport i:
    desired[i] = { display nodes D : D.enabled && (D.viewportMask & (1<<i)) }
```

It diffs `desired[i]` against what index `i` currently holds. For each
newly-included display node it `pull()`s the node's `renderable` Value; on
completion it translates that descriptor into ANARI objects added to index `i`'s
world (the ANARI translation lives entirely in the bridge). When a node leaves the
set (mask bit cleared, disabled, or deleted) its objects are removed from that
index.

A display node present in masks for VP0 and VP2 is realized as ANARI objects in
**two** indices. The mask is the membership predicate — no layer-visibility
retrofit, no shared editable scene.

### Residency hand-off & cross-device cost

- Because the `renderable` Value carries residency, the bridge can hand a
  CUDA-resident array straight to a CUDA-capable ANARI device via existing
  array interop — no host round-trip.
- For viewports on **different** devices (e.g. VP0 RTX / VP1 GL — both masking the
  same display node), the bridge requests the pull in each viewport's required
  residency; the evaluator returns the appropriate per-residency cached copy and
  records the transfer in the EvalReport. Payloads are shared **only** among
  viewports with identical residency.
- Cost model: per-viewport ANARI handle sets are cheap; **device memory is
  duplicated across viewports on distinct devices** (real, potentially large for
  scivis volumes — a documented cost, not a bounded one).

### Dirty -> rerender

When a display node's `renderable` is recomputed, the published output triggers
the bridge to update the affected indices; the viewport's existing render loop
redraws. Viewports whose masked nodes are unaffected do not touch their indices.

### RenderIndex extension

`RenderIndex` already exposes `setExternalInstances()` (replace-all). v1 adds a
thin **incremental add/remove** path so the bridge can update a single viewport's
world per display-node change rather than rebuilding it.

## v1 node catalog (scientific volumes & fields)

Ports shown as `name:type`.

### Sources

| Node | Outputs | Notes |
|------|---------|-------|
| `ImportRaw` | `field:spatialField` | RAW structured field; dims/spacing/dtype params; stores file ref |
| `ImportNanoVDB` | `field:spatialField` | `.nvdb`; native CUDA residency when `TSD_USE_CUDA` |
| `ImportVTI` | `field:spatialField` | VTK image data |
| `ImportVTU` | `field:spatialField` | unstructured -> field |
| `GenerateNoiseVolume` | `field:spatialField` | procedural (adapter over `generate_noiseVolume`); non-cacheable |

Sources wrap existing `tsd_io` importers (adapters where they mutate the Scene or
return a Volume); they produce a field `Value`.

### Processors

| Node | In -> Out | Notes |
|------|-----------|-------|
| `ScalarRange` | `field` -> `range` | reuses `computeScalarRange`; cheap, populates `contentTag` |
| `TransferFunction` | `field` + `range` (required) -> `transferFunction` | color-map presets; editor binds to the selected node |
| `Probe` / `BoundingBox` | `field` -> `surface` | simple surface-producing op exercising the surface path (replaces deferred cutting-plane Slice) |
| `Resample` / `Crop` | `field` -> `field` | grid ops; async/residency test cases |

`transferFunction` payload = opacity/color control points + a sampled LUT array;
`range` payload = `float2` domain.

### Sinks (display)

| Node | In | Produces (`renderable`) |
|------|----|----------|
| `DisplayVolume` | `field` + `transferFunction` | volume descriptor; carries `viewportMask` |
| `DisplaySurface` | `surface` | surface + default material; carries `viewportMask` |

Plus the generic `LuaNode` (any ports via a Lua table; non-cacheable by default).

### End-to-end demo target

*Import -> ScalarRange -> TransferFunction -> DisplayVolume* into two masked
viewports, and *Import -> Probe -> DisplaySurface* alongside — exercising sources,
processors, both sinks, masks, residency transfer (NanoVDB CUDA -> Lua host),
debounced async recompute on a TF drag, and undo/redo of edits.

## Persistence (DataTree embedded in `.tsd`)

The graph serializes into a `graph` subtree of the existing `.tsd` `DataTree`,
alongside scene/app state, via `GraphSerialization`:

```
.tsd (DataTree)
└── graph/
    ├── nodes/        [ {id, typeToken, params{}, canvasPos} ]
    ├── connections/  [ {id, fromNode, fromPort, toNode, toPort} ]
    ├── displays/     [ {nodeId, viewportMask, enabled} ]
    └── viewports/    [ {index, device, renderer, camera} ]
```

- **Node identity**: stable 64-bit ids; types referenced by registry **token**.
  Unknown token -> placeholder "missing node type" node preserving params +
  connections, so a file from a newer build round-trips.
- **Sources store file references, not baked data** — reload re-imports. A later
  opt-in "bake" can cache heavy results; out of scope for v1.
- **Viewport reconciliation**: viewports are created from the persisted
  `viewports[]`; if the runtime viewport count differs, mask bits are preserved
  and clamped to the available set.
- **Not persisted**: cached outputs, versions, `evalState`, EvalReport — all
  transient; the graph loads `Dirty` and lazily re-evaluates on first pull.
- **Round-trip is the contract**: load -> save produces an equivalent tree; a core
  test.
- **Security note**: `LuaNode` carries executable script; a loaded `.tsd` may
  contain code. Treat `.tsd` files as you would any executable content — do not
  load untrusted graphs.
- Reuses `Application`'s existing `.tsd` load/save path; the graph is another
  registered subtree, so the app's Save/Open work unchanged.

## Error handling

Validate hard at the **link** and **file** boundaries; inside the eval loop,
isolate failures to the node so the rest of the graph and the live UI survive.

| Boundary | Strategy |
|----------|----------|
| Connection attempt | Link-time validation: type mismatch with no conversion -> reject with reason; cycle -> reject. Residency mismatch never an error. UI shows the refusal reason. |
| Missing required input | Node enters "missing input" `Error` (e.g. after upstream deletion) until reconnected; surfaced as a continuous per-node validity badge independent of evaluation. |
| Missing/bad source file | Source node -> `Error` with the IO message; downstream pulls short-circuit; node + dependent wires badged. Other branches keep working. |
| Node `evaluate()` throws/fails | Caught by evaluator -> node `Error` + message; no partial outputs published; consumers short-circuit. Last good cached render stays on screen. |
| No transfer path | Eval error on that wire with explicit "no host<->X transfer registered" message; surfaced in the Implicit Operations list as a failed op, not a crash. |
| Unknown node type on load | Placeholder node preserves params + connections; file round-trips; UI flags it. |
| Cancellation | Cooperative; rapid re-dirty discards in-flight results; cancelled node returns to `Dirty`, not `Error`. |
| Lua node runtime error | Captured at the Lua/Sol boundary -> node `Error` + script message; never aborts the worker. |

## Known limitations (v1)

- **No cache eviction / budget.** Per-output + per-residency caches can grow
  unbounded on large volumes. v1 ships without eviction; an LRU budget over clean,
  re-derivable cached outputs is the planned follow-up.
- Single sequential eval thread (no concurrent multi-node evaluation).
- ≤64 viewports (`viewportMask` width).
- Lineage/audit provenance recorded as a stub only (no UI).
- Non-3D inspection limited to tooltips.

## Testing

### Headless engine unit tests (`tsd/tests/`, no UI, no ANARI)

- **Type/residency registry**: registration, lookup, conversion/transfer
  resolution; per-port accepted-backend negotiation.
- **Dirty propagation**: edit marks the correct downstream set; unrelated branches
  stay clean; node deletion marks consumers dirty / missing-input error.
- **Lazy pull + versioning**: only dirty/changed-version ancestors recompute;
  version short-circuit verified via an eval counter on a probe node;
  `isCacheable=false` node always recomputes.
- **Implicit ops**: mismatched residency/type inserts the expected
  transfer/conversion; transfer cache keyed on target deviceId (no cross-device
  collision); `EvalReport` records the connection id, direction + cost; missing
  path -> failed op, not crash.
- **Scheduling/publish**: snapshot isolation (UI edit mid-pull doesn't corrupt the
  worker); cancellation discards results and returns to `Dirty`; debounce
  coalesces rapid edits; a `Value` is never observed before its completion handle
  signals (fake-backend event).
- **Serialization**: round-trip equality (load->save->load); unknown-token
  placeholder preserves connections; undo/redo + paste id-remap round-trip.
- A fake backend (`"test"` residency + registered transfers + a completion event)
  exercises residency, multi-device keying, and publish ordering **without CUDA**,
  so these run anywhere in CI.

### Bridge tests (with a `helide`/null ANARI device)

- Mask membership -> correct per-index object sets.
- Mask edit adds/removes from the right indices (incremental, not rebuild).
- Same display node masked to two viewports of differing residency -> two
  per-residency copies, both recorded in EvalReport.
- Dirty recompute updates only affected indices.

### Manual/integration smoke (the app)

- The end-to-end demo graph above across two masked viewports.
- Debounced async recompute on a TF drag keeps the UI live and the EvalReport
  shows the NanoVDB CUDA->host transfer.
- Undo/redo and copy/paste of a subgraph.

## Phasing (implementation)

Each phase is independently testable and mergeable.

1. **Engine core, headless.** PortType/Residency/Value (version stamps),
   registries (Type/Transfer/Conversion), Port/Node/Registry (incl. `isCacheable`,
   per-port accepted backends, param binding), Graph (stable-id connections, link
   validation, cycle rejection, dirty propagation, deletion semantics), Evaluator
   (lazy pull, per-output/per-residency cache, version short-circuit) — sync only,
   `"test"` fake backend. No CUDA/UI/ANARI.
2. **Async scheduling + publish.** Single eval thread on `TaskQueue`, snapshot
   isolation, cooperative cancellation + debounce, completion-gated publish,
   `Computing` state, EvalReport accumulation, error isolation. Still headless.
3. **Bridge + viewport + RenderIndex incremental path.** `GraphRenderBridge`,
   per-viewport index, mask diffing, residency hand-off, multi-device
   per-residency copies. Bridge tests on `helide`. First ANARI dependency.
4. **Catalog + node editor UI + GraphInspector.** v1 sources/processors/sinks
   (incl. import/Volume/Ensight adapters and the Probe surface op), CUDA transfer
   functions (`TransferRegistry`), ImNodes editor, param auto-UI, wire badges,
   Implicit Operations list, TF editor panel, validity badges, undo/redo,
   copy/paste. End-to-end demo first runs here.
5. **Lua node + persistence.** `LuaNode` (guarded by `TSD_USE_LUA`) and
   `GraphSerialization` round-trip into `.tsd` with unknown-token placeholders.
   Two parallelizable workstreams; persistence can start once the Node/param model
   (Phase 1) is stable.

## Out of scope for v1

- Lineage/audit provenance UI (recorded as a stub only).
- Non-3D inspection panels (stats, histograms, tables) — tooltips only.
- Cutting-plane Slice (replaced by Probe/BoundingBox); baking source results into
  `.tsd`; cache eviction/budget; concurrent multi-node evaluation.
- Mesh/geometry and general-purpose catalogs (volumes & fields first).
- Backends beyond host + CUDA (the abstraction is pluggable; no other backend
  implemented).
- Standalone `.tsdgraph` export (graph lives in `.tsd`).
