# Handoff — VisRTX mutable Image GPU-refresh fix

## Next-session focus

Implement and verify the **VisRTX backend fix** for existing image textures not
refreshing after their ANARI Array texels are modified. This is not another Vela
feature implementation. Barney has a separately diagnosed defect; leave that
track out of this fix unless explicitly requested.

No backend patch has been applied. Diagnosis is complete enough to start with a
small, deterministic, red-capable regression rather than another broad search.

## Context pointers — read these, do not reconstruct the conversation

Define the original notes root as:

`$HOME/Code/ANARI/vela/pi-integration/.scratch`

- **Primary diagnosis:** `texture-update-diagnosis/README.md`. Contains exact
  commands, hypotheses/controls, repeated results, loaded-plugin provenance,
  pinned source URLs, and backend-specific mechanisms/repair directions.
- **Minimal public-ANARI rendered-pixel reproduction:**
  `texture-update-diagnosis/render_probe.cpp` and built `render_probe`.
  `render_probe visrtx none` currently exits **1**: a green 1×1 texture remains
  green after correct mapped-pointer red writes, even with a fresh Frame.
- **CPU-memory control:** `texture-update-diagnosis/array_probe.cpp`. Correct
  bytes and pointer aliasing rule out the earlier ignored-map-pointer theory
  for this reproduction; do not start by adding a speculative VSR memcpy fix.
- **Control evidence:** `probe-results.log`, `dependency-probes.log`, and
  `repeat-results.log` in that diagnosis folder. Three repeats consistently
  distinguish unchanged commits, real filter changes, and Array recreation.
- **Pinned source snapshots:** `texture-update-diagnosis/sources/` includes
  `visrtx-Array.cpp`, `visrtx-Array2D.cpp`, and `visrtx-Image2D.cpp`.
  These are evidence copies, **not an editable backend checkout**.
- **Live-scene evidence:** `image-improvement/visrtx-update-observation.md`,
  `visrtx-green-test.json`, and `visrtx-green-settled.json` establish correct
  green live bytes versus a still-ruby actual VisRTX viewport.
- Vela work already delivered: `agent-matrix-image-camera/execution.md`,
  accepted `spec.md`, resolved `issues/01`–`issues/07`, and delivery evidence.
  Do not redo/split those tickets or edit the frozen spec for this backend fix.

## VisRTX diagnosis in one paragraph

`Array::unmap()` marks data modified and notifies observers. The image Sampler
observes the Array, but its finalization obtains storage through
`Array2D::acquireCUDAArray()`. That method calls `makeCudaArray()` only when no
CUDA array exists; otherwise it returns resident storage without refreshing
texels. `Array2D::uploadArrayData()` has a refresh path, but ordinary
non-emissive image sampling does not reach it. Thus a new texture object can
still sample old CUDA-array contents. Recreating the Array loads current bytes;
recommitting or genuinely changing the Sampler filter does not fix VisRTX.
See the primary diagnosis for complete source and behavioral evidence.

## Source/build provenance and practical limits

The actual loaded plugin is:

`/nix/store/hh4h60p56nck6v640043vqk54pr7j9ss-visrtx-tsd_removed-unstable-2026-09-30/lib/libanari_library_visrtx.so`

The pinned package recipe matching that version fetches NVIDIA/VisRTX commit
**`53e20fb60563cd2ec1e30526e98e1eca61856687`**. Source URLs and the corresponding
anari-nix lock revision are recorded in the diagnosis. The runtime test uses
this installed binary directly; sandbox Nix DB queries could not resolve its
deriver, so source provenance is through the locked matching package recipe,
not a successful local derivation inspection.

The previously mentioned local VisRTX checkout path was **absent** in this
session. Locate/create an appropriate backend checkout/workspace before editing;
do not pretend that changing evidence copies or Vela source patches the loaded
Nix plugin. Read that checkout's AGENTS.md, device guidance, coding standards,
and build instructions first. Use Jujutsu where configured; never raw git inside
a jj repository. Do not publish GitHub issues/PRs/pushes without user approval.

## Suggested implementation sequence

1. Rerun the minimal failing VisRTX command against the selected baseline and
   record the library actually loaded. Copy/adapt its public rendered-colour
   assertions into the existing VisRTX API-test suite before the production fix.
2. Repair dirty-data propagation into the **existing CUDA texture storage**.
   Use the appropriate modification/upload stamps; do not depend on emission
   statistics queries, device recreation, camera changes or accumulation resets.
3. Keep refresh in place where appropriate: existing CUDA texture objects and
   multiple Samplers can reference the same CUDA array. Audit acquisition/release
   balance and avoid freeing/replacing storage behind live texture objects.
   Do not broaden into unrelated resource refactoring.
4. Verify green→red→green live updates; repeated writes; shared Samplers;
   relevant application/captured/device-managed ownership; sRGB colour and linear
   alpha; and fresh versus reused Frames. Extend supported-format coverage using
   existing tests. The diagnosis's runtime proof is specifically 2D RGBA8 sRGB.
5. Run authoritative backend build/tests and the isolated probe against the
   patched library with explicit library-path provenance. The formerly failing
   `visrtx none` case must now render red and exit0; Array recreation is a control,
   not the repair. Helide remains a positive control; do not require an unrelated
   Barney patch to pass this track.
6. Review standards/correctness, remove temporary instrumentation, and document
   any deployment/restart step. Rebuilding a package without proving that the
   viewer loaded it is not verification.

A VSR RenderIndex fallback that recreates only downstream ANARI Array handles is
an alternative, **not the chosen task**. Do not replace the VSR Image, rebind its
Scene references, enable object-reference editing, or invalidate undo merely to
work around this backend defect.

## Running scene — preserve it

The Vela integration workspace is:
`$HOME/Code/ANARI/vela/agent-matrix-image-camera`.
Its installed viewer/proxy live in `_out/bin`; package/device libraries are
separate. The original Pi project remains
`$HOME/Code/ANARI/vela/pi-integration`.

The user's running viewport was switched manually to **VisRTX**. The Scene has
unsaved Fool's Mate staging (Black Queen h4, White King e1). The Queen's VSR
Image bytes are restored to original **green**, while the last confirmed VisRTX
viewport still displayed cached **ruby**. Do not silently reapply ruby, change
devices/camera, or restart the viewer. Restart/reconnect requires coordination
and can lose live edits. Consult the existing viewer-agent skill if operational
reconnection is needed; never read/share launch descriptor contents or credentials.

This diagnosis created only isolated short-lived ANARI processes and local
scratch artifacts; no installed library, production source, or live Scene was
changed. The required native regressions are independent of the user viewer.

## Suggested skills

Read these under `~/.pi/agent/skills/<skill>/SKILL.md`:

- `ask-matt` — route the next work and check repo skill setup.
- `diagnosing-bugs` — preserve the proven fast failing loop and falsifiable controls.
- `tdd` — red-before-green at the public API/rendered-pixel seam; read its linked
  testing/mocking guidance, avoid private-state-only or tautological assertions.
- `implement` — scoped build/test/commit workflow and jj translation.
- `code-review` — independent Standards and Spec axes after the fix; use this
  handoff plus primary diagnosis as the originating contract.

If the new backend checkout lacks tracker/triage configuration, follow
`setup-matt-pocock-skills` before using tracker-dependent flows. Keep working
notes/handoffs under a workspace-local `.scratch/`, not in source or GitHub.
