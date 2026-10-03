# 08: Renderer carve-outs, manual mode, and an audit of array links

**What to build:** Finishes off the remaining edge cases of the
scoped-invalidation model.

- **Renderer carve-outs:** `denoise*` and `sampleLimit` affect only
  post-processing or the stop condition. Changing them does not trigger an
  Accumulation Reset.
- **Pipeline:** an MDL material library change rebuilds the Renderer Pipeline
  of every renderer that uses it. This stays device-global, because it is
  device state. A Frame resets only if its own renderer's pipeline rebuilt.
- **Manual mode:** with `accumulationVersion` set, the application owns resets,
  except that a buffer reallocation (size or channel change) always resets.
- **Array audit:** every array a geometry, spatial field or light references
  must notify its owner when it changes. Today several don't (cone, curve and
  cylinder arrays, isosurface → field, among others). Convert them, and add a
  test for one representative of each.

**Blocked by:** 03 (World contents reach the Frame).

**Milestone:** scoped-invalidation

**Status:** wontfix

- [ ] Toggling the renderer's `denoise`: neither Frame resets
- [ ] Changing the renderer's `sampleLimit`: neither Frame resets
- [ ] An MDL library change: Frames using an MDL-dependent renderer reset;
      Frames whose renderer did not rebuild do not
- [ ] Manual mode, resizing Frame A: A resets; an unchanged
      `accumulationVersion` otherwise prevents resets
- [ ] Unmapping a cylinder's radius array: the owning geometry re-finalizes,
      and its Frame resets
- [ ] `TestMdlPipelineRebuild` and all existing RTX API tests pass

## Comments

### 2026-10-03 iteration — wontfix
split into 10, 11, 12. The storage-sensitive renderer carve-outs/manual mode,
flush-ordered pipeline invalidation, and live array-edge audit are separate
verification units; completing all three safely exceeds one 20-minute sitting.
Read resolved blocker 03, prior outcomes 01–07 and hole 09. Actual cone/curve/
cylinder and isosurface references are already observed; the remaining live
plain array references and pipeline ordering findings are recorded in the parts.
Added a shared Milestone line because the original lacked one. No implementation
started, no acceptance criteria claimed, and spec/addenda remain untouched.
Verified: `python3` ticket-graph validation (recorded in
`.scratch/scoped-invalidation/logs/08-split-validation.log`) → original wontfix;
three ready-for-agent children share scoped-invalidation milestone; blocker
references exist and graph is acyclic; all listed file paths/globs exist.
Verified: `jj diff --check` → unsupported by installed jj; replaced with
`python3` whitespace validation of the four ticket files → pass.
No build/test run: documentation-only split.
Holes filed: none
