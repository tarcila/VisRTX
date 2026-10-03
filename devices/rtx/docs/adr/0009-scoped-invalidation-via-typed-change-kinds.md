# Scoped invalidation via typed Change Kinds

Status: accepted

Invalidation used to be device-wide. A commit anywhere reset accumulation on
every Frame, and any surface or light change rebuilt every World's BVHs and
light sets. We now scope it to Frame Dependencies. When an object finalizes,
it emits one or more typed Change Kinds: Slot, BLAS, TLAS, light set,
Renderer Pipeline, and Image. These are pushed up along observer edges as
monotonic stamps during the commit-buffer flush. A fixed table per edge type
translates each child kind into the parent's kind. Each consumer records the
last stamp it consumed for each kind, so a dependency shared by several
consumers works correctly. Each consumer reacts only to the kinds it consumes:
a Frame resets its accumulation, a Group rebuilds a BLAS, a World rebuilds its
TLAS.

## Considered Options

- **Keep helium's default observer behaviour** (notify ⇒ re-finalize the
  observer). Rejected. Re-finalizing every parent when a child changes defeats
  the Registry Index design, which exists so that parents never need updating
  when a referenced object recommits.
- **A single untyped "changed" signal.** Rejected. Every material color edit
  would rebuild BVHs, even though only the Slot and the Image are stale.
- **Pull at render time** (each Frame or World walks its graph and compares
  `lastFinalized` stamps). Rejected. It costs O(scene) per frame, even when
  nothing changed.
- **Dirty bits cleared by the consumer.** Rejected. When two Frames share a
  camera, the first Frame to render clears the bit and the second never sees
  the change. That multi-frame case is what exposed the original bug.
- **Fix in helium (ANARI-SDK) first.** Deferred. The mechanism lives in VisRTX
  (overriding `notifyChangeObserver`, plus device-side edges). It is shaped so
  it can be upstreamed, and in the meantime we don't depend on other helium
  devices' release cycles.

## Consequences

- Device-level state (MDL library, search paths, MaterialX Search Chain)
  remains a deliberate global trigger, but only for the consumers of its kind.
  A Frame resets only if its Renderer Pipeline actually rebuilt.
- Deferred Upload is pure transport and no longer an invalidation source. The
  array's unmap is the change.
- Every array unmap counts as a change, even when the bytes are identical.
- A validity change carries the parent's structural kind (BLAS, light set or
  TLAS). There is no separate kind for it.
- In manual `accumulationVersion` mode the application owns Accumulation
  Resets. The exception is a frame buffer reallocation, which always resets.
