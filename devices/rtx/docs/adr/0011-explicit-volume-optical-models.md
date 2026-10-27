---
status: accepted
---

# Explicit volume optical models

Introduce a VFX-oriented `principledVolume` with renderer-independent absorption,
scattering, and emission semantics, while making `transferFunction1D` consistently
emission–absorption. The current quality renderer interprets TF color as scattering
albedo whereas interactive rendering composites it as source color. Selecting a
renderer should change the transport approximation, not the meaning of a medium.

The accepted compatibility trade-off is an intentional quality-rendering appearance
change with migration guidance, rather than a permanent legacy scattering switch.
Principled volumes expose field-driven optical controls but no built-in TF tables;
complex remapping belongs upstream. Explicit finite volume bounds avoid guessing a
medium domain from unrelated input fields and support homogeneous media without
dummy grids.

This is an accepted design decision, **not an implemented capability**. The
[design document](../principled-volume.md) separates accepted semantics from proposed
parameter names and unresolved field/transport interfaces. Its [survey](../principled-volume-survey.md)
records external precedents and evidence limits.
