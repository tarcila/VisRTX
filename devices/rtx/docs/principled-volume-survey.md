# VFX volume model survey

Purpose: establish precedents for the [principledVolume design](principled-volume.md).
This is a documentation survey, not an implementation-source audit or a claim of
pixel compatibility between renderers. Official pages below were retrieved during
the design session. Live manuals may change; their labels are recorded where known.

## Blender / Cycles Principled Volume

Source: [Principled Volume, Blender manual](https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/volume_principled.html)
(retrieved page labeled Blender 5.2 LTS).

The manual lists density, scattering color, anisotropy, absorption color, emission
strength/color, blackbody intensity/tint, and temperature. It provides named density,
color, and temperature attributes for volume grids. Temperature is in kelvin;
blackbody intensity one is described as physically accurate.

The listed node inputs do not include a color/opacity transfer-function table.
That is a statement about this node's documented interface, not about what Blender
shader graphs can author upstream. The short manual is insufficient to establish
exact coefficient equations, emission-density coupling, or numerical calibration.

Design implication: an integrated VFX medium should support scattering and thermal
emission, not just opacity plus unlit color. Do not assume Blender's absorption
color has the same convention as our proposed transmission color.

## Arnold Standard Volume

Source: [Standard Volume, Arnold Core manual](https://help.autodesk.com/cloudhelp/ENU/AR-Core/files/ac-shading/ac-volume-shaders/arnold_core_ac_standard_volume_html.html).

The shader offers density, scatter weight/color, transparent color/depth, anisotropy,
emission, and temperature/blackbody controls. Volume channels can drive components;
other parameters act as multipliers. Shader connections can provide more complex
variation. The documentation warns that volume shader evaluation is frequent and
recommends lean shading networks.

Scatter weight is described as the ratio scattered rather than absorbed. The phase
function is Henyey–Greenstein, with an optional second lobe and blend. Transparent
color/depth control attenuation. The documented interface does not list a native
TF color/opacity table.

Emission modes include none, channel, density, and blackbody. Independently, emission
scaling can be none, opacity-based, or thermodynamic. The thermodynamic description
accounts for reduced thermal emission at higher scattering albedo. The manual calls
uncoupled scaling unphysical in this context; our design deliberately permits an
independent artistic RGB source while coupling the thermal term to absorption.
Temperature controls include a kelvin conversion multiplier, blackbody intensity,
and artistic contrast.

Design implication: do not leave emission's coupling implicit. Temperature-driven
fire, independent RGB emission, and two-lobe scattering have direct VFX precedents.
Our proposed equations are our contract, not a reverse-engineered Arnold formula.

## Houdini / Karma volume authoring

Sources:
- [Karma Pyro Smoke Color](https://www.sidefx.com/docs/houdini/nodes/vop/kma_pyrosmokecolor.html)
- [Pyro Bake Volume](https://www.sidefx.com/docs/houdini/nodes/sop/pyrobakevolume.html)

The official Karma Pyro Smoke Color documentation describes a node that reads a
density volume and outputs scattering and absorption for Karma Volume. Its controls
include density remapping, smoke color, absorption color, shadow color, and ramps
with source ranges. Pyro Bake Volume likewise documents a smoke-color ramp driven
by density and named volume bindings.

Thus scalar-to-optical-property remapping is present in real VFX workflows. It would
be incorrect to claim that VFX renderers never use transfer functions. It is still
reasonable to keep those authoring operations outside a compact core medium.

Evidence limitation: direct SideFX extraction included extensive navigation; the
specific parameter descriptions were also retrieved through search excerpts of
these official pages. No claim of a complete Karma core-input audit is made.
An initially suggested Houdini node named “Principled Volume” was not verified;
its guessed documentation URL returned 404. Do not cite that as an existing node.

## RenderMan PxrVolume

Source: [PxrVolume, RenderMan 27](https://renderman.atlassian.net/wiki/spaces/REN27/pages/542220781/PxrVolume)
([Pixar-hosted route](https://rmanwiki-27.pixar.com/space/REN27/542220781/PxrVolume)).
Both routes became directly accessible during the session.

The page explicitly describes PxrVolume as a basic material with a small number of
volumetric properties; complicated properties belong upstream or in baked volume
data such as OpenVDB. Density is identified as extinction, the sum of absorption
and scattering. Diffuse color is explicitly scattering albedo. Density can be
scalar or RGB; RGB density supports colored attenuation. Primvar bindings and
shader connections provide spatial variation.

It offers emission color, primary/secondary anisotropy and a lobe blend. The cloud
workflow motivates two lobes and multiple scattering. Its multiple-scattering
control is an integrator-dependent hint; aggregate volumes have different behavior.
Light-source sampling has backend/aggregate restrictions.

The page contains a stale-looking performance paragraph saying volume emission
cannot yet be sampled as a full light source, while its Light Source section
describes that feature and its restrictions. Do not use that paragraph as a
universal current capability claim. Neither section establishes our required
sampling algorithm.

Design implication: extinction plus albedo is a clear optical vocabulary; complex
field authoring need not be embedded in the core volume. Colored extinction and
multiple scattering are substantive transport requirements, not just extra controls.

## Current VisRTX implementation

Repository paths below are relative to `devices/rtx/`:

- `device/volume/TransferFunction1D.cpp`: one scalar field, RGB/opacity classification,
  reference `unitDistance`; color alpha multiplies opacity.
- `device/gpu/volumeIntegrationDetail.h`: marching accumulates unlit source color
  with opacity-corrected steps. The distance-sampling path converts opacity to
  scalar extinction and returns TF RGB as albedo.
- `device/renderer/Quality_ptx.cu`: volume lighting uses that albedo and an isotropic
  phase; continuation multiplies throughput by the albedo.
- `device/gpu/renderer/raygen_helpers.h`: interactive rendering composites marched
  volume color into the camera result.

This confirms a semantic difference, not just a difference in sampling quality:
interactive treats color as an emission–absorption source color, while quality
uses it as scattering albedo. The accepted design removes that implicit switch.

## Synthesis and limits

The agreed design borrows common concepts, not one renderer's complete parameter
set or numerical implementation:

- Optical controls with explicit field-driven variation.
- Density/extinction, scattering albedo, phase, and emission as distinct concepts.
- Thermal emission as a first-class fire workflow.
- Remapping outside the core subtype, while acknowledging Houdini's ramp workflows.
- Stable optical meanings across integrators; declared approximations instead of
  redefining the medium.

None of these sources alone specifies our ANARI field representation, required
bounds, outside-value contract, RGB calibration, or overlap algorithm. Those are
local design decisions recorded separately. Blackbody normalization still needs
primary-source and repository verification before an implementation spec is frozen.
