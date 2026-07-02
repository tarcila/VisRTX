# VisRTX Project Guide

VisRTX is an NVIDIA visualization framework implementing the Khronos ANARI standard.

This file is the **repo-root, workspace-agnostic** guide. Per-component context lives in:

- `<workspace>/CLAUDE.md` — project / device / TSD architecture
- `<workspace>/devices/rtx/CLAUDE.md`, `<workspace>/devices/gl/CLAUDE.md`, `<workspace>/tsd/CLAUDE.md` — per-component details
- `<workspace>/STYLEGUIDE.md` — C++/CUDA coding conventions

Don't restate what those files already cover.

## Version Control: Jujutsu

This project uses **Jujutsu (jj)**. Never use raw `git` in this repo.

Key concepts:
- Working copy is tracked automatically (no staging)
- Changes are immutable once published — use `jj new` to start fresh work
- `jj describe` sets commit message; `jj log` shows history

### Workspace Workflow

The repo uses multiple jj workspaces — each is an independent checkout at a different revision, sharing one underlying store. Workspaces sit as sibling directories under the repo root (`next_release/`, `majorant/`, `pbrt-v4-importer/`, …).

```bash
jj workspace list                              # list all workspaces
jj workspace add ../VisRTX-feature-name        # new workspace at @
jj workspace add ../VisRTX-bugfix --revision @- # at specific revision
cd ../VisRTX-other-workspace                   # switch focus
```

**Naming**: descriptive, hyphen-separated reflecting work focus (`VisRTX-volume-sampler`, `VisRTX-fix-triangle-normals`).

Each workspace has its own build dir, so RTX work / GL work / TSD work can run in parallel without context switching.

### Common Commands

```bash
jj status               # working copy status
jj diff                 # uncommitted changes
jj log                  # graphical history
jj new                  # new change on top of current
jj edit <rev>           # edit existing change (working copy IS that rev)
jj squash               # squash into parent
jj rebase -d main       # rebase onto main
jj abandon              # discard current change
```

### Feature Development Loop

1. Build after each change: `just build`
2. Commit with a terse message: `jj commit -m "Add foo support"`
3. To iterate on an existing commit: `jj edit <rev>` → fix → `just build` → `jj new <tip>` to return

Keeps history clean: each commit is a complete, working change.

## Build (via `just`)

`Justfile` lives at the repo root and is shared across workspaces.

```bash
just prepare            # generate .clangd, .envrc, VS Code, Emacs configs
just configure          # configure CMake (MDL on, GL off, TSD on, tests on)
just configure_mdl_off  # configure without MDL
just configure_physicallybased_mdl
just build              # build RelWithDebInfo (default)
just br | brd | bd      # Release / RelWithDebInfo / Debug
just rebuild            # clean + build
just view  *ARGS        # run tsdViewer
just render *ARGS       # run tsdRender
just clean              # remove _out/
```

Run `just --list` for the current set; this list may drift.

Build dir: `_out/_cmake/`. Install dir: `_out/`. Binaries: `_out/bin/`.

For the full CMake option set and direct-cmake instructions, see `<workspace>/CLAUDE.md` and `<workspace>/devices/*/CLAUDE.md`.

## CMake Patterns

### Optional feature flags (NVML, MDL, NVTX, …)

Convention: CMake option uses `ENABLE_` prefix; preprocessor define uses `USE_` prefix.

1. Component root: declare and hide
   ```cmake
   option(VISRTX_ENABLE_NVML "Enable NVML for runtime driver checks" ON)
   mark_as_advanced(VISRTX_ENABLE_NVML)
   ```
2. Target CMakeLists: add define
   ```cmake
   if (VISRTX_ENABLE_NVML)
     project_compile_definitions(PRIVATE USE_NVML)
   endif()
   ```
3. Source: guard with brace block so locals stay scoped
   ```cpp
   #ifdef USE_NVML
   { /* code */ }
   #endif
   ```

Reference: `devices/rtx/CMakeLists.txt`, `devices/rtx/device/CMakeLists.txt`.

### External dependency fetching

Use `anari_sdk_fetch_project(NAME ... URL ... MD5 ...)` (from `find_package(anari)`) — caches under `.anari_deps/`, returns `${NAME}_LOCATION`.

- Prefer archive URLs (`.zip`, `.tar.gz`) over git clones — faster, deterministic when hash-pinned
- Add with `EXCLUDE_FROM_ALL` so external targets don't build by default

Reference: `devices/rtx/CMakeLists.txt` (OptiX headers), `tsd/external/tsd_imguizmo/CMakeLists.txt`.

## Environment Setup

Nix flake provides the dev shell:

```bash
nix develop        # or: direnv allow
```

The shell wires up multiple ANARI devices (visrtx, barney, helide, cycles, ospray, visionaray; visgl on x86_64 Linux). For env vars (`TSD_ANARI_LIBRARIES`, `ANARI_LIBRARY`, …) see `<workspace>/CLAUDE.md` and `<workspace>/tsd/CLAUDE.md`.

## CI/CD

GitHub Actions in `.github/workflows/`:
- `build-visrtx.yml` — RTX device CI
- `build-tsd.yml` — TSD CI (Ubuntu/Windows × Release/Debug × optional CUDA)

Custom actions in `.github/actions/` (ANARI-SDK build, CUDA setup).

## External References

### ANARI Standard

| Resource | URL |
|----------|-----|
| ANARI 1.2 Specification | https://registry.khronos.org/ANARI/specs/1.2-draft/ANARI-1.2.html |
| ANARI Registry | https://registry.khronos.org/ANARI/ |
| ANARI-SDK | https://github.com/KhronosGroup/ANARI-SDK |
| ANARI-Docs | https://github.com/KhronosGroup/ANARI-Docs |
| Khronos ANARI Overview | https://www.khronos.org/anari/ |

For ANARI behavior questions, cite `~/Code/ANARI/ANARI-Docs/next_release/chapters/`, not device source.

### NVIDIA

| Resource | URL |
|----------|-----|
| VisRTX | https://github.com/NVIDIA/VisRTX |
| OptiX 9.1 Programming Guide | https://raytracing-docs.nvidia.com/optix9/guide/index.html |
| OptiX 7.7 Programming Guide | https://raytracing-docs.nvidia.com/optix7/guide/index.html |
| CUDA Programming Guide | https://docs.nvidia.com/cuda/cuda-programming-guide/index.html |
| MDL SDK | https://github.com/NVIDIA/MDL-SDK |
| MDL Documentation | https://raytracing-docs.nvidia.com/mdl/index.html |
| MDL Handbook | https://www.mdlhandbook.com/ |

### Supporting Libraries

| Resource | URL |
|----------|-----|
| GLM | https://github.com/g-truc/glm |
| GLM API | https://glm.g-truc.net/0.9.9/api/index.html |
| OpenVDB / NanoVDB | https://github.com/AcademySoftwareFoundation/openvdb |
| NanoVDB docs | https://academysoftwarefoundation.github.io/openvdb/NanoVDB_MainPage.html |
