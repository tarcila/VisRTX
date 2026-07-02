mod cmake '~/_just/cmake'
mod dotfiles '~/_just/dotfiles'
mod repo '~/_just/repo'

import '~/_just/dirs/mod.just'

cfg := env("cfg", "RelWithDebInfo")
_source_dir := _ROOT_DIR
_root_dir := _ROOT_DIR
_build_dir := _root_dir / "_out/_cmake"
_install_dir := _root_dir / "_out"
_envrc_dir := _root_dir
_clangd_dir := _root_dir
_compile_commands_dir := _build_dir
_vscode_dir := _root_dir / ".vscode"
_emacs_desktop_dir := _root_dir / ".emacs.d"
_emacs_dape_dir := _root_dir

# Per-workspace ANARI library name so dev builds coexist with an official install.
_visrtx_library_name := "visrtx_" + replace_regex(file_name(_root_dir), '[^A-Za-z0-9_]+', '_')
# Prepend this workspace's device to the flake-provided list and make it the default.
_anari_envrc := 'export TSD_ANARI_LIBRARIES="' + _visrtx_library_name + '\${TSD_ANARI_LIBRARIES:+,\${TSD_ANARI_LIBRARIES}}"' + "\n" + 'export ANARI_LIBRARY="' + _visrtx_library_name + '"'

_compiler := if os() == "macos" { "clang++" } else { "g++" }

_default: build

# Some important aliases

[doc("Build the project (Release)")]
[metadata("task", "pm=$gcc")]
br: (build "Release")

[doc("Build the project (RelWithDebInfo)")]
[metadata("task", "pm=$gcc")]
brd: (build "RelWithDebInfo")

[doc("Build the project (Debug)")]
[metadata("task", "pm=$gcc")]
bd: (build "Debug")

# Repo recipes

alias new_workspace := repo::new_workspace
alias forget_workspace := repo::forget_workspace

# Build recipes

[doc("Generate configure files")]
prepare: generate_envrc generate_clangd generate_vscode_tasks_json generate_vscode_launch_json generate_emacs_desktop generate_emacs_dape
    direnv allow

[private]
_cmake_options_base := replace_regex('
    -DBUILD_SHARED_LIBS=OFF
    -DOPTIX_FETCH_VERSION=9.0
    -DVISRTX_BUILD_GL_DEVICE=OFF
    -DVISRTX_BUILD_TSD=ON
    -DVISRTX_BUILD_TESTS=ON
', '\s*\n\s*', ' ') + " -DVISRTX_LIBRARY_NAME=" + _visrtx_library_name

[private]
_cmake_options := _cmake_options_base + " -DVISRTX_ENABLE_MDL_SUPPORT=ON"
    
[private]
_cmake_options_mdl_off := _cmake_options_base + " -DVISRTX_ENABLE_MDL_SUPPORT=OFF"

[doc("Configure")]
configure: (cmake::force_configure _source_dir _build_dir _install_dir _cmake_options "-DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=OFF")

[doc("Configure PhysicallyBased MDL")]
configure_physicallybased_mdl: (cmake::force_configure _source_dir _build_dir _install_dir _cmake_options "-DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=ON")

[doc("Configure no MDL")]
configure_mdl_off: (cmake::force_configure _source_dir _build_dir _install_dir _cmake_options_mdl_off "-DVISRTX_USE_MDL_FOR_PHYSICALLY_BASED=OFF")

[doc("Reconfigure")]
reconfigure: (cmake::configure _source_dir _build_dir _install_dir _cmake_options)

[doc("Build")]
[metadata("task", "pm=$gcc")]
build cfg=cfg: reconfigure (cmake::install cfg _build_dir)
    mkdir -p "{{ _install_dir }}/bin"
    cp -u "{{ _build_dir }}/{{ cfg }}/tsd"* "{{ _install_dir }}/bin"

[doc("Rebuild")]
[metadata("task", "pm=$gcc")]
rebuild cfg=cfg: clean (build cfg)

# Run recipes

[doc("Run a command")]
[private]
[script("bash", "-euo", "pipefail")]
run COMMAND *ARGS:
    exec {{ COMMAND }} {{ ARGS }}

[doc("Run tsdViewer")]
[metadata("launch", "debug")]
view *ARGS: (run "tsdViewer" "-v" "-e" ARGS)

[doc("Run tsdRender")]
[metadata("launch", "debug")]
render *ARGS: (run "tsdRender" ARGS)

# Auxiliary recipes

[doc("Generate .envrc file")]
generate_envrc: (dotfiles::generate_envrc _envrc_dir _install_dir _anari_envrc)

[doc("Generate .clangd config file")]
generate_clangd: (dotfiles::generate_clangd _clangd_dir _compile_commands_dir _compiler)

[doc("Generate .vscode/tasks.json")]
generate_vscode_tasks_json: (dotfiles::generate_vscode_tasks_json _vscode_dir)

[doc("Generate .vscode/launch.json")]
generate_vscode_launch_json: (dotfiles::generate_vscode_launch_json _vscode_dir)

[doc("Generate .emacs.desktop file")]
generate_emacs_desktop: (dotfiles::generate_emacs_desktop _emacs_desktop_dir)

[doc("Generate .emacs.desktop file")]
generate_emacs_dape: (dotfiles::generate_emacs_dape _emacs_dape_dir)

[doc("Remove the installation dir")]
clean:
    rm -fr "{{ _install_dir }}"
