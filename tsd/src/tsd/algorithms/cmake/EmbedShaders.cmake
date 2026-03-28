# EmbedShaders.cmake — concatenates math headers + .metal shader files into
# a single C++ header containing MSL source for runtime compilation.
#
# Input variables:
#   SHADER_MANIFEST — path to a file listing source paths, one per line
#                     (math headers first in dependency order, then .metal files)
#   OUTPUT_FILE     — path to the generated C++ header

file(STRINGS "${SHADER_MANIFEST}" shader_files)

set(concatenated "")

# metal_stdlib included once at top
string(APPEND concatenated "#include <metal_stdlib>\nusing namespace metal;\n\n")

foreach(file ${shader_files})
  file(READ "${file}" content)
  # Strip #pragma once (no filesystem context in runtime compilation)
  string(REGEX REPLACE "#pragma once[^\n]*\n" "" content "${content}")
  # Strip local #include "..." (content is already inlined in order)
  string(REGEX REPLACE "#include \"[^\"]*\"[^\n]*\n" "" content "${content}")
  # Strip #include <metal_stdlib> / using namespace metal (already at top)
  string(REGEX REPLACE "#include <metal_stdlib>[^\n]*\n" "" content "${content}")
  string(REGEX REPLACE "using namespace metal;[^\n]*\n" "" content "${content}")
  string(APPEND concatenated "${content}\n")
endforeach()

file(WRITE "${OUTPUT_FILE}"
  "#pragma once\n"
  "#include <string_view>\n"
  "namespace tsd::algorithms::metal {\n"
  "inline constexpr std::string_view kShaderSource = R\"(\n"
  "${concatenated}"
  ")\";\n"
  "} // namespace tsd::algorithms::metal\n"
)
