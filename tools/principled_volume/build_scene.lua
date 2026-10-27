-- Usage: vsrLua build_scene.lua scene.lua output.vsr
-- Scene.lua is trusted generated Lua configuration, NOT an untrusted data file.
assert(arg and #arg == 2, "usage: vsrLua build_scene.lua scene.lua output.vsr")
local cfg = dofile(arg[1])
assert(cfg.schema_version == 2, "regenerate schema-2 full-cell scene data; historical archives are not rewritten")
local s = vsr.Scene.new()
local root = s:defaultLayer():root()
local function vec(v) return vsr.float3(v[1], v[2], v[3]) end
local function close(a, b) assert(math.abs(a-b) < 2e-6 * math.max(1, math.abs(b))) end
local fields = {}
for name, info in pairs(cfg.fields) do
    local field = vsr.io.importNVDB(s, info.file)
    assert(field:valid(), "failed to load " .. info.file)
    assert(info.data_centering == "cell" and info.filter == "linear", "expected full-cell metadata")
    field:setParameter("dataCentering", info.data_centering)
    field:setParameter("filter", info.filter)
    field:setParameter("roi", vsr.box3.new(vec(info.domain[1]), vec(info.domain[2])))
    field.name = name
    fields[name] = field
end
local volume = s:createVolume("principledVolume")
volume.name = cfg.scene .. "/" .. cfg.preset
volume:setParameter("bounds", vsr.box3.new(vec(cfg.bounds[1]), vec(cfg.bounds[2])))
for key, value in pairs(cfg.parameters) do
    if key:sub(-6) == ".field" then
        volume:setParameter(key, assert(fields[value]))
    elseif type(value) == "table" then
        volume:setParameter(key, vec(value))
    else
        assert(type(value) ~= "number" or math.type(value) == "float", "optical scalar must be float")
        volume:setParameter(key, value)
    end
end
volume:setMetadata("testbed.status", cfg.status)
volume:setMetadata("testbed.sourceManifest", cfg.source_manifest)
s:insertObjectNode(root, volume, volume.name)
for _, mesh in ipairs(cfg.meshes) do
    local geometry = s:createGeometry("triangle")
    geometry:setParameterArray("vertex.position", "float3", #mesh.vertices, mesh.vertices)
    local indices = {}
    for i, triangle in ipairs(mesh.triangles) do
        indices[i] = {math.tointeger(triangle[1]), math.tointeger(triangle[2]), math.tointeger(triangle[3])}
    end
    geometry:setParameterArray("primitive.index", "uint3", #indices, indices)
    local material = s:createMaterial("matte", {color=vec(mesh.color)})
    local surface = s:createSurface(mesh.name, geometry, material)
    s:insertObjectNode(root, surface, mesh.name)
end
local env = cfg.environment
if env.scale > 0 then
    if env.file ~= "" then
        -- Current Vela importer requires an AnimationManager argument.
        vsr.io.importHDRI(s, animationMgr, env.file)
        assert(s:numberOfObjects(vsr.LIGHT) == 1, "HDRI import failed")
    else
        local light = s:createLight("hdri")
        light:setParameterArray("radiance", "float3", 1, 1, {{1.,1.,1.}})
        s:insertObjectNode(root, light, "constant environment")
    end
    s:forEachLight(function(light)
        light:setParameter("direction", vec(env.direction))
        light:setParameter("up", vec(env.up))
        light:setParameter("scale", env.scale)
        light:setParameter("visible", true)
    end)
end
-- New VSR scenes already own a default perspective camera.
local camera = s:getCamera(0)
assert(camera:subtype() == "perspective")
camera.name = "testbed_camera"
camera:setParameter("position", vec(cfg.camera.position))
camera:setParameter("direction", vsr.normalize(vec(cfg.camera.target)-vec(cfg.camera.position)))
camera:setParameter("up", vec(cfg.camera.up))
camera:setParameter("fovy", vsr.radians(cfg.camera.fovy_degrees))
camera:setParameter("aspect", cfg.render.width/cfg.render.height)
camera:setMetadata("testbed.exposureStops", cfg.render.exposure)
camera:setMetadata("testbed.renderer", cfg.render.renderer)
vsr.io.saveSceneArchive(s, arg[2])

-- Round-trip validation does not instantiate an ANARI device or render anything.
local loaded = vsr.Scene.new()
vsr.io.loadSceneArchive(loaded, arg[2])
assert(loaded:numberOfObjects(vsr.VOLUME) == 1)
assert(loaded:numberOfObjects(vsr.SURFACE) == #cfg.meshes)
assert(loaded:numberOfObjects(vsr.LIGHT) == (env.scale > 0 and 1 or 0))
local restored = loaded:getVolume(0)
assert(restored:valid() and restored:subtype() == "principledVolume")
local bounds = restored:getParameter("bounds")
for i, component in ipairs({"x", "y", "z"}) do
    close(bounds.lower[component], cfg.bounds[1][i])
    close(bounds.upper[component], cfg.bounds[2][i])
end
for key, expected in pairs(cfg.parameters) do
    local value = restored:getParameter(key)
    if key:sub(-6) == ".field" then
        assert(value:valid() and value:subtype() == "nanovdb")
        assert(value:getParameter("data"):size() == fields[expected]:getParameter("data"):size())
        assert(value.name == expected)
        local info = cfg.fields[expected]
        assert(value:getParameter("dataCentering") == info.data_centering)
        assert(value:getParameter("filter") == info.filter)
        assert(value:parameter("dataCentering"):isEnabled())
        assert(value:parameter("filter"):isEnabled())
        assert(value:parameter("roi"):isEnabled())
        local roi = value:getParameter("roi")
        for i, component in ipairs({"x", "y", "z"}) do
            close(roi.lower[component], info.domain[1][i])
            close(roi.upper[component], info.domain[2][i])
        end
    elseif type(expected) == "table" then
        close(value.x, expected[1]); close(value.y, expected[2]); close(value.z, expected[3])
    else
        assert(math.type(value) == "float", "wrong optical scalar type: " .. key)
        close(value, expected)
    end
end
assert(loaded:numberOfObjects(vsr.CAMERA) == 1)
close(loaded:getCamera(0):getParameter("fovy"), vsr.radians(cfg.camera.fovy_degrees))
assert(not loaded:defaultLayer():empty())
local report = assert(io.open(arg[2] .. ".validation.txt", "w"))
report:write("PASS: archive save/reload; typed bounds, float optical scalars, field references and byte counts, cell centering/filter/ROI, object counts, camera FOV, nonempty scene graph.\n")
report:write("NOT TESTED: VisRTX rendering; backend principled transport is not yet supported.\n")
report:close()
print("PASS archive validation: " .. arg[2])
