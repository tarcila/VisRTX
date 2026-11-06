-- Invoked by package_animation.py; scene and animationMgr belong to vsrLua.
assert(#arg == 2 or #arg == 3, "expected config.lua output-directory [validate]")
local cfg = dofile(arg[1])
local output = arg[2]
local function vec(v) return vsr.float3(v[1], v[2], v[3]) end
local function fieldName(name, i) return string.format("%s/%04d", name, i-1) end
local n = #cfg.frames

if arg[3] == "validate" then
    vsr.io.loadSceneArchive(scene, output .. "/scene.vsr")
    vsr.io.loadAnimationManagerArchive(animationMgr, output .. "/animations.vsr")
    assert(animationMgr:numberOfAnimations() == 1)
    assert(scene:numberOfObjects(vsr.VOLUME) == 1)
    assert(scene:numberOfObjects(vsr.SPATIAL_FIELD) == n * #cfg.names)
    assert(animationMgr:getAnimationTotalFrames() == n+1)
    assert(math.abs(animationMgr:getAnimationFPS() - n/cfg.duration) < 1e-5)
    local volume = scene:getVolume(0)
    for i = 1, n+1 do
        animationMgr:setAnimationTime((i-1)/n)
        local sample = i <= n and i or 1
        for key, name in pairs(cfg.parameters) do
            if key:sub(-6) == ".field" then
                local f = volume:getParameter(key)
                assert(f:valid() and f.name == fieldName(name, sample), key .. " out of sync")
                assert(f:getParameter("data"):size() > 0)
                assert(f:getParameter("dataCentering") == "cell")
                assert(f:getParameter("filter") == "linear")
                local roi = f:getParameter("roi")
                local domain = cfg.frames[sample].fields[name].domain
                for axis, component in ipairs({"x", "y", "z"}) do
                    assert(math.abs(roi.lower[component]-domain[1][axis]) < 1e-6)
                    assert(math.abs(roi.upper[component]-domain[2][axis]) < 1e-6)
                end
            end
        end
    end
    local report = assert(io.open(output .. "/animation.validation.txt", "w"))
    report:write("PASS: fresh-process scene/animation reload; every timestep and closure; synchronized field references, nonempty arrays, cell/filter/ROI, object count and playback clock.\n")
    report:write("Rendering is a separate check; voxel interpolation between frames is not supported.\n")
    report:close()
    print("PASS animated archive validation")
    return
end

local fields, times = {}, {}
for _, name in ipairs(cfg.names) do fields[name] = {} end
for i, frame in ipairs(cfg.frames) do
    times[i] = frame.phase
    for _, name in ipairs(cfg.names) do
        local info = frame.fields[name]
        local f = vsr.io.importNVDB(scene, info.file)
        assert(f:valid(), "failed to load " .. info.file)
        f.name = fieldName(name, i)
        f:setParameter("dataCentering", "cell")
        f:setParameter("filter", "linear")
        f:setParameter("roi", vsr.box3.new(vec(info.domain[1]), vec(info.domain[2])))
        fields[name][i] = f
    end
end
times[n+1] = 1.
for _, name in ipairs(cfg.names) do fields[name][n+1] = fields[name][1] end
local volume = scene:createVolume("principledVolume")
volume.name = "animated plume / " .. cfg.preset
volume:setParameter("bounds", vsr.box3.new(vec(cfg.bounds[1]), vec(cfg.bounds[2])))
local animation = animationMgr:addAnimation("plume")
assert(animation.addObjectParameterBinding, "Vela must support object-parameter bindings")
for key, value in pairs(cfg.parameters) do
    if key:sub(-6) == ".field" then
        volume:setParameter(key, fields[value][1])
        animation:addObjectParameterBinding(volume, key, "spatialField", fields[value], times, "step")
    elseif type(value) == "table" then
        volume:setParameter(key, vec(value))
    else
        volume:setParameter(key, value)
    end
end
scene:insertObjectNode(scene:defaultLayer():root(), volume, volume.name)
local camera = scene:getCamera(0)
camera:setParameter("position", vsr.float3(3.4, -5., 2.7))
camera:setParameter("direction", vsr.normalize(vsr.float3(0., 0., 1.)-vsr.float3(3.4, -5., 2.7)))
camera:setParameter("up", vsr.float3(0., 0., 1.))
camera:setParameter("fovy", vsr.radians(40.))
camera:setParameter("aspect", 1.)
camera:setMetadata("testbed.exposureStops", -6.)
animationMgr:setAnimationTotalFrames(n+1)
animationMgr:setAnimationFPS(n/cfg.duration)
animationMgr:setAnimationIncrement(1./n)
animationMgr:setAnimationTime(0.)
vsr.io.saveSceneArchive(scene, output .. "/scene.vsr")
vsr.io.saveAnimationManagerArchive(animationMgr, output .. "/animations.vsr")
