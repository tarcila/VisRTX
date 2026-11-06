-- Render representative animation samples through the configured ANARI device.
-- Usage: vsrLua render_animation.lua sequence-directory [samples] [image-size]
assert(#arg >= 1 and #arg <= 3, "expected sequence-directory [samples] [image-size]")
local directory = arg[1]
local samples = math.tointeger(tonumber(arg[2] or "32"))
local size = math.tointeger(tonumber(arg[3] or "512"))
assert(samples and samples > 0 and size and size > 0)
vsr.io.loadSceneArchive(scene, directory .. "/scene.vsr")
vsr.io.loadAnimationManagerArchive(animationMgr, directory .. "/animations.vsr")
local device = vsr.render.loadDevice("environment")
local index = vsr.render.createRenderIndex(scene, device)
index:populate()
local camera = vsr.CameraSetup.new()
camera.position = vsr.float3(3.4, -5., 2.7)
camera.direction = vsr.normalize(vsr.float3(0., 0., 1.)-camera.position)
camera.up = vsr.float3(0., 0., 1.)
camera.fovy = 40.
camera.aspect = 1.
for _, phase in ipairs({0., .25, .5, .75, 1.}) do
    animationMgr:setAnimationTime(phase)
    -- New renderer resets accumulation for each field change.
    local renderer = vsr.render.createRenderer(size, size, device, index, camera,
        {renderer="quality", background=vsr.float4(0., 0., 0., 1.), denoise=false})
    renderer:renderToFile(samples, string.format("%s/preview_%03d.png", directory, phase*100))
end
-- Release render delegates before vsrLua destroys its global scene.
index = nil
collectgarbage("collect")
device = nil
collectgarbage("collect")
print("Rendered representative phases and loop closure (raw renderer exposure).")
