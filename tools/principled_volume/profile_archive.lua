-- Read-only scene measurement. Use an extracted scene archive for viewer-state
-- files; never save the input. CameraSetup uses degrees, ANARI camera uses radians.
-- vsrLua profile_archive.lua scene.vsr output renderer size launches spp [mode]
--   [seed] [controlled|viewer] [volumeRate] [shadowRate] [reference|fast]
local archive,out,subtype=assert(arg[1]),assert(arg[2]),assert(arg[3])
local size=math.tointeger(tonumber(arg[4] or '640'))
local launches=math.tointeger(tonumber(arg[5] or '20'))
local spp=math.tointeger(tonumber(arg[6] or '1'))
local mode=arg[7] or 'static'
local seed=math.tointeger(tonumber(arg[8] or '0'))
local s=vsr.Scene.new();vsr.io.loadSceneArchive(s,archive)
local device=vsr.render.loadDevice('visrtx_principle_volume')
local index=vsr.render.createRenderIndex(s,device);index:populate()
local c=s:getCamera(0);local camera=vsr.CameraSetup.new()
camera.position=c:getParameter('position');camera.direction=c:getParameter('direction')
camera.up=c:getParameter('up');camera.fovy=math.deg(c:getParameter('fovy'));camera.aspect=1.;camera.aperture=0.
print('CAMERA',tostring(camera.position),tostring(camera.direction),camera.fovy)
local profile=arg[9] or 'controlled'
local params={renderer=subtype,pixelSamples=spp,sampleLimit=0,maxRayDepth=32,randomSeed=seed,
 denoise=false,checkerboarding=false,fireflyFilterMode='none',ambientRadiance=0.,ambientSamples=0,
 volumeSamplingRate=1.,volumeSamplingRateShadows=1.,background=vsr.float4(0,0,0,0)}
if profile=='viewer' then
 -- Recorded settings from test-principled-volume.vsr; not generic archive discovery.
 if subtype=='quality' then
  params.denoise=true;params.checkerboarding=true;params.maxRayDepth=12;params.sampleLimit=4096
  params.background=vsr.float4(0,0,0,1)
 else
  params.ambientRadiance=.25;params.ambientSamples=1;params.fireflyFilterMode='tonemap';params.sampleLimit=128
  params.background=vsr.float4(.05,.05,.05,1)
 end
elseif profile~='controlled' then error('unknown profile')end
local volumeRate=tonumber(arg[10] or '1')
local shadowRate=tonumber(arg[11] or '1')
assert(volumeRate and volumeRate>=.001 and volumeRate<math.huge, 'invalid volume rate')
assert(shadowRate and shadowRate>=.0001 and shadowRate<math.huge, 'invalid shadow rate')
-- Vela preserves Lua integer/float types when setting ANARI parameters.
params.volumeSamplingRate=volumeRate+0.0;params.volumeSamplingRateShadows=shadowRate+0.0
local fieldSampling=arg[12] or 'reference'
assert(fieldSampling=='reference' or fieldSampling=='fast', 'invalid Quality field sampling')
if fieldSampling=='fast' then
 assert(subtype=='quality', 'fast field sampling requires Quality')
 params.volumeFastSampling=true
end
print('PROFILE',profile,'VOLUME_RATES',volumeRate,shadowRate,'FIELD_SAMPLING',fieldSampling)
local renderer=vsr.render.createRenderer(size,size,device,index,camera,params)
local volumes={};s:forEachVolume(function(v)table.insert(volumes,{object=v,density=v:getParameter('density'),temperature=v:getParameter('temperature')})end)
if mode=='volume-off' then
 for _,v in ipairs(volumes)do v.object:setParameter('visible',false)end
elseif mode=='source-off' then
 for _,v in ipairs(volumes)do v.object:setParameter('blackbodyIntensity',0.0);v.object:setParameter('emission',vsr.float3(0,0,0))end
end
for i=1,launches do
 local delta=.1*math.sin(.2*(i-1))
 if mode=='density' or mode=='temperature' then
  for _,v in ipairs(volumes)do v.object:setParameter(mode,v[mode]*(1.+delta))end
 elseif mode~='static' and mode~='volume-off' and mode~='source-off' then error('unsupported motion '..mode)end
 renderer:renderToFile(1,out..'/frame-'..i..'.png')
end
for line in io.lines('/proc/self/maps')do if line:match('libanari_library_visrtx_principle_volume')then print('LIBRARY',line)end end
renderer=nil;index=nil;collectgarbage('collect');device=nil;collectgarbage('collect')
