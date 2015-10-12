using Images, TestImages, FixedSizeArrays, Interpolations
using RegisterDriver, RegisterWorkerRigid, RegisterWorkerApertures, RegisterDeformation
using Base.Test

### Apertured registration
# Create the data
img = testimage("cameraman")
gridsizeg = (4,4) # for generation
shift_amplitude = 10
u_dfm = shift_amplitude*randn(2, gridsizeg...)
knotsg = map(d->linspace(1,size(img,d),gridsizeg[d]), (1,2))
ϕ_dfm = GridDeformation(u_dfm, knotsg)
wimg = copyproperties(img, warp(img, ϕ_dfm))
o = 3*shift_amplitude
fixed = getindexim(img, o+1:size(img,1)-o, o+1:size(img,2)-o)
moving = getindexim(wimg, o+1:size(img,1)-o, o+1:size(img,2)-o)

# Set up the range of λ, and prepare for plotting
λrange = (1e-6,10)
nλ = round(Int, log2(nextpow2(round(Int, λrange[2]/λrange[1]))))
λnext = λrange[1]
λs = Float64[(λ = λnext; λnext *= 2; λ) for i = 1:nλ]
dp = zero(λs)

# Perform the registration
gridsize = (17,17)  # for correction
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1,2))
umax = maximum(abs(u_dfm))
maxshift = (ceil(Int, umax)+5, ceil(Int, umax)+5)
algorithm = RegisterWorkerApertures.Apertures(fixed, knots, maxshift, λrange)
mon = Dict{Symbol,Any}(:u => Array(Vec{2,Float64}, gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => dp,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
driver(algorithm, moving, mon)

# Analysis
ϕ = GridDeformation(mon[:u], knots)
gd0 = warpgrid(ϕ_dfm, showidentity=true)
ϕi = interpolate(ϕ_dfm)
gd1 = warpgrid(ϕi(interpolate(ϕ)), showidentity=true)

# Consider:
# ImagePlayer.view(gd0)
# ImagePlayer.view(gd1)
# showoverlay(fixed, moving)
# showoverlay(fixed, mon[:warped])
# plot(x=λs, y=mon[:datapenalty], xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

nothing
