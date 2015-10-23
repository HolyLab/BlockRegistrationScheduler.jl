using Images, TestImages, FixedSizeArrays, Interpolations
using BlockRegistration, BlockRegistrationScheduler
using RegisterDriver, RegisterWorkerApertures, RegisterDeformation
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

# To make sure it runs, try the example in the docs, even though it's
# not well-tuned for this case
pp = img -> imfilter_gaussian(img, [3, 3])
knots = (linspace(1, size(fixed,1), 5), linspace(1, size(fixed,2), 7))
fixedfilt = pp(fixed)
maxshift = (30,30)
alg = Apertures(fixedfilt, knots, maxshift, λrange, pp)
mon = monitor(alg, (), Dict(:λs=>0, :datapenalty=>0, :λ=>0, :u=>0, :warped0 => Array(Float64, size(fixed))))
mon = driver(alg, moving, mon)
datapenalty = mon[:datapenalty]
@test !all(mon[:warped0] .== 0)
# plot(x=λs, y=datapenalty, xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

# Perform the registration
gridsize = (17,17)  # for correction
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1,2))
umax = maximum(abs(u_dfm))
maxshift = (ceil(Int, umax)+5, ceil(Int, umax)+5)
algorithm = RegisterWorkerApertures.Apertures(fixed, knots, maxshift, λrange)
mon = Dict{Symbol,Any}(:u => Array(Vec{2,Float64}, gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => 0,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
mon = driver(algorithm, moving, mon)

# Analysis
ϕ = GridDeformation(mon[:u], knots)
gd0 = warpgrid(ϕ_dfm, showidentity=true)
ϕi = interpolate(ϕ_dfm)
gd1 = warpgrid(ϕi(interpolate(ϕ)), showidentity=true)

using RegisterMismatch, RegisterCore
r0 = ratio(mismatch0(fixed, moving), 0)
r1 = ratio(mismatch0(fixed, mon[:warped]), 0)
@test r1 < r0

# Consider:
# ImagePlayer.view(gd0)
# ImagePlayer.view(gd1)
# showoverlay(fixed, moving)
# showoverlay(fixed, mon[:warped])
# plot(x=λs, y=mon[:datapenalty], xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

nothing
