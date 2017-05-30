using Images, TestImages, StaticArrays, Interpolations
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
wimg = warp(img, ϕ_dfm)
o = 3*shift_amplitude
fixed = img[o+1:size(img,1)-o, o+1:size(img,2)-o]
moving = wimg[o+1:size(img,1)-o, o+1:size(img,2)-o]

# Set up the range of λ, and prepare for plotting
λrange = (1e-6,10)

# To make sure it runs, try the example in the docs, even though it's
# not well-tuned for this case
pp = img -> imfilter(img, KernelFactors.IIRGaussian((3,3)))
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
mon = Dict{Symbol,Any}(:u => Array{SVector{2,Float64}}(gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => 0,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
mon = driver(algorithm, moving, mon)

# With aperture overlap
using RegisterMismatch
apertureoverlap = 0.3;  #Aperture overlap percentage (between 0 and 1)
aperture_width = default_aperture_width(fixed, gridsize)
overlap_t = map(x->round(Int64,x*apertureoverlap), aperture_width)
algorithm = RegisterWorkerApertures.Apertures(fixed, knots, maxshift, λrange; overlap=overlap_t)
mon_overlap = Dict{Symbol,Any}(:u => Array{SVector{2,Float64}}(gridsize),
                       :mismatch => 0.0,
                       :λ => 0.0,
                       :datapenalty => 0,
                       :sigmoid_quality => 0.0,
                       :warped => copy(moving))
mon_overlap = driver(algorithm, moving, mon_overlap)



# Analysis
ϕ = GridDeformation(mon[:u], knots)
ϕ_overlap = GridDeformation(mon_overlap[:u], knots)

gd0 = warpgrid(ϕ_dfm, showidentity=true)
ϕi = interpolate(ϕ_dfm)

gd1 = warpgrid(ϕi(interpolate(ϕ)), showidentity=true)
gd1_overlap = warpgrid(ϕi(interpolate(ϕ_overlap)), showidentity=true)

using RegisterMismatch, RegisterCore
r0 = ratio(mismatch0(fixed, moving), 0)
r1 = ratio(mismatch0(fixed, mon[:warped]), 0)
r1_overlap = ratio(mismatch0(fixed, mon_overlap[:warped]), 0)
@test r1 < r0
@test r1_overlap < r0

# Consider:
# using RegisterGUI
# ImagePlayer.view(gd0)
# ImagePlayer.view(gd1)
# showoverlay(fixed, moving)
# showoverlay(fixed, mon[:warped])
#
# plot(x=λs, y=mon[:datapenalty], xintercept=[mon[:λ]], Geom.point, Geom.vline, Guide.xlabel("λ"), Guide.ylabel("Data penalty"), Scale.x_log10)

nothing
