aperturedprocs = addprocs(2)
# aperturedprocs = [myid()]

using Images, TestImages, FixedSizeArrays
using BlockRegistration, RegisterDeformation
using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerApertures

# Work around julia #3674
@sync for p in aperturedprocs
    @spawnat p eval(quote
        using Images, TestImages, FixedSizeArrays
        using BlockRegistration, RegisterDeformation
        using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerApertures
    end)
end
using Base.Test

### Apertured registration
# Create the data
fixed = testimage("cameraman")
gridsize = (5,5)
shift_amplitude = 5
u_dfm = shift_amplitude*randn(2, gridsize..., 4)
img = copyproperties(fixed, SharedArray(Float64, (size(fixed)..., 4), pids = union(myid(), aperturedprocs)))
img["timedim"] = 3
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1,2))
for i = 1:4
    ϕ_dfm = GridDeformation(u_dfm[:,:,:,i], knots)
    img["t", i] = warp(fixed, ϕ_dfm)
end
# Perform the registration
fn = joinpath(tempdir(), "apertured.jld")
maxshift = (3*shift_amplitude, 3*shift_amplitude)
algorithms = [Apertures(fixed, knots, maxshift, 0.001; pid=p) for p in aperturedprocs]
mons = [Dict{Symbol,Any}(:u => SharedArray(Vec{2,Float64}, gridsize, pids=[myid(),p]), :warped => SharedArray(Float64, size(fixed), pids=[1,p]), :mismatch => 0.0) for p in aperturedprocs]
driver(fn, algorithms, img, mons)
# driver(algorithms[1], getindexim(img, "t", 1), mons[1])

rmprocs(aperturedprocs)
