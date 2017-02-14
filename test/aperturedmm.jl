aperturedprocs = addprocs(2)
# aperturedprocs = [myid()]

using Images, TestImages, FixedSizeArrays
using BlockRegistration, RegisterDeformation, RegisterCore, RegisterOptimize
using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerAperturesMismatch

# Work around julia #3674
@sync for p in aperturedprocs
    @spawnat p eval(quote
        using Images, TestImages, FixedSizeArrays
        using BlockRegistration, RegisterDeformation
        using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerAperturesMismatch
    end)
end
using Base.Test

workdir = mktempdir()

### Apertured registration
# Create the data
fixed = testimage("cameraman")
gridsize = (5,5)
ntimes = 4
shift_amplitude = 5
u_dfm = shift_amplitude*randn(2, gridsize..., ntimes)
img = AxisArray(SharedArray(Float64, (size(fixed)..., ntimes), pids = union(myid(), aperturedprocs)), :y, :x, :time)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1,2))
tax = timeaxis(img)
for i = 1:ntimes
    ϕ_dfm = GridDeformation(u_dfm[:,:,:,i], knots)
    img[tax(i)] = warp(fixed, ϕ_dfm)
end
# Perform the registration
fn = joinpath(workdir, "apertured.jld")
maxshift = (3*shift_amplitude, 3*shift_amplitude)
algorithms = AperturesMismatch[AperturesMismatch(fixed, knots, maxshift; pid=p) for p in aperturedprocs]
mons = monitor(algorithms, (:Es, :cs, :Qs, :mmis))
driver(fn, algorithms, img, mons)

# With preprocessing
fn_pp = joinpath(workdir, "apertured_pp.jld")
pp = PreprocessSNF(0.1, [2,2], [10,10])
algorithms = AperturesMismatch[AperturesMismatch(pp(fixed), knots, maxshift, pp; pid=p) for p in aperturedprocs]
mons = monitor(algorithms, (:Es, :cs, :Qs, :mmis))
driver(fn_pp, algorithms, img, mons)


rmprocs(aperturedprocs, waitfor=1.0)

using JLD, RegisterMismatch

for file in (jldopen(fn, "r"), jldopen(fn_pp, "r"))
    dEs, dcs, dQs, dmmis = file["Es"], file["cs"], file["Qs"], file["mmis"]
    for d in (dEs, dcs, dQs, dmmis)
        @test eltype(d) == Float32
    end
    @test size(dEs) == (gridsize..., ntimes)
    @test size(dcs) == (2, gridsize..., ntimes)
    @test size(dQs) == (2, 2, gridsize..., ntimes)
    innersize = map(x->2x+1, maxshift)
    @test size(dmmis) == (2, innersize..., gridsize..., ntimes)
    close(file)
end

cs, Qs, mmis = jldopen(fn, mmaparrays=true) do file
    read(file, "cs"), read(file, "Qs"), read(file, "mmis")
end
ϕs, mismatch = fixed_λ(cs, Qs, knots, AffinePenalty(knots, 0.001), 1e-5, mmis)
for t = 1:nimages(img)
    moving = img[tax(t)]
    warped = warp(moving, ϕs[t])
    r_m = ratio(mismatch0(fixed, moving), 0)
    r_w = ratio(mismatch0(fixed, warped), 0)
    @test r_w < r_m
end

cs = Qs = mmis = 0   # since we're mmapping, we'd better free these before deleting files
gc()

rm(workdir, recursive=true)
