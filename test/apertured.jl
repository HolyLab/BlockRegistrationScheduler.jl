aperturedprocs = addprocs(2)
# aperturedprocs = [myid()]

using Images, TestImages, StaticArrays
using BlockRegistration, RegisterDeformation, RegisterCore
using BlockRegistrationScheduler, RegisterDriver, RegisterWorkerApertures

# Work around julia #3674
@sync for p in aperturedprocs
    @spawnat p eval(quote
        using Images, TestImages, StaticArrays
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
img = AxisArray(SharedArray(Float64, (size(fixed)..., 4), pids = union(myid(), aperturedprocs)), :y, :x, :time)
tax = timeaxis(img)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1,2))
for i = 1:4
    ϕ_dfm = GridDeformation(u_dfm[:,:,:,i], knots)
    img[tax(i)] = warp(fixed, ϕ_dfm)
end
# Perform the registration
fn = joinpath(tempdir(), "apertured.jld")
maxshift = (3*shift_amplitude, 3*shift_amplitude)
algorithms = Apertures[Apertures(fixed, knots, maxshift, 0.001; pid=p) for p in aperturedprocs]
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(gridsize),
                    :warped => Array{Float64}(size(fixed)),
                    :mismatch => 0.0))
driver(fn, algorithms, img, mons)

# With preprocessing
fn_pp = joinpath(tempdir(), "apertured_pp.jld")
pp = PreprocessSNF(0.1, [2,2], [10,10])
algorithms = Apertures[Apertures(pp(fixed), knots, maxshift, 0.001, pp; pid=p) for p in aperturedprocs]
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(gridsize),
                    :warped => Array{Float64}(size(fixed)),
                    :warped0 => Array{Float64}(size(fixed)),
                    :mismatch => 0.0))
driver(fn_pp, algorithms, img, mons)

# Again, this time with a constant shift (an easy case)
imgt = copy(img)
u_dfmt = copy(u_dfm)
for i = 1:4
    fill!(view(u_dfmt, 1, :, :, i), i)
    fill!(view(u_dfmt, 2, :, :, i), 5-i)
    ϕ_dfm = GridDeformation(u_dfmt[:,:,:,i], knots)
    imgt[tax(i)] = warp(fixed, ϕ_dfm)
end

fnt = joinpath(tempdir(), "apertured_translate.jld")
maxshift = (3*shift_amplitude, 3*shift_amplitude)
algorithms = Apertures[Apertures(fixed, knots, maxshift, 0.001; pid=p) for p in aperturedprocs]
mons = monitor(algorithms,
               (),
               Dict(:u => Array{SVector{2,Float64}}(gridsize),
                    :warped => Array{Float64}(size(fixed)),
                    :mismatch => 0.0))
driver(fnt, algorithms, imgt, mons)



rmprocs(aperturedprocs, waitfor=1.0)

using JLD, RegisterCore, RegisterMismatch

jldopen(fnt) do f
    mm = read(f["mismatch"])
    u = read(f["u"])
    @test maxabs(u+u_dfmt) < 0.5
    warped = read(f["warped"])
    for i = 1:nimages(img)
        r0 = ratio(mismatch0(fixed, imgt[tax(i)]),0)
        r1 = ratio(mismatch0(fixed, warped[:,:,i]), 0)
        @test r0 > r1
    end
end

nfailures = 0

jldopen(fn) do f
    global nfailures
    mm = read(f["mismatch"])
    @test all(mm .> 0)
    warped = read(f["warped"])
    for i = 1:nimages(img)
        r0 = ratio(mismatch0(fixed, img[tax(i)]),0)
        r1 = ratio(mismatch0(fixed, warped[:,:,i]), 0)
        nfailures += r0 <= r1
    end
end

jldopen(fn_pp) do f
    global nfailures
    mm = read(f["mismatch"])
    @test all(mm .> 0)
    warped = read(f["warped"])
    for i = 1:nimages(img)
        r0 = ratio(mismatch0(pp(fixed), pp(img[tax(i)])),0)
        r1 = ratio(mismatch0(pp(fixed), warped[:,:,i]), 0)
        nfailures += r0 <= r1
    end
    warped0 = read(f["warped0"])
    for i = 1:nimages(img)
        r0 = ratio(mismatch0(fixed, img[tax(i)]),0)
        r1 = ratio(mismatch0(fixed, warped0[:,:,i]), 0)
        nfailures += r0 <= r1
    end
end

@test nfailures <= 2
