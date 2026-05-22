# BlockRegistrationScheduler.jl

[![CI](https://github.com/HolyLab/BlockRegistrationScheduler.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/BlockRegistrationScheduler.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/BlockRegistrationScheduler.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/BlockRegistrationScheduler.jl)

Convenience bundle that re-exports the block-registration scheduling stack. Loading this package gives you:

- [`RegisterDriver`](https://github.com/HolyLab/RegisterDriver.jl) — the `driver` function that schedules jobs across worker processes
- [`RegisterWorkerShell`](https://github.com/HolyLab/RegisterWorkerShell.jl) — base worker infrastructure
- [`RegisterWorkerApertures`](https://github.com/HolyLab/RegisterWorkerApertures.jl) — deformable (aperture-based) registration
- [`RegisterWorkerAperturesMismatch`](https://github.com/HolyLab/RegisterWorkerAperturesMismatch.jl) — two-stage deformable registration (writes mismatch data to disk for whole-experiment optimization)

You can use this package as a single import, or load the individual packages directly. For smaller images that don't require parallel workers, consider using [BlockRegistration](https://github.com/HolyLab/BlockRegistration) directly.

## Installation

This package is registered in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry). Add the registry once, then install normally:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("BlockRegistrationScheduler")
```

## Overview

These packages implement deformable image registration for large volumetric image sequences using a **worker–driver** architecture:

- **Workers** encapsulate a registration algorithm and run in separate Julia processes (launched with `addprocs`). Multiple workers speed up processing of large image sequences.
- The **`driver` function** schedules frames across workers and collects results into an output file.
- The **`monitor` function** specifies which intermediate variables each worker should return.

The prototypical usage is:

```julia
using BlockRegistration, BlockRegistrationScheduler

wpids = addprocs(4)

algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; pid=wpids[i]) for i in 1:length(wpids)]
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u => ArrayDecl(Array{SVector{3,Float64},3}, gridsize)))

driver(fileout, algorithm, img, mon)

rmprocs(wpids)
```

## Stack-by-stack registration example

> **Note:** the examples below were written against older package versions. Some file-loading and image-metadata APIs may differ in current versions of the individual packages. Treat them as a conceptual guide and consult the documentation of the individual packages for up-to-date APIs.

```julia
wpids = addprocs(8)

using Images, Unitful, StaticArrays, JLD2, AxisArrays
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerApertures

const μm = u"μm"

fn = "exp1_20150814.imagine"
img0 = load(fn, mode="r")

# Optionally crop to a region of interest
roi = (751:950, 821:1045, 12:28)
img = view(img0, roi..., :)

# Choose the middle frame as the fixed (reference) image
fixedidx = (nimages(img) + 1) ÷ 2
fixed0 = view(img, timeaxis(img)(fixedidx))

# Preprocessing: highpass filter to suppress background
ps = pixelspacing(img)
σ = 25μm
sigmahp = Float64[σ/x for x in ps]
sigmalp = [0, 0, 0]
bias = reinterpret(eltype(img0), UInt16(100))
pp = PreprocessSNF(bias, sigmalp, sigmahp)
fixed = pp(fixed0)

# Set up the aperture grid
gridsize = (15, 15, 9)
knots = map(d -> LinRange(1, size(fixed, d), gridsize[d]), (1:ndims(fixed)...,))
mxshift = (15, 15, 3)
λ = 0.01

algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; pid=wpids[i], correctbias=false) for i in 1:length(wpids)]
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u => ArrayDecl(Array{SVector{3,Float64},3}, gridsize)))

fileout = string(splitext(splitdir(fn)[2])[1], ".register")
@time driver(fileout, algorithm, img, mon)

jldopen(fileout, "r+") do io
    io["imagefile"] = fn
    io["roi"]       = roi
    io["fixedidx"]  = fixedidx
    io["knots"]     = knots
    io["sigmalp"]   = sigmalp
    io["sigmahp"]   = sigmahp
end
```

To warp the registered images:

```julia
using Images, ImagineFormat, FileIO, BlockRegistration

u     = jldopen(io -> io["u"],      fileout)
roi   = jldopen(io -> io["roi"],    fileout)
knots = jldopen(io -> io["knots"],  fileout)
img0  = load(fn, mode="r")
img   = view(img0, roi..., :)

ϕs = medfilt(griddeformations(u, knots), 3)

basename = "exp1_20150814_register"
open(string(basename, ".cam"), "w") do file
    warp!(Float32, file, img, ϕs; nworkers=3)
end
ImagineFormat.save_header(string(basename, ".imagine"), fn, img, Float32)
```

## Whole-experiment optimization example

`RegisterWorkerAperturesMismatch` runs registration in two stages: it first writes mismatch data to disk, then a separate optimization pass finds the global deformation. This is useful for calcium imaging where you want to optimize across the entire time series.

```julia
wpids = addprocs(3)

using Images, Unitful, StaticArrays, CUDA, JLD2, AxisArrays
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerAperturesMismatch

const μm = u"μm"

fn = "exp8_20150818.imagine"
img0 = load(fn, mode="r")
roi = (351:1120, :, :)
img = view(img0, roi..., :)

fixedidx = (nimages(img) + 1) ÷ 2
fixed0 = img[timeaxis(img)(fixedidx)]

ps = pixelspacing(img)
sigmahp = Float64[25μm/x for x in ps]
sigmalp = [0, 0, 0]
bias = reinterpret(eltype(img0), UInt16(100))
pp = PreprocessSNF(bias, sigmalp, sigmahp)
fixed = pp(fixed0)

gridsize = (15, 13, 5)
knots = map(d -> LinRange(1, size(fixed, d), gridsize[d]), (1:ndims(fixed)...,))
mxshift = (15, 15, 3)

devs = 0:2
algorithm = AperturesMismatch[AperturesMismatch(fixed, knots, mxshift, pp; dev=devs[i], pid=wpids[i]) for i in 1:length(wpids)]
mon = monitor(algorithm, (:Es, :cs, :Qs, :mmis))

fileout = string(splitext(splitdir(fn)[2])[1], ".mm")
@time driver(fileout, algorithm, img, mon)

rmprocs(wpids)

jldopen(fileout, "r+") do io
    io["imagefile"] = fn
    io["roi"]       = roi
    io["fixedidx"]  = fixedidx
    io["knots"]     = knots
    io["sigmalp"]   = sigmalp
    io["sigmahp"]   = sigmahp
end
```

The resulting `.mm` file is used as input to the optimization phase; see the [BlockRegistration README](https://github.com/HolyLab/BlockRegistration) for the next steps.

## Registration tips

The following tips come from practical experience registering large volumetric calcium-imaging data.

**Image selection and preprocessing**

1. **Choose images without evoked activity** — select time points free of stimulation artifacts before registering. Consider temporal median filtering to reduce noise and spontaneous activity.
2. **Clip high-intensity pixels** — moving bright objects (e.g., surface debris) can derail registration. Replace them before registering: `img[img .> thresh] .= NaN`.

**Starting parameters** — tune on a small subset of frames before committing to a full run. Example values for a 1120×1080×60 voxel volume with 0.577×0.577×5 µm voxels:

```julia
bias     = 100
sigmahp  = Float64[20μm / x for x in ps]   # highpass over ~20 µm
sigmalp  = [3, 3, 0]
mxshift  = (30, 30, 3)                      # ~17×17×15 µm maximum shift
gridsize = (24, 24, 13)
λ        = 1e-4
```

Once parameters are optimized on a subset, use `tinterpolate` to propagate deformations across all time points before warping the full dataset.
