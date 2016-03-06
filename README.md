# BlockRegistrationScheduler

This package implements various algorithms for image registration. The algorithms are encapsulated as "workers," and include the following:

- `RegisterWorkerRigid`: rigid registration (rotation+translation)
- `RegisterWorkerApertures`: deformable registration
- `RegisterWorkerAperturesMismatch`: deformable registration in two stages (this writes "mismatch data" to disk)

These workers are executed by the `driver` function found in
`RegisterDriver`, which schedules jobs for workers running in separate
processes.

## Usage

The prototypical usage is

```jl
using BlockRegistration, BlockRegistrationScheduler
# Use RegisterDriver and the Algorithm of our choice
using RegisterDriver, RegisterWorkerAlgorithm

# Do whatever preparative work you need to do

# Define the algorithm and which results we want passed back
algorithm = Algorithm[Algorithm(parameters...) for i = 1:nprocesses]
mon = monitor(algorithm, (:var1, :var2), Dict(:var3=>0, ...))

# Run the registration
mon = driver(outputfilename, algorithm, moving_image_sequence, mon)
```

The results are stored in the JLD file `outputfilename`. You can learn
more about what each of these functions does from the help (e.g.,
`?monitor`).

## Stack-by-stack registration example

```jl
wpids = addprocs(8)  # use 8 worker processes (no CUDA)

using Images, SIUnits.ShortUnits, FixedSizeArrays, JLD, MAT
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerApertures

# Here's the input file we'll be processing
fn = "exp1_20150814.imagine"

### Apertured registration
# Load the data
#   The mode="r" is needed if you don't have write permission to the
#   file, or don't want to risk accidents
img0 = load(fn, mode="r")
# Snip out a region of interest (discard empty portions of the image stack)
roi = (751:950, 821:1045, 12:28)
img = subim(img0, roi..., :)  # you could alternatively select a subset of times
# Select our "fixed" image
fixedidx = (nimages(img)+1) >> 1
fixed0 = sliceim(img, "t", fixedidx)

# Important: you should manually inspect fixed0 to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

# Make sure the pixelspacing property is set correctly; edit the .imagine
# file with a text editor if necessary.
ps = img["pixelspacing"]
# Define preprocessing. Here we'll bandpass filter between 2 pixels
# and 25μm, but these numbers are likely to be image-dependent.
sigmahp = Float64[25e-6m/x for x in ps]
sigmalp = [0,0,0]  # lowpass filtering is not currently recommended
pp = PreprocessSNF(100, sigmalp, sigmahp)
fixed = copy(pp(fixed0))  # use copy because of julia #14625

# Set up the grid of apertures for aligning the images. You should use
# a grid that is fine enough to capture the regional differences, but
# keep in mind that speed of processing is dramatically worsened by
# grids that are bigger than necessary. This will likely require some
# experimentation.
gridsize = (15,15,9)  # or you could use round(Int, Float64[50e-6m/x for x in ps]) for one aperture each 50μm
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))

# Define the "maxshift" needed for alignment, the largest number of
# pixels moved by any feature in the image along any axis. You should
# be able to determine this reasonably well by just looking at the
# images and zooming in on small regions, examining the movement over
# time.
mxshift = (15,15,3)

# Choose volume regularization penalty. See the README for
# BlockRegistration and `fixed_λ`.
λ = 0.01

## Set up aperture overlap (Optional)
# overlap_t = (10, 10, 2)   # Set the number of overlapping pixels in each dimension 
## Otherwise, aperture overlap ratio can be used instead of the number of overlapping pixels:
# using RegisterMismatch
# apertureoverlap = 0.2;  # Aperture overlap ratio (e.g. 0.2 (20%)); 0 generally works fine
# aperture_width = default_aperture_width(fixed, gridsize) # Obtain the default aperture width.
# overlap_t = map(x->round(Int64,x*apertureoverlap), aperture_width) 

# Create the worker algorithm structures. We assign one per worker process.
algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; overlap = overlap_t, pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u=>ArrayDecl(Array{Vec{3,Float64},3}, gridsize)))

# Define the output file and run the job
basename = splitext(splitdir(fn)[2])[1]
@show basename
fileout = string(basename, ".register")
@time driver(fileout, algorithm, img, mon)

# Append important extra information to the file
jldopen(fileout, "r+") do io
    write(io, "imagefile", fn)
    write(io, "roi", roi)
    write(io, "fixedidx", fixedidx)
    write(io, "knots", knots)
    write(io, "sigmalp", sigmalp)
    write(io, "sigmahp", sigmahp)
end
```

To warp the images:
```jl
# Read the image data and deformation
using Images, ImagineFormat, FileIO
u = load(fileout, "u")
roi = load(fileout, "roi")
knots = load(fileout, "knots")
img0 = load(fn, mode="r")
img = sliceim(img0, roi..., :)

# You might also use `tinterpolate` in the following call, if you processed
# a subset of time slices. (Be sure to save the particular slices to that
# JLD file!)
ϕs = medfilt(griddeformations(u, knots), 3)

# Write the warped image
basename = "exp1_20150814_register_tinyblock"
open(string(basename, ".cam"), "w") do file
    warp!(Float32, file, img, ϕs; nworkers=3)
end
ImagineFormat.save_header(string(basename, ".imagine"), fn, img, Float32)
```

## Whole-experiment optimization example

For calcium imaging data, an alternative worker is
`RegisterWorkerAperturesMismatch`. Here is a complete example of how
one might use it:

```jl
wpids = addprocs(3)   # launch 3 worker processes

using Images, SIUnits.ShortUnits, FixedSizeArrays, CUDArt, JLD
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerAperturesMismatch

# Here's the input file we'll be processing
fn = "/fish_raid/donghoon/dl_revision_005/20150818/exp8_20150818.imagine"

### Apertured registration
# Load the data
#   The mode="r" is needed if you don't have write permission to the
#   file, or don't want to risk accidents
img0 = load(fn, mode="r")
# Snip out a region of interest (discard empty portions of the image stack)
roi = (351:1120, :, :)
img = subim(img0, roi..., :)
# Select our "fixed" image
fixedidx = (nimages(img)+1) >> 1
fixed0 = getindexim(img, "t", fixedidx)

# Important: you should manually inspect fixed0 to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

# Make sure the pixelspacing property is set correctly; edit the .imagine
# file with a text editor if necessary.
ps = img["pixelspacing"]
# Define preprocessing. Here we'll bandpass filter between 2 pixels
# and 25μm, but these numbers are likely to be image-dependent.
sigmahp = Float64[25e-6m/x for x in ps]
sigmalp = [2,2,0]
pp = RegisterWorkerAperturesMismatch.PreprocessSNF(100, sigmalp, sigmahp)
fixed = pp(fixed0)

# Inspect fixed to see if it looks reasonable. It should be smooth
# (blurry) enough that images will "flow" into alignment, without
# getting trapped by pixel noise.  (When you later create the corrected
# images, this blurring will not be present---this is used only for
# the purpose of defining the deformation that best aligns the images.)
# But don't blur so much that you destroy features that help with alignment.

# Set up the grid of apertures for aligning the images. You should use
# a grid that is fine enough to capture the regional differences, but
# keep in mind that speed of processing is dramatically worsened by
# grids that are bigger than necessary. This will likely require some
# experimentation.
gridsize = (15,13,5)  # or you could use round(Int, Float64[50e-6m/x for x in ps]) for one aperture each 50μm
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))

# Define the "maxshift" needed for alignment, the largest number of
# pixels moved by any feature in the image along any axis. You should
# be able to determine this reasonably well by just looking at the
# images and zooming in on small regions, examining the movement over
# time.
# The bigger this is, the bigger the resulting mismatch file and the
# slower processing will be.  So choose something adequate but not
# excessive.  TODO: if necessary, perform registration in two steps: one
# with a big mxshift and coarse grid, and then a second registration
# with a smaller mxshift and finer grid.
mxshift = (15,15,3)

# Specify which GPUs we'll be using. See CUDArt.
devs = 0:2
# Create the worker algorithm structures. We assign one per worker
# process/GPU combination.
algorithm = AperturesMismatch[AperturesMismatch(fixed, knots, mxshift, pp; dev=devs[i],pid=wpids[i]) for i = 1:length(wpids)]
# We'll store the variables Es, cs, Qs, and mmis. See the help for
# AperturesMismatch.
mon = monitor(algorithm, (:Es, :cs, :Qs, :mmis))

# Wait for the GPUs to become free (if you don't do this, you might trash someone else's job!)
wait_free(devs)
# Define the output file and run the job
basename = splitext(splitdir(fn)[2])[1]
fileout = string(basename, ".mm")
@time driver(fileout, algorithm, img, mon)

# Shut down the worker processes
rmprocs(wpids)

# Append important extra information to the file
jldopen(fileout, "r+") do io
    write(io, "imagefile", fn)
    write(io, "roi", roi)
    write(io, "fixedidx", fixedidx)
    write(io, "knots", knots)
    write(io, "sigmalp", sigmalp)
    write(io, "sigmahp", sigmahp)
end
```

At the conclusion of this script's execution, `fileout` will have been
written. You use this as the input to the optimization phase of
registration; see the README for BlockRegistration.
