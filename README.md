# BlockRegistrationScheduler

This package implements various algorithms for image registration. The algorithms are encapsulated as "workers," and include the following:

- `RegisterWorkerRigid`: rigid registration (rotation+translation)
- `RegisterWorkerApertures`: deformable registration
- `RegisterWorkerAperturesMismatch`: deformable registration in two stages (this writes "mismatch data" to disk)

These workers are executed by the `driver` function found in
`RegisterDriver`, which schedules jobs for workers running in separate
processes. Having multiple workers can speed processing of large
images.

If your images are not large, you may find it easier to use
[BlockRegistration](https://github.com/HolyLab/BlockRegistration) directly.


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
# If your images are big (tens/hundreds of gigabytes or more), start up
# some worker processes to speed things up
wpids = addprocs(8)  # use 8 worker processes (no CUDA)
# Alternatively if your images aren't big, it's much easier to use wpids = [1]
# because this runs everything in the main process, and any error messages
# are easier to interpret.

using Images, Unitful, StaticArrays, JLD, MAT, AxisArrays
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerApertures

# Some physical units we'll need
const μm = u"μm"  # micrometers
const s  = u"s"   # seconds

# Here's the input file we'll be processing
fn = "exp1_20150814.imagine"

### Apertured registration
# Load the data
#   The mode="r" is needed if you don't have write permission to the
#   file, or don't want to risk accidents
img0 = load(fn, mode="r")
# Note: if you're loading from a file type that doesn't return an AxisArray,
# add something like this:
#    img0 = AxisArray(img0, (:y, :x, :time), (Δy, Δx, Δt))  # for a 2d image + time
# where Δy, Δx is the pixel spacing along y and x, respectively, and
# Δt the time between successive frames. (The latter isn't really used for anything.)

# Optionally snip out a region of interest (discard empty portions of the image stack).
# If you instead want to use the whole image, set img=img0 and skip the next two lines
roi = (751:950, 821:1045, 12:28)
img = view(img0, roi..., :)  # you could alternatively select a subset of times
# Pick one particular image to serve as the reference image.
# fixedidx records the index of the reference (fixed) image
fixedidx = (nimages(img)+1) ÷ 2  # ÷ can be obtained with "\div[TAB]"
# Select our "fixed" image
fixed0 = view(img, timeaxis(img)(fixedidx))

# Important: you should manually inspect fixed0 to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

## This next block is necessary only if you want highpass filtering,
## e.g., to get rid of some background. If you don't need this, set
##   fixed = fixed0
## and skip ahead to the part setting the grid.

  # Make sure the pixelspacing property is set correctly; edit the .imagine
  # file with a text editor if necessary or use the AxisArray as above.
  ps = pixelspacing(img)
  # Define preprocessing. Here we'll highpass filter over 25μm, but these
  # numbers are likely to be image-dependent.
  σ = 25μm
  sigmahp = Float64[σ/x for x in ps]
  sigmalp = [0,0,0]  # lowpass filtering is not currently recommended
  # The pco cameras have a bias of 100 in "digital number"
  # units. Convert this into the units of the image intensity.
  # If you're using a different camera, or using a PMT, this won't apply to you.
  # If you don't know anything better, you can set
  #     bias = zero(eltype(img0))
  bias = reinterpret(eltype(img0), UInt16(100))
  pp = PreprocessSNF(bias, sigmalp, sigmahp)
  fixed = pp(fixed0)
  # End of highpass filtering block

# Set up the grid of apertures for aligning the images. You should use
# a grid that is fine enough to capture the regional differences, but
# keep in mind that speed of processing is dramatically worsened by
# grids that are bigger than necessary. This will likely require some
# experimentation.
gridsize = (15,15,9)  # or you could use round(Int, Float64[50μm/x for x in ps]) for one aperture each 50μm
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

# Create the worker algorithm structures. We assign one per worker process.
algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
## Set up aperture overlap (Optional)
# overlap_t = (10, 10, 2)   # Set the number of overlapping pixels in each dimension
# algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; overlap = overlap_t, pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
## Aperture overlap ratio can be used instead of the number of overlapping pixels:
# using RegisterMismatch
# apertureoverlap = 0.2;  # Aperture overlap ratio (e.g. 0.2 (20%)); 0 generally works fine
# aperture_width = default_aperture_width(fixed, gridsize) # Obtain the default aperture width.
# overlap_t = map(x->round(Int64,x*apertureoverlap), aperture_width)
# `u` is an array of displacements, which we encode as `SVector`s from the StaticArrays package.
# Since this example is 3-dimensional, these displacements are `SVector{3,T}` where `T` is a number type like Float64.
# Moreover, `u` is a 3d array of these displacements, because the grid is a 3d grid. So if `V = SVector{3,Float64}` is
# the type of our displacements, then `u` is an `Array{V,3}`.
# If you're working in 2d, then change these 3s to 2s.
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u=>ArrayDecl(Array{SVector{3,Float64},3}, gridsize)))

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

Once this is done, to warp the images:
```jl
# Read the image data and deformation
using Images, ImagineFormat, FileIO
u = load(fileout, "u")
roi = load(fileout, "roi")
knots = load(fileout, "knots")
img0 = load(fn, mode="r")
img = view(img0, roi..., :)

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
`RegisterWorkerAperturesMismatch`. Here is a fairly complete example of how
one might use it (but see above for more detail about some of the steps):

```jl
wpids = addprocs(3)   # launch 3 worker processes, and we'll use CUDA this time

using Images, Unitful, StaticArrays, CUDArt, JLD, AxisArrays
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerAperturesMismatch

# Some physical units we'll need
const μm = u"μm"  # micrometers
const s  = u"s"   # seconds

# Here's the input file we'll be processing
fn = "/fish_raid/donghoon/dl_revision_005/20150818/exp8_20150818.imagine"

### Apertured registration
# Load the data
#   The mode="r" is needed if you don't have write permission to the
#   file, or don't want to risk accidents
img0 = load(fn, mode="r")
# Snip out a region of interest (discard empty portions of the image stack)
roi = (351:1120, :, :)
img = view(img0, roi..., :)
# Select our "fixed" image
fixedidx = (nimages(img)+1) ÷ 2
fixed0 = img[timeaxis(img)(fixedidx)]

# Important: you should manually inspect fixed0 to make sure there are
# no anomalies. Do not proceed to the next step until you have done this.

# Make sure the pixelspacing property is set correctly; edit the .imagine
# file with a text editor if necessary.
ps = img["pixelspacing"]
# Define preprocessing. Here we'll highpass filter over 25μm, but these
# numbers are likely to be image-dependent.
sigmahp = Float64[25μm/x for x in ps]
sigmalp = [0,0,0]

# The pco cameras have a bias of 100 in "digital number"
# units. Convert this into the units of the image intensity
bias = reinterpret(eltype(img0), UInt16(100))
pp = PreprocessSNF(bias, sigmalp, sigmahp)
fixed = pp(fixed0)

# Inspect fixed to see if it looks reasonable. If your images are very
# noisy, you might need to do some smoothing (adjust sigmalp) to
# ensure that images will "flow" into alignment, without getting
# trapped by pixel noise.  (When you later create the corrected
# images, this blurring will not be present---this is used only for
# the purpose of defining the deformation that best aligns the
# images.)  But don't blur so much that you destroy features that help
# with alignment.

# Set up the grid of apertures for aligning the images. You should use
# a grid that is fine enough to capture the regional differences, but
# keep in mind that speed of processing is dramatically worsened by
# grids that are bigger than necessary. This will likely require some
# experimentation.
gridsize = (15,13,5)  # or you could use round(Int, Float64[50μm/x for x in ps]) for one aperture each 50μm
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


## Registration tricks
(by Jerry)
Here, I would like to share my experience. Briefly, I sample images at a few time points, preprocess them, and obtain deformation vectors through registration. Then, the vectors are interpolated across time and applied to the original image. By the way, do not expect a perfect registration, but aim to obtain analyzable data.

I acquired a volumetric timelapse image (x, y, z, time) using OCPI1 in Holy Lab.
The below are some properties of my image:
- The object is an ex vivo neuronal tissue expressing a calcium indicator (thus, intensity fluctuates over time). 
- The voxel size is 0.577 by 0.5770 by 5 micrometer.
- The size of image is 1120 x 1080 x 60 pixels.
- One stack per 2 seconds (, and I often acquire more than 2000 stacks; occasionally more than 5000 stacks).
- More non-linear, regional movement (warping) than translational movement.
- Warping is neither rapid nor drastic. For example, two images at time = 0 and at time = 3 min are fairly identical.
- Without neuronal activity, signal-noise ratio was low (Camera bias is 100. Pixel intensity in tissue region is about 120~130.).

If image moves rapidly, the first two below might not be a good option.
1. Choose good images for registration:
I selected images that do not show evoked calcium activity. There are still spontaneous activity.

2. Median or quantile filtering across time (temporal median/quantile filtering):
This is to reduce noise and spontaneous activity.

3. Replace too high intensity pixels with NaN:
Sometimes, I observed high-intensity objects moving around tissue surface. While these things could be biologically important features, this is disastrous for registration. I ended up with replacing high intensity object (or pixels) with NaN. e.g) `img[img .> thresh] = NaN`

4. Run test registration:
I first registered a few sample image stacks, adjusting parameters. There might be good starting parameter values. The below are the parameters that I frequently play around. With the size of my image and the degree of warping, I initially chose parameters below:
```jl
#### Parameters for fixed image.
bias = 100
ps = [0.5770, 0.5770, 5] # pixel spacing
sigma = 20 μm # either 20 or 25  seems work fine.
sigmahp = Float64[sigma/x for x in ps] #Highpass filter
sigmalp = [3, 3, 0] #Lowpass filter

#### Parameters for algorithm structure
maxshift = (30, 30, 3)  # This corresponds to (17.3 micrometer x 17.3 micrometer, 15micrometer). This depends on degree of warping in your image.
gridsize = (15, 15, 8) # or gridsize = (24,24,12) # Again, my image size is 1120 X 1080 X 60. Both gridsizes gave fairly good results, but I prefer larger grid size.
λ = 1e-4 
algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; pid=wpids[i], correctbias=false) for i = 1:length(wpids)] #Notice that correctbias = false 

#### In warping step, 
ϕ_s = griddeformations(u, knots) #I didn't apply median filtering. 
```

The steps above aim to register only a few selected images. If that give a good result, interpolate the deformation vector `ϕ_s` using `tinterpolate`. Finally, apply the interpolated ϕ_s to warp entire dataset.

Example script: by the way, I included one of my own modules.
```jl
wpids = addprocs(6)
using BlockRegistration, BlockRegistrationScheduler
using RegisterWorkerApertures
using Images, ImageView
using Unitful, StaticArrays, AxisArrays, JLD
using Jerry_RegisterUtils #This module is in `LabShare` repository

# Some physical units we'll need
const μm = u"μm"  # micrometers
const s  = u"s"   # seconds

#### Load image
fn = "/mnt/donghoon_036/20170830/exp1_20170830.imagine"
img0 = load(fn, mode="r")

#### 1. Sample stim stacks
stimidx = sampleStimstacks(view(img0, :,:,30,:), 0.0000040, -3) #in Jerry_RegisterUtils; See `?sampleStimstacks`
tindex0 = [20; stimidx; nimages(img0)-50] #this will be used for registration

#### 2. Temporal median filtering (Run only one time)
tmedian_filter(Float32, "exp1_med.cam", img0, collect(-3:3), tindex0) #in Jerry_RegisterUtils; See `?tmedian_filter`
ImagineFormat.save_header("exp1_med.imagine", fn, view(img0, :,:,:,tindex0), Float32) #Create header file.

#### 3. High intensity thresholding
img0 = load("exp1_med.imagine") #Load the filtered image
img1 = mappedarray(val -> val > 140 ? NaN : val, img0); # Replace high intensity pixels with NaN. 140 is a threshold. NaN is a value for replacement.
img1 = AxisArray(img1, (:x, :l, :z, :time), (0.5770μm, 0.5770μm, 5μm, 2s)); # Assign axes again. 
# `img1` is a preprocessed image. It will be used for getting deformation vectors. However, the original image will use deformation vectors and be warped.

#### 4. Select image stacks for test registration
tind = [1:10:21; 23; 31:10:length(tindex0)]
tindex1 = tindex0[tind] 
roi = (:, :, :, tindex1)
img = view(img1, roi[1:3]..., tind)

#### Once parameters have been decided, load an image for actual registration
#roi = (:, :, :, tindex0) #This `roi` will be used for warping the original image. 
#img = img1

#### Fixed image
fixedidx = (nimages(img)+1) ÷ 2
fixedidx = 23
fixed0 = view(img, timeaxis(img0)(fixedidx))
  ps = pixelspacing(img)
  σ = 20μm
  sigmahp = Float64[σ/x for x in ps]
  sigmalp = [3,3,0]  # lowpass filtering is not currently recommended
  bias = 100 #bias = reinterpret(eltype(img0), UInt16(100))
  pp = PreprocessSNF(bias, sigmalp, sigmahp)
  fixed = pp(fixed0)

#### Set parameters and create the worker algorithm structures. 
gridsize = (24,24,13)
knots = map(d->linspace(1,size(fixed,d),gridsize[d]), (1:ndims(fixed)...))
mxshift = (30,30,3)
λ = 1e-4
algorithm = Apertures[Apertures(fixed, knots, mxshift, λ, pp; pid=wpids[i], correctbias=false) for i = 1:length(wpids)]
mon = monitor(algorithm, (), Dict{Symbol,Any}(:u=>ArrayDecl(Array{SVector{3,Float64},3}, gridsize)))
bname = splitext(splitdir(fn)[2])[1]
@show bname
fileout = string(bname, ".register")

#### Run reg
@time driver(fileout, algorithm, img, mon)
jldopen(fileout, "r+") do io
    write(io, "imagefile", fn)
    write(io, "roi", roi)
    write(io, "fixedidx", fixedidx)
    write(io, "knots", knots)
    write(io, "sigmalp", sigmalp)
    write(io, "sigmahp", sigmahp)
end

#### Warp selected images: The original image will be warped instead of the preprocessed image.
using Images, ImagineFormat, FileIO
using BlockRegistration
u = load(fileout, "u");
roi = load(fileout, "roi");
knots = load(fileout, "knots");
img0 = load(fn, mode="r");
img = view(img0, roi...);

ϕs = griddeformations(u, knots)
bname_warp = "exp1_20170830_register"
open(string(bname_warp, ".cam"), "w") do file
    warp!(Float32, file, img, ϕs; nworkers=3)
end
ImagineFormat.save_header(string(bname_warp, ".imagine"), fn, img, Float32)

#### Warp all! : Do not run this unless all parameters are optimized.
#ϕs_warpall = tinterpolate(ϕs, tindex0, nimages(img0));
#bname_warpall = "exp1_20170830_warpall"
#open(string(bname_warpall, ".cam"), "w") do file
#    warp!(Float32, file, img0, ϕs_warpall; nworkers = 8)
#end
#ImagineFormat.save_header(string(bname_warpall, ".imagine"), fn, img0, Float32)
```
