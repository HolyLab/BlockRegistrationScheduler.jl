"""
    BlockRegistrationScheduler

Convenience bundle that re-exports the block-registration scheduling stack:
[`RegisterDriver`](@ref), [`RegisterWorkerShell`](@ref),
[`RegisterWorkerApertures`](@ref), and [`RegisterWorkerAperturesMismatch`](@ref).

Typical usage:

```julia
using BlockRegistrationScheduler
algorithm = AperturesMismatch(fixed, nodes, maxshift, λrange)
driver(outfile, algorithm, img, mon)
```
"""
module BlockRegistrationScheduler

using Reexport
using BlockRegistration

@reexport using RegisterDriver
@reexport using RegisterWorkerShell
@reexport using RegisterWorkerApertures
@reexport using RegisterWorkerAperturesMismatch

end
