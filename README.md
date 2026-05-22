# BlockRegistrationScheduler.jl

[![CI](https://github.com/HolyLab/BlockRegistrationScheduler.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/BlockRegistrationScheduler.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/BlockRegistrationScheduler.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/BlockRegistrationScheduler.jl)

This meta-package has been replaced by individual packages:

- [RegisterDriver](https://github.com/HolyLab/RegisterDriver.jl)
- [RegisterWorkerApertures](https://github.com/HolyLab/RegisterWorkerApertures.jl)
- [RegisterWorkerAperturesMismatch](https://github.com/HolyLab/RegisterWorkerAperturesMismatch.jl)

You can load this as a package, and it will load these others, but generally you should
just use these packages directly.

To get started, see the documentation for [RegisterWorkerApertures](https://github.com/HolyLab/RegisterWorkerApertures.jl).

The old README is available [here](OLD_README.md).

## Installation

This package is registered in the [HolyLab registry](https://github.com/HolyLab/HolyLabRegistry). Add the registry once, then install normally:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("BlockRegistrationScheduler")
```
