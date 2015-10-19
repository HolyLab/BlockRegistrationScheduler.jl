__precompile__(false)  # can't precompile because of path issues

module BlockRegistrationScheduler

using Reexport
using BlockRegistration

thisdir = splitdir(@__FILE__)[1]
if !any(LOAD_PATH .== thisdir)
    push!(LOAD_PATH, thisdir)
end

@reexport using RegisterDriver
@reexport using RegisterWorkerShell
@reexport using RegisterWorkerApertures
@reexport using RegisterWorkerAperturesMismatch
@reexport using RegisterWorkerRigid

end
