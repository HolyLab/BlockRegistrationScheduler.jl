if Base.find_in_path("BlockRegistration") == nothing
    Pkg.clone("git@github.com:HolyLab/BlockRegistration.git")
    include(Pkg.dir("BlockRegistration","build","build.jl"))
end
Pkg.checkout("JLD")  # need master branch
