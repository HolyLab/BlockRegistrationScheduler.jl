if Base.find_in_path("BlockRegistration") == nothing
    Pkg.clone("git@github.com:HolyLab/BlockRegistration.git")
    Pkg.build("BlockRegistration")
end
