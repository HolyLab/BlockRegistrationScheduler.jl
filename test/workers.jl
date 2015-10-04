using Images, AffineTransforms, TestImages, RegisterDriver, RegisterWorkerRigid
using Base.Test

# workdir = tempname()
# mkdir(workdir)

fixed = testimage("cameraman")
tfm = tformrotate(pi/12)
moving = transform(fixed, tfm)

alg = Rigid(fixed, pat=false, print_level=5)
mon = monitor(alg, ())
mon[:tform] = nothing
mon[:mismatch] = 0.0
mon = driver(alg, moving, mon)
ptfm = mon[:tform]*tfm
@test_approx_eq_eps ptfm.scalefwd eye(2) 0.01
@test all(abs(ptfm.offset) .< 0.2)
@test mon[:mismatch] < 0.001
