using AudioClustering, Test


# @testset "AudioClustering" begin
g = AudioClustering.audiograph(randn(3,20), 5)
@test size(g) == (20,20)
@test nnz(g.weights) >= 20*5
@test all(>=(0), g.weights)

g = AudioClustering.audiograph(randn(3,20), 5, λ=1.)
@test size(g) == (20,20)
@test nnz(g.weights) >= 20*5
@test all(>=(0), g.weights)
@test all(<=(1), g.weights)


inds, dists = AudioClustering.knn(randn(2,100), 5)
@test length(inds) == 100
@test length(inds[1]) == 5
@test length(dists[1]) == 5


dist =


















# end
