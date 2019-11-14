using AudioClustering, Test
using SpectralDistances, Random, DSP

@testset "AudioClustering" begin

    @testset "Graph stuff" begin
        @info "Testing Graph stuff"

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



    end

    @testset "Lowrank" begin
        @info "Testing Lowrank"


        Random.seed!(1)
        N_clusters = 5
        N_signals = 50

        signals = map(1:N_clusters) do i
            passband = sort(0.5rand(2))
            signals = map(1:N_signals) do s
                SpectralDistances.bp_filter(randn(500), passband)
            end

        end

        perm = randperm(sum(length, signals))
        assi = repeat((1:N_clusters)', N_signals)[perm]
        allsounds = reduce(vcat, signals)[perm]

        na = 8
        fitmethod = TLS(na=na)
        # fitmethod = PLR(na=na, nc=1, initial=TLS(na=20))
        @time models = tmap(allsounds) do sound
            fitmodel(fitmethod, sound)
        end

        X = embeddings(models)

        @test size(X) == (na, length(allsounds))

        k = size(X,1)-2
        U,V,ch = lowrankmodel(X, k; λ=0.00001)

        @test size(U) == (k,size(X,1))
        @test size(V) == (k,size(X,2))

        @test mean(abs2, U'V - X) / mean(abs2, X) < 1e-5

        # heatmap(U)
        # heatmap(V)
        # I = sortperm(V[2,:])
        # heatmap(V[:,I])


        g = audiograph(V, N_clusters)
        # heatmap(g.weights)

        @testset "Save interesting" begin
            @info "Testing Save interesting"


            sounddir = AudioClustering.save_interesting(allsounds, [1,3])
            @test walkdir(sounddir) |> first |> last |> length == 3

        end

    end




end
