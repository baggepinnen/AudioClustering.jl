module AudioClustering
using LinearAlgebra, Statistics, Base.Threads
using DelimitedFiles, Reexport, ThreadTools, Lazy
using NearestNeighbors, Arpack, LightGraphs, SimpleWeightedGraphs, LowRankModels

@reexport using DSP
@reexport using SpectralDistances
@reexport using Distances
@reexport using WAV
@reexport using Clustering
@reexport using SparseArrays


# struct DistanceMatrix{T} <: AbstractMatrix
#     M::T
# end
#
# struct AdjacencyMatrix{T} <: AbstractMatrix
#     M::T
# end
#
# const AbstractGraphMatrix = Union{DistanceMatrix, AdjacencyMatrix}
#
# SparseArrays.SparseMatrixCSC(m::AbstractGraphMatrix) = m.M
#
# Lazy.@forward DistanceMatrix.M (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, LinearAlgebra.eigen, LinearAlgebra.eigvals, LinearAlgebra.svd, LinearAlgebra.qr, Arpack.eigs)
# Lazy.@forward AdjacencyMatrix.M (Base.length, Base.getindex, Base.setindex!, Base.size, Base.enumerate, LinearAlgebra.eigen, LinearAlgebra.eigvals, LinearAlgebra.svd, LinearAlgebra.qr, Arpack.eigs)


function mapsoundfiles(f::F,files,windowlength, extension=nothing) where F
    GC.gc()
    extension === nothing || (files = filter(f->splitext(f)[end] == extension, files))
    @time embeddings = tmap(2nthreads(),files) do file
        @info "Reading file $file"
        sound = wavread(file)[1]
        map(f,Iterators.partition(sound, windowlength))
    end
end
function model2file(model, files)
    findres = findfirst.(==(model), models)
    fileno = findfirst(!=(nothing), findres)
    files[fileno]
end
function WAV.wavplay(x::AbstractArray,fs)
    path = "/tmp/a.wav"
    wavwrite(x,path,Fs=fs)
    @info "Wrote to file $path"
    Threads.@spawn run(`totem $path`)
end

function _nn2graph(inds, dists, λ=0)
    N = length(inds)
    k = length(inds[1])
    SimpleWeightedDiGraph(repeat((1:N)', k)[:], reduce(vcat, inds),
                    λ == 0 ? reduce(vcat, dists) : exp.(.-λ .*reduce(vcat, dists)))
end

function NearestNeighbors.knn(X::AbstractMatrix,k::Int)
    tree = NearestNeighbors.KDTree(X)
    inds, dists = knn(tree, X, k)
end

function audiograph(X::AbstractMatrix, k::Int, distance, models::Vector{<:SpectralDistances.AbstractModel}; λ=0)
    inds, _ = knn(X, k)
    N = length(inds)
    k = length(inds[1])
    dists = tmap1(nthreads(),1:N) do i
        map(inds[i]) do neighbor
            distance(models[i],models[neighbor])
        end
    end
    SimpleWeightedDiGraph(repeat((1:N)', k)[:], reduce(vcat, inds),
                    λ == 0 ? reduce(vcat, dists) : exp.(.-λ .*reduce(vcat, dists)))
end

function audiograph(X::AbstractMatrix,k::Int=5; λ=0)
    inds, dists = knn(X, k)
    A = _nn2graph(inds, dists, λ)
end


save_interesting(files, inds, args...) = save_interesting(files, findall(inds), args...)

function save_interesting(files, inds::Vector{Int}, contextwindow=1)
    tempdir = mktempdir()
    error("This should work on wavfiles instead")
    for ind ∈ inds
        extended_inds = max(1, ind-contextwindow):min(length(files), ind+contextwindow)
        sound = map(extended_inds) do i
            deserialize(files[i])
        end
        sound = reduce(vcat, sound)[:]
        tempfile = joinpath(tempdir, splitpath(files[ind])[end]*".wav")
        sound .-= mean(Float32.(sound))
        sound .*= 1/maximum(abs.(sound))
        wavwrite(sound, tempfile, Fs=fs)
        println(tempfile)
    end
end


end # module
