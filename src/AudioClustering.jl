"""
Stuff you can do:
$(EXPORTS)
- `mapsoundfiles`
- `audiograph`
- `model2file`
- `save_interesting`
- `embeddings`
- `interactive_heatmap`
- `associate_clusters`
"""
module AudioClustering
using DocStringExtensions
using LinearAlgebra, Statistics, Base.Threads
using DelimitedFiles, Reexport, ThreadTools, Lazy
using NearestNeighbors, Arpack, LightGraphs, SimpleWeightedGraphs, Hungarian




export mapsoundfiles, audiograph, model2file, save_interesting, embeddings, invembeddings, invembedding, associate_clusters
export interactive_heatmap

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


"""
    mapsoundfiles(f::F, files, [windowlength]) where F
    $(SIGNATURES)

map a Function over a list of wav files.

#Arguments:
- `f`: a Function
- `files`: DESCRIPTION
- `windowlength`: Optional. If supplied, `f` will be applied over non-overlapping windows for the sound, and a `Vector{Vector{RT}}` is returned, where `RT` is the return type of `f`.
"""
function mapsoundfiles(f::F,files,windowlength) where F
    GC.gc()
    # extension === nothing || (files = filter(f->splitext(f)[end] == extension, files))
    @time embeddings = tmap1(max(nthreads()-1,1),files) do file
        @info "Reading file $file"
        sound = wavread(file)[1]
        map(f,Iterators.partition(sound, windowlength))
    end
end

function mapsoundfiles(f::F,files) where F
    GC.gc()
    # extension === nothing || (files = filter(f->splitext(f)[end] == extension, files))
    @time embeddings = tmap1(max(nthreads()-1,1),files) do file
        @info "Reading file $file"
        sound = wavread(file)[1]
        f(sound)
    end
end

"""
    model2file(model, models, files)

Finds the file corresponding to the model
"""
function model2file(model, models, files)
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

function WAV.wavplay(path::String)
    Threads.@spawn run(`totem $path`)
end

"""
    new_assignment, permutation = associate_clusters(labels, assignments)

Return a permutation between cluster assignments and true (supervised) labels such that clusters corresponds to labels. This is useful evaluate how well a clustering corresponds to known classes. Clustering assigns arbitrary numbers to the clusters, and this function finds the best associations between true and assigned labels.
"""
function associate_clusters(labels::Vector{Int}, assignments::Vector{Int})
    ulabels = sort(unique(labels))
    D = [sum((labels .== i) .& (assignments .!= j)) for i in ulabels, j in ulabels]
    perm = hungarian(D)[1]
    replace(assignments, Pair.(ulabels,perm)...), perm
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

"""
    audiograph(X::AbstractMatrix, k::Int, distance, models::Vector{<:SpectralDistances.AbstractModel}; λ=0)

The returned graph is built using distances between models. To speed up computations, `k` nearest neighbors are first computed using Euclidean distances on `X`. After this, `distance` is computed between models that are determined to be neighbors.

#Arguments:
- `X`: The data matrix, size n_feature × n_data
- `k`: number of neighbors
- `distance`: DESCRIPTION
- `models`: DESCRIPTION
- `λ`: Kernel precision. Higher value means tighter kernel.
"""
function audiograph(X::AbstractMatrix, k::Int, distance, models::Vector{<:SpectralDistances.AbstractModel}; λ=0)
    N = length(models)
    inds, _ = knn(X, k)
    @assert k == length(inds[1])
    dists = tmap1(nthreads(),1:N) do i
        map(inds[i]) do neighbor
            distance(models[i],models[neighbor])
        end
    end
    SimpleWeightedDiGraph(repeat((1:N)', k)[:], reduce(vcat, inds),
                    λ == 0 ? Float32.(reduce(vcat, dists)) : Float32.(exp.(.-λ .*reduce(vcat, dists))))
end

"""
    audiograph(X::AbstractMatrix, k::Int=5; λ=0)

Simple Euclidean nearest-neighbor graph.

#Arguments:
- `X`: The data matrix, size n_feature × n_data
- `k`: number of neighbors
- `λ`: Kernel precision. Higher value means tighter kernel.
"""
function audiograph(X::AbstractMatrix,k::Int=5; λ=0)
    inds, dists = knn(X, k)
    A = _nn2graph(inds, dists, λ)
end


save_interesting(sounds, inds, args...) = save_interesting(sounds, findall(inds), args...)

"""
    save_interesting(sounds::Vector, inds::Vector{Int}, contextwindow=1)

Saves interesting wav files to disk, together with one concatenated file which contains all the itneresting sounds. The paths will be printed in the terminal.

#Arguments:
- `inds`: A list of indices to interesting sound clips
- `contextwindow`: Saves this many clips before and after
"""
function save_interesting(sounds::AbstractVector{T}, inds::Vector{Int}, contextwindow=1, fs=41000) where T
    tempdir = mktempdir()
    for ind ∈ inds
        extended_inds = max(1, ind-contextwindow):min(length(sounds), ind+contextwindow)
        sound = map(extended_inds) do i
            getsound(sounds[i])
        end
        sound = reduce(vcat, sound)[:]
        filename = T <: AbstractString ? splitpath(files[ind])[end] : string(ind)
        tempfile = joinpath(tempdir, filename*".wav")
        sound .-= mean(Float32.(sound))
        sound .*= 1/maximum(abs.(sound))
        wavwrite(sound, tempfile, Fs=fs)
        println(tempfile)
    end
    save_interesting_concat(sounds, inds, tempdir, fs=fs)
end

getsound(sound) = sound
function getsound(sound::AbstractString)
    ext = splitext(sound)[2]
    if ext == "wav"
        wavread(sound)
    else
        deserialize(sound)
    end
end

function save_interesting_concat(sounds, inds::Vector{Int}, tempdir=mktempdir(); fs=41000)
    sound = map(inds) do i
        sound = getsound(sounds[i])
        sound .-= mean(Float32.(sound))
        sound .*= 1/maximum(abs.(sound))
    end
    sound = reduce(vcat, sound)[:]
    tempfile = joinpath(tempdir, "concatenated.wav")
    wavwrite(sound, tempfile, Fs=fs)
    println(tempfile)
    tempdir
end

include("plotting.jl")

"""
    embeddings(models::AbstractVector{<:AbstractModel})

Returns an embedding matrix that contains poles of the system models, expanded into real and imaginary parts. Redundant poles (complex conjugates) are removed, so the height of the embedding matrix is the same as the number of poles in the system models.
"""
function embeddings(models::AbstractVector{<: AbstractModel})
    emb = ContinuousRoots.(move_real_poles.(roots.(Ref(Discrete()), models), 1e-2))
    X0 = reduce(hcat, emb)
    X = Float64.([real(X0[1:end÷2,:]); imag(X0[1:end÷2,:])])
end

function invembeddings(embeddings::AbstractMatrix)
    map(invembedding, eachcol(embeddings))
end

function invembedding(e::AbstractVector)
    r = complex.(e[1:end÷2], e[end÷2+1:end])
    cr = ContinuousRoots([r; conj.(r)])
    AR(cr)
end



# """
#     U,V,convergence_history = lowrankmodel(X, k=size(X, 1) - 4; λ=1.0e-5)
#
# Fit a low-rank model `X ≈ U'V`.
# `U` will be the dictionary and `V` the activations. Try `heatmap(V)` and see if you can spot any patterns.
#
# #Arguments:
# - `X`: The data matrix, size n_feature × n_data
# - `k`: number of features
# - `λ`: Regularization parameter, higher number yields sparser result
# """
# function lowrankmodel(X, k=size(X,1)-4; λ=0.00001)
#     losses = QuadLoss() # minimize squared distance to cluster centroids
#     rx     = OneReg(λ)
#     ry     = QuadReg(0.01)
#     glrm   = GLRM(X,losses,ry,rx,k, offset=true, scale=false)
#     init_svd!(glrm)
#     U,V,ch = fit!(glrm, ProxGradParams(1.0,max_iter=50,inner_iter=1,abs_tol=0.00001,rel_tol=0.0001))
#     @info "Relative error: $(mean(abs2,X - U'V)/mean(abs2,X))"
#     U,V,ch
# end

#
#
# import MLJBase
#
# Base.@kwdef mutable struct SparseLowRank <: MLJBase.Unsupervised
#     k::Int
#     λx::Float64 = 0.01
#     λy::Float64 = 0.01
# end
#
# # fit returns coefficients minimizing a penalized rms loss function:
# function MLJBase.fit(model::SparseLowRank, verbosity::Int, X)
#     x = MLJBase.matrix(X)                     # convert table to matrix
#     losses = QuadLoss()
#     rx = NonNegOneReg(model.λx)
#     ry = QuadReg(model.λy)
#     glrm = GLRM(X,losses,ry,rx,model.k, offset=true, scale=true)
#     init_svd!(glrm)
#     U,V,ch = fit!(glrm, ProxGradParams(1.0,max_iter=1000,inner_iter=1,abs_tol=1e-9,rel_tol=1e-9), verbosity=verbosity)
#     (U,V,glrm), 0, ch
# end
#
# # predict uses coefficients to make new prediction:
# function MLJBase.transform(model::SparseLowRank, fitresult, X)
#     U,V,glrm = fitresult
#     x = MLJBase.matrix(X)                     # convert table to matrix
#     losses = QuadLoss()
#     m2 = deepcopy(glrm)
#     m2.X .= U
#     U,V,ch = fit(glrm, ProxGradParams(1.0,max_iter=1000,inner_iter=1,abs_tol=1e-9,rel_tol=1e-9))
#     V
# end
#
# function MLJBase.inverse_transform(::SparseLowRank, fitresult, Xnew)
#     fitresult[1]' * MLJBase.matrix(Xnew)
# end


using Requires
function __init__()
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        @require AbstractPlotting = "537997a7-5e4e-5d89-9595-2241ea00577e" begin
            @require Observables = "510215fc-4207-5dde-b226-833fc4488ee2" begin
                include("plotting.jl")
            end
        end
    end
end

end # module
