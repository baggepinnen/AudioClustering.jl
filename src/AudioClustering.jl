module AudioClustering
using LinearAlgebra, Statistics, Base.Threads
using DelimitedFiles, Reexport, ThreadTools, Lazy
using NearestNeighbors, Arpack, LightGraphs, SimpleWeightedGraphs, LowRankModels

using Makie, AbstractPlotting, Observables


export mapsoundfiles, audiograph, lowrankmodel, model2file, save_interesting, embeddings
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
    mapsoundfiles(f::F, files, windowlength) where F

map a Function over a list of wav files.

#Arguments:
- `f`: a Function
- `files`: DESCRIPTION
- `windowlength`: DESCRIPTION
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

DOCSTRING

#Arguments:
- `X`: The data matrix, size n_feature × n_data
- `k`: number of neighbors
- `distance`: DESCRIPTION
- `models`: DESCRIPTION
- `λ`: Kernel precision. Higher value means tighter kernel.
"""
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

"""
    audiograph(X::AbstractMatrix, k::Int=5; λ=0)

DOCSTRING

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


"""
    U,V,convergence_history = lowrankmodel(X, k=size(X, 1) - 4; λ=1.0e-5)

Fit a low-rank model `X ≈ U'V`.
`U` will be the dictionary and `V` the activations. Try `heatmap(V)` and see if you can spot any patterns.

#Arguments:
- `X`: The data matrix, size n_feature × n_data
- `k`: number of features
- `λ`: Regularization parameter, higher number yields sparser result
"""
function lowrankmodel(X, k=size(X,1)-4; λ=0.00001)
    losses = QuadLoss() # minimize squared distance to cluster centroids
    rx     = OneReg(λ)
    ry     = QuadReg(0.01)
    glrm   = GLRM(X,losses,ry,rx,k, offset=true, scale=false)
    init_svd!(glrm)
    U,V,ch = fit!(glrm, ProxGradParams(1.0,max_iter=50,inner_iter=1,abs_tol=0.00001,rel_tol=0.0001))
    @info "Relative error: $(mean(abs2,X - U'V)/mean(abs2,X))"
    U,V,ch
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




import MLJBase

Base.@kwdef mutable struct SparseLowRank <: MLJBase.Unsupervised
    k::Int
    λx::Float64 = 0.01
    λy::Float64 = 0.01
end

# fit returns coefficients minimizing a penalized rms loss function:
function MLJBase.fit(model::SparseLowRank, verbosity::Int, X)
    x = MLJBase.matrix(X)                     # convert table to matrix
    losses = QuadLoss()
    rx = NonNegOneReg(model.λx)
    ry = QuadReg(model.λy)
    glrm = GLRM(X,losses,ry,rx,model.k, offset=true, scale=true)
    init_svd!(glrm)
    U,V,ch = fit!(glrm, ProxGradParams(1.0,max_iter=1000,inner_iter=1,abs_tol=1e-9,rel_tol=1e-9), verbosity=verbosity)
    (U,V,glrm), 0, ch
end

# predict uses coefficients to make new prediction:
function MLJBase.transform(model::SparseLowRank, fitresult, X)
    U,V,glrm = fitresult
    x = MLJBase.matrix(X)                     # convert table to matrix
    losses = QuadLoss()
    m2 = deepcopy(glrm)
    m2.X .= U
    U,V,ch = fit(glrm, ProxGradParams(1.0,max_iter=1000,inner_iter=1,abs_tol=1e-9,rel_tol=1e-9))
    V
end

function MLJBase.inverse_transform(::SparseLowRank, fitresult, Xnew)
    fitresult[1]' * MLJBase.matrix(Xnew)
end











end # module
