"""
Stuff you can do:
$(EXPORTS)
- `mapsoundfiles`
- `audiograph`
- `thing2file`
- `save_interesting`
- `embeddings`
- `interactive_heatmap`
- `associate_clusters`
"""
module AudioClustering
using DocStringExtensions
using LinearAlgebra, Statistics, Base.Threads, Printf
using DelimitedFiles, Reexport, ThreadTools, Lazy
using NearestNeighbors, Arpack, LightGraphs, SimpleWeightedGraphs, Hungarian

using Optim

using AbstractTrees


export mapsoundfiles, audiograph, model2file, save_interesting, embeddings, invembeddings, invembedding, associate_clusters, distmat2similarity, distmat2nn_graph, knn, knn_accelerated
export interactive_heatmap

@reexport using DetectionIoTools
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



include("graphs_neighbors.jl")
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



## Hierarchical clustering approaches

struct Cluster
    inds::Vector{Int}
    level::Int
end

struct Clusters
    clusters::Vector{Cluster}
    clustered_indices::Vector{Int}
end

struct TopDownResult
    D::SparseMatrixCSC{Float64, Int}
    clusters::Clusters
end


AbstractTrees.children(node::Cluster)                      = node.inds
AbstractTrees.printnode(io::IO, node::Cluster)             = print(io, node.data)
AbstractTrees.children(node::TopDownResult)                = node.clusters
Base.eltype(::Type{<:TreeIterator{Cluster}})               = Cluster
Base.eltype(::Type{<:TreeIterator{TopDownResult}})         = Cluster
AbstractTrees.nodetype(::Cluster)                          = Cluster
AbstractTrees.nodetype(::TopDownResult)                    = Cluster
Base.IteratorEltype(::Type{<:TreeIterator{Cluster}})       = Base.HasEltype()
Base.IteratorEltype(::Type{<:TreeIterator{TopDownResult}}) = Base.HasEltype()


function topdown(d::ConvOptimalTransportDistance, Xi, k; n_init = 100, kwargs...)
    X  = s1.(normalize_spectrogram(Xi, d.dynamic_floor))
    N  = length(X)
    workspaces = [SCWorkspace(X[1], X[1], d.β) for _ in 1:Threads.nthreads()]
    D = spzeros(N,N)

    # Compute some number of random distances to get some statistics
    for k = 1:n_init
        i,j = rand(1:N, 2)
        D[i,j] == 0 || continue
        D[i,j] = sinkhorn_convolutional(w, X[i], X[j]; β = d.β, kwargs...)
        D[j,i] = D[i,j]
    end

    # Find the smallest computed distance
    m = minimum(nonzeros(D))
    i = findfirst(==(m), D) # seed index

    # Compute all distances for one of the members of the nearest-neighbor pair
    for j = 1:N
        D[i,j] == 0 || continue
        D[i,j] = sinkhorn_convolutional(w, X[i], X[j]; β = d.β, kwargs...)
        D[j,i] = D[i,j]
    end

    # Group the smallest 1/k quantile into the first cluster.
    # QUESTION: would it be better to group all within a certain distance relative to the statistics from above
    perm = partialsortperm(D[:,i], 1:N÷k)
    clusters = Clusters([Cluster(perm, 1)], perm)

end



function SpectralDistances.barycenter(d::ConvOptimalTransportDistance, X::Vector{<:Periodograms.TFR}, inds::Vector{<:Tuple}; kwargs...)

    tmap(inds) do x
        barycenter(d, X[[x...]]; kwargs...)
    end

end




include("lowrank.jl")


end # module
