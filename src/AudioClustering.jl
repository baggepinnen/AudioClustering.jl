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
using LinearAlgebra, Statistics, Base.Threads
using DelimitedFiles, Reexport, ThreadTools, Lazy
using NearestNeighbors, Arpack, LightGraphs, SimpleWeightedGraphs, Hungarian




export mapsoundfiles, audiograph, model2file, save_interesting, embeddings, invembeddings, invembedding, associate_clusters, distmat2similarity, distmat2nn_graph
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

end # module
