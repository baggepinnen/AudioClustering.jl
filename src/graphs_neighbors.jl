"""
    new_assignment, permutation = associate_clusters(labels, assignments)

Return a permutation between cluster assignments and true (supervised) labels such that clusters corresponds to labels. This is useful evaluate how well a clustering corresponds to known classes. Clustering assigns arbitrary numbers to the clusters, and this function finds the best associations between true and assigned labels.
"""
function associate_clusters(labels::Vector{Int}, assignments::Vector{Int})
    ulabels = sort(unique(labels))
    D = [(sum((labels .!= i) .& (assignments .== j))) for i in ulabels, j in ulabels]
    perm = hungarian(D')[1]
    replace(assignments, Pair.(ulabels,perm)...), perm
end

function _nn2graph(inds, dists, λ=0)
    N = length(inds)
    k = length(inds[1])
    SimpleWeightedDiGraph(repeat((1:N)', k)[:], reduce(vcat, inds),
                    λ == 0 ? reduce(vcat, dists) : exp.(.-λ .*reduce(vcat, dists)))
end


function distmat2nn_graph(D, k::Int, distance=!all(diag(D) .== 1))
    @assert k > 0 "k must be >= 1 for this to make sense"
    k += 1
    N = size(D,1)
    G = spzeros(size(D)...)
    @inbounds for i = 1:N
        d = sortperm(D[:,i], rev=!distance)
        for j = 2:k
            if distance
                G[i,d[j]] = 1
                G[d[j],i] = 1
            else
                G[i,d[j]] = D[d[j],i]
                G[d[j],i] = D[d[j],i]
            end
        end
    end
    G
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


"""
    distmat2similarity(D,σ=5,λ=0.5)

Converts a distance matrix to a similarity matrix. If `σ` is an integer, then an adative density parameter is used based on the distance to the `σ`:th nearest neighbor (including itself). If `σ` is a float, a fixed value is used `exp(-λD/2σ²)`.
"""
function distmat2similarity(D,σ::Int=5,λ=0.5)
    N = size(D,1)
    ϵ = eps(eltype(D))
    @assert σ <= N
    nns = map(1:N) do i
        sort(D[:,i])[σ]
    end
    @. exp(-λ*D/sqrt(nns*nns' .+ ϵ))
end

function distmat2similarity(D,σ::AbstractFloat)
    σ² = 2σ^2
    @. exp(-D/σ²)
end
