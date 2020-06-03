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

function NearestNeighbors.knn(X::AbstractMatrix,k::Int, args...)
    tree = NearestNeighbors.KDTree(X)
    inds, dists = knn(tree, X, k, args...)
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


"""
    inds, dists, D = knn_accelerated(d, X, k; kwargs...)

Find the nearest neighbor using distance `d` by first finding the `k` nearest neighbors using Euclidean distance, and then using `d` do find the smallest distance within those `k`.

`D` is a sparse matrix with all the computed distances from `d`. This matrix contains raw distance measurements, to symmetrize, call `SpectralDistances.symmetrize!(D)`.
The returned `dists` are already symmetrized.

# Arguments:
- `d`: distance supporting `evaluate`.
- `X`: data
- `k`: number of Euclidean nearest neighbors to consider. The computational cost scales linearly with `k`
- `kwargs`: are sent to `evaluate`
"""
function knn_accelerated(d, X, k; kwargs...)

    N = length(X)
    XE = reduce(hcat, vec.(X))
    m,N = size(XE)
    inds, dists = knn(XE, k, true)


    # if m > 1000 || N < 5000
    #     verbose && @info "Computing NN using distance matrix"
    #     if N < 10_000
    #         D = pairwise(Euclidean(), XE)
    #         D[diagind(D)] .= Inf
    #         dists, inds = findmin(D, dims=2)
    #         inds = vec(getindex.(inds, 2))
    #     else
    #         error("Dimension too large, this will take a long time")
    #     end
    # else
    #     verbose && @info "Computing NN using KD-tree"
    #     inds, dists = knn(X, 2) # TODO: this takes forever for large representations
    #     inds, dists = getindex.(inds, 1), getindex.(dists, 1)
    # end

    dists = [deepcopy(d) for _ in 1:Threads.nthreads()]

    DD = tmap(1:N) do i
        evaluate(dists[Threads.threadid()], X[i], X[i]; kwargs...)
    end
    D = spdiagm(0 => DD)
    l = ReentrantLock()
    NN = tmap(eachindex(inds)) do i
        real_dists = map(inds[i]) do j
            j == i && return Inf
            dij = evaluate(dists[Threads.threadid()], X[i], X[j]; kwargs...)
            lock(l) do
                D[i,j] = dij
                D[j,i] = dij
            end
            dij
        end
        rd,ri = findmin(real_dists)
        j = inds[i][ri]
        local dᵢⱼ
        lock(l) do
            dᵢⱼ = rd - 0.5*(D[i,i] + D[j,j])
        end
        dᵢⱼ, j
    end

    newinds = getindex.(NN,2)
    @info "Euclidean and $(typeof(d)) agreement: $(mean(newinds .== getindex.(inds,2)))"

    newinds, getindex.(NN,1), D

end
