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
    inds, dists, D = knn_accelerated(d, X, k, Xe=X; kwargs...)

Find the nearest neighbor using distance `d` by first finding the `k` nearest neighbors using Euclidean distance on embeddings Xe, and then using `d` do find the smallest distance within those `k`.

`D` is a sparse matrix with all the computed distances from `d`. This matrix contains raw distance measurements, to symmetrize, call `SpectralDistances.symmetrize!(D)`.
The returned `dists` are already symmetrized.

# Arguments:
- `d`: distance supporting `evaluate`.
- `X`: data
- `k`: number of Euclidean nearest neighbors to consider. The computational cost scales linearly with `k`
- `kwargs`: are sent to `evaluate`
"""
function knn_accelerated(d, X::AbstractVector, k, xe=X; kwargs...)

    GC.gc()
    N = length(X)
    XE = embeddings(xe)
    m,N = size(XE)
    @info "Size of embedding: " (m,N)
    inds, _ = knn(XE, k+1, true)
    GC.gc()

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

    dists = [deepcopy(d) for _ in 1:nthreads()]

    prog = Progress(N, 1, "Calculating diagonal")
    DD = tmap(1:N) do i
        val = evaluate(dists[threadid()], X[i], X[i]; kwargs...)
        next!(prog)
        val
    end
    D = spdiagm(0 => DD)
    l = ReentrantLock()
    GC.gc()

    prog = Progress(length(inds), 1, "Calculating neighbors")
    # tmap(nthreads(), eachindex(inds)) do i
    @threads for i in eachindex(inds)
        for j in @views(inds[i][2:end])
            @assert j != i "This should not occur"
            dij = evaluate(dists[threadid()], X[i], X[j]; kwargs...)
            lock(l) do
                D[i,j] = dij
                D[j,i] = dij
            end
        end
        next!(prog)
    end
    GC.gc()

    D2 = symmetrize!(deepcopy(D))
    offset = 1.1*maximum(D2)
    nzcache = copy(D2.nzval)
    D2.nzval .-= offset
    newinds = vec(getindex.(argmin(D2+Inf*I, dims=2), 2))
    D2.nzval .= nzcache

    @info "Euclidean and $(typeof(d)) agreement: $(mean(newinds .== getindex.(inds,2)))"

    newinds, [D2[i,j] for (i,j) in zip(1:N, newinds)], D

end

function mutual_neighbors(Xi; verbose=true, kwargs...)
    X = Xi isa Matrix ? Xi : reduce(hcat, vec.(Xi))
    m,N  = size(X)
    if m > 1000 || N < 5000
        verbose && @info "Computing NN using distance matrix"
        if N < 10_000
            D = pairwise(Euclidean(), X)
            D[diagind(D)] .= Inf
            dists, inds = findmin(D, dims=2)
            inds = vec(getindex.(inds, 2))
        else
            error("Dimension too large, this will take a long time")
        end
    else
        verbose && @info "Computing NN using KD-tree"
        inds, dists = knn(X, 2, true) # TODO: this takes forever for large representations
        inds, dists = getindex.(inds, 2), getindex.(dists, 2)
    end
    # workspaces = [SCWorkspace(X[1], X[1], d.β) for _ in 1:nthreads()]

    ordered_tuple(a,b) = a < b ? (a,b) : (b,a)
    mutual_neighbors = Set{Tuple{Int,Int}}()
    for i in 1:N
        ni = inds[i]
        if inds[ni] == i
            push!(mutual_neighbors, ordered_tuple(i,ni))
        end
    end
    if isempty(mutual_neighbors)
        verbose && @info "Failed to find enough pairs of mutual nearest neighbors"
        return Tuple{Int, Int}[]
    end
    verbose && @info "Found $(length(mutual_neighbors)) mutual neighbor pairs"
    [mutual_neighbors...]

end

function reduce_dataset(d, X; verbose = true, recursive = 0, kwargs...)
    @show mn = mutual_neighbors(X; verbose = verbose)
    isempty(mn) && return X
    B = tmap(mn) do (i,j)
        barycenter(d, [X[i], X[j]]; kwargs...)
    end
    reduced = [getindex.(mn, 1); getindex.(mn, 2)]
    X2 = copy(X)
    X2 = deleteat!(X2, sort(reduced))
    if recursive > 0
        return [reduce_dataset(d, X2; verbose = verbose, recursive = recursive-1, kwargs...); B]
    end
    return [X2; B]
end


function softmax(x)
    e = exp.(x)
    e ./ sum(e)
end

# function WL_costfun(d,X,λ; kwargs...)
#     λ2 = softmax.(eachcol(λ))
#     sum(enumerate(λ2)) do (i,λi)
#         B = barycenter(d, X, λi; kwargs...)
#         evaluate(d, B, X[i]; kwargs...)
#     end
# end

#
# function WL_costfun(d,X,λ; kwargs...)
#     λ2 = softmax.(eachcol(λ))
#     B = map(enumerate(λ2)) do (i,λi)
#         B = sum(X[j] * λi[j] for j in eachindex(X))
#     end
#     sum(evaluate(d, B, X[i]; kwargs...)
# end
#
# function Wlowrankmodel(d, X, k; kwargs...)
#     N = length(X)
#     λ = fill(1/k, N, k)
#     B = X[randperm(N)[1:k]]
#
# end


function pattern_classify(dist, patterns, X; kwargs...)
    N      = length(X)
    labels = Vector{Int}(undef, N)
    D      = zeros(length(patterns), N)
    dists  = [deepcopy(dist) for _ in 1:nthreads()]
    prog   = Progress(N, 1, "Classifying")
    @threads for i in eachindex(X)
        di = @view(D[:,i])
        map!(di, patterns) do p
            # NOTE: If using ZNormalizer, one must take care of un-equal lengths as ZNorm divides by the length. If using NormNormalizer, this is no longer a problem as all windows will have the same "√energy"
            evaluate(dists[threadid()], X[i], p; kwargs...)
        end
        labels[i] = argmin(di)
        next!(prog)
    end
    labels, D
end
