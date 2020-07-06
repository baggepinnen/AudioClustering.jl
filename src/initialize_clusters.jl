"""
    calc_N_random!(dist, D, X, N; kwargs...)

Calculate the `dist` between `N` random paris of points from `X` and store in `D`. Since `D` is symmetric, `2N` spots will be filled.

# Arguments:
- `dist`: A distance
- `D`: A sparse matrix
- `X`: A vector of datapoints
"""
function calc_N_random!(dist, D, X, Nr; kwargs...)
    N = length(X)
    dists = [deepcopy(dist) for _ in 1:nthreads()]
    prog = Progress(N*Nr, 1, "Calculating initial random-pair distances")
    l = ReentrantLock()
    GC.gc()
    @threads for i in 1:N
        for r = 1:Nr
            i,j = rand(1:N, 2)
            d = evaluate(dists[threadid()], X[i], X[j]; kwargs...)
            lock(l) do
                D[i,j] = d
                D[j,i] = d
            end
            rand() < 0.001 && GC.gc()
            next!(prog)
        end
    end
    D
end

"""
    inds, D = initialize_clusters(dist, X; init_multiplier = 10, N_seeds = 100)

Return a vector of indices into `X` that indicate good cluster centers. `D` is a sparse matrix containing all calculated distances.

# Arguments:
- `dist`: a distance object
- `X`: a vector of datapoints
- `init_multiplier`: this many random distances per data point are calculated
- `N_seeds`: this many indices are returned
"""
function initialize_clusters(dist, matrices; init_multiplier = 10, N_seeds = 100)
    D = spzeros(N,N)
    calc_N_random!(dist, D, matrices, init_multiplier)
    GC.gc(true);GC.gc();GC.gc(true);GC.gc();GC.gc(true);
    g = SimpleWeightedDiGraph(D)
    c = Parallel.pagerank(g); GC.gc()
    inds0 = copy(partialsortperm(c, 1:min(N_seeds,length(c)), rev=true)), D
end

"""
    crop_time(x, maxtime)

Corp `x` along second dimension (time) if it's greater than `maxtime`. The window is selected to contain maximum mass.
"""
function crop_time(x, maxtime)
    maxtime >= lastlength(x) && return x
    masses = vec(sum(x, dims=1))
    sm,_ = sliding_mean_std(masses, maxtime)
    i = argmax(sm)
    x[:,i:i+maxtime-1]
end
