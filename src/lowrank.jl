using Optim, Zygote, Plots
function normalize_mass!(x)
    # x .-= minimum(x, dims=1)
    x .= max.(0, x)
    x ./= sum(x, dims=1)
end

# function normalize_grad!(x)
#     x .-= mean(x, dims=1)
# end

function entropy(x)
    sum(x .* log.(x .+ eps(eltype(x))))
    # sum(x->x*log(x + eps(eltype(x))), x)
end

function init_A_X(dists, A, k)
    N = length(A)
    T = eltype(eltype(A))
    m, n = size(A[1])

    @info "Smoothing inputs"
    xi1 = Matrix{T}(undef, m, m)
    xi2 = Matrix{T}(undef, n, n)
    SpectralDistances._initialize_conv_op!(xi1, xi2, dists[1].β)
    A = tmap(A) do x
        s1(xi1 * x * xi2)
    end

    Am = reduce(hcat, vec.(A))
    q = quantile(vec(Am), 0.2)
    # Xvm = s1.([A[i] .+ q for i in round.(Int, range(1, stop = N, length = k + 2)[2:end-1])])
    Xvm = s1.([A[i] .+ q .* rand.() for i in round.(Int, range(1, stop = N, length = k))])
    @assert length(Xvm) == k


    Xm = reduce(hcat, vec.(Xvm))
    @assert all(isfinite, Xm)
    @assert all(isfinite, Am)
    @assert all(≈(1), sum(Xm, dims=1))
    A, Xm, Xvm
end

function init_Y(dists,A,Xvm)
    @info "Calculating initial barycentric coordinates"
    T = eltype(eltype(A))
    m, n = size(A[1])
    k = length(Xvm)
    Yvv = tmap(A) do a
        l = barycentric_coordinates(dists[Threads.threadid()], Xvm, a)[1] .+ 0/k .* rand.() |> s1
        all(isfinite, l) || return s1(ones(k) + rand(k))
        l
        # l |> softmax # Applying this increases entropy a bit
        # log.(l .+ T(1e-6)) # log transform since we take softmax in cost function
    end
    Ym = reduce(hcat, Yvv)
    @assert all(isfinite, Ym)
    @assert all(≈(1), sum(Ym, dims=1))
    Ym
end


function lowrankmodel_optim(d, A, k; epochs = 100, kwargs...)

    N = length(A)
    T = eltype(eltype(A))
    m, n = size(A[1])
    dists = [deepcopy(d) for _ = 1:Threads.nthreads()]

    A, Xm, Xvm = init_A_X(dists, A, k)
    Ym = init_Y(dists,A,Xvm)

    @info "Calculating diagonal costs"
    diagonal_costs = tmap(eachindex(A)) do i
        evaluate(dists[Threads.threadid()], A[i], A[i]; kwargs...)
    end
    @show diagonal_costs
    @assert all(isfinite, diagonal_costs)

    function lowrankcost(Xmi, Ymi)
        Ah = Xmi * Ymi
        cs = sum(eachindex(A)) do i
            Ai = reshape((Ah[:, i]), size(A[1]))
            # c2 = Threads.@spawn evaluate(dists[Threads.threadid()], Ai, Ai; kwargs...)
            c1 = evaluate(dists[Threads.threadid()], A[i], Ai; kwargs...)
            # ct = fetch(c1) - T(0.5) * (fetch(c2) + diagonal_costs[i]) #+ 10abs2(sAi - 1)
            ct = fetch(c1) - diagonal_costs[i] #+ 0.01entropy(Ym)
        end
        return cs/length(A)
    end

    losses = T[]
    @show Ym

    function fg!(F, G, XmYm)
        m = length(Xm)
        Xm .= reshape(@view(XmYm[1:m]), size(Xm))
        Ym .= reshape(@view(XmYm[m+1:end]), size(Ym))
        normalize_mass!(Xm)
        normalize_mass!(Ym)
        if G != nothing
            loss, back = Zygote.pullback(lowrankcost, Xm, Ym)
            Xmg, Ymg = back(Zygote.sensitivity(loss))
            G[1:length(Xmg)] .= vec(Xmg)
            G[length(Xmg)+1:end] .= vec(Ymg)
            return loss
        end
        if F != nothing
            return lowrankcost(Xm, Ym)
        end
    end


    function callback(trace)
        push!(losses, trace[end].value)
        plot(
            heatmap(Ym, colorbar = true, color = :blues),
            # heatmap(-Ymg, colorbar = true),
            plot([heatmap(reshape((x), size(A[1])), axis = false) for x in eachcol(Xm)]..., colorbar=false),
            plot(losses, lab = ""),
            # plot(heatmap(A[1]), heatmap(reshape((Xm * Ym[:, 1]), m, n)), colorbar = true),
            # plot([heatmap(reshape((x), size(A[1])), axis = false) for x in eachcol(-Xmg)]...),
            size = (2500, 1200),
        ) |> display
        false
    end


    # @show size(Xm), size(Ym)
    @info "Starting optimization"
    res = Optim.optimize(
        Optim.only_fg!(fg!),
        [vec(Xm); vec(Ym)],
        NGMRES(alphaguess = Optim.LineSearches.InitialHagerZhang(αmax=0.001),
        linesearch = Optim.LineSearches.HagerZhang(alphamax=0.001)),
        # SimulatedAnnealing(),
        Optim.Options(
            store_trace = true,
            show_trace = true,
            show_every = 1,
            iterations = epochs,
            allow_f_increases = true,
            time_limit = 3000,
            x_tol = 1e-10,
            f_tol = 1e-10,
            g_tol = 1e-5,
            callback = callback
        ),
    )

    m = length(Xm)
    Xm = reshape(res.minimizer[1:m], size(Xm))
    Ym = reshape(res.minimizer[m+1:end], size(Ym))

    [reshape(x, size(A[1])) for x in eachcol(Xm)], Ym, res

end


function lowrankmodel_sgd(d, A, k; epochs = 100, optX, optY, kwargs...)

    N = length(A)
    T = eltype(eltype(A))
    m, n = size(A[1])
    dists = [deepcopy(d) for _ = 1:Threads.nthreads()]
    A, Xm, Xvm = init_A_X(dists, A, k)
    Ym = init_Y(dists,A,Xvm)

    @info "Calculating diagonal costs"
    diagonal_costs = tmap(eachindex(A)) do i
        evaluate(dists[Threads.threadid()], A[i], A[i]; kwargs...)
    end
    @show diagonal_costs
    @assert all(isfinite, diagonal_costs)

    function lowrankcost(Xmi, Ymi)
        Ah = Xmi * Ymi
        cs = sum(eachindex(A)) do i
            Ai = reshape((Ah[:, i]), size(A[1]))
            # c2 = Threads.@spawn evaluate(dists[Threads.threadid()], Ai, Ai; kwargs...)
            c1 = evaluate(dists[Threads.threadid()], A[i], Ai; kwargs...)
            # ct = fetch(c1) - T(0.5) * (fetch(c2) + diagonal_costs[i]) #+ 10abs2(sAi - 1)
            ct = fetch(c1) - diagonal_costs[i] + 0.001entropy(Ym)

            # c2 = 0
            # c1 = 1-dot(A[i], Ai)/sqrt(dot(A[i], A[i])*dot(Ai, Ai))
            # c1 = norm(A[i] - Ai) + 0.1entropy(Ym) #- 0.01entropy(Xm)
            # ct = (c1) - diagonal_costs[i]
            # c12 = evaluate(dists[1], A[i], Ai; kwargs...)
            # c22 = evaluate(dists[2], Ai, Ai; kwargs...)
            # ct2 = c12 - 0.5*(c22 + diagonal_costs[i])
            # @show ct-ct2
            # ct
            ct
        end
        return cs/length(A)

        # cs #+ 0.0sum(abs, sAh .- 1) + 0.0sum(abs, mAh)
    end

    losses = T[]
    @show Ym


    # Gradtest ===================
    # g1 = SpectralDistances.ngradient(Xm) do Xm
    #     lowrankcost(Xm, Ym)
    # end
    #
    # g2 = Zygote.gradient(Xm) do Xm
    #     lowrankcost(Xm, Ym)
    # end[1]
    #
    # @show norm(g2 - g1)/sqrt(norm(g1)*norm(g2))
    # @assert norm(g2 - g1)/sqrt(norm(g1)*norm(g2)) < 0.1
    # =======================================
    try
        for epoch = 1:epochs
            ps = (Xm, Ym)
            loss, back = Zygote.pullback(lowrankcost, ps...)
            @printf("Iter: %3d loss: %g\n", epoch, loss)
            isfinite(loss) || error("Loss is not finite")
            gs = back(Zygote.sensitivity(loss))
            epoch == 1 && continue
            push!(losses, loss)
            Xmg, Ymg = gs
            clamp!(Xmg, -0.5, 0.5)
            # normalize_grad!(Xmg)
            # normalize_grad!(Ymg)

            epoch % 40 == 1 &&
                plot(
                    heatmap(Ym, colorbar = true, color = :blues),
                    heatmap(-Ymg, colorbar = true),
                    plot([heatmap(reshape((x), size(A[1])), axis = false) for x in eachcol(Xm)]...),
                    plot(losses, lab = ""),
                    plot(heatmap(A[1]), heatmap(reshape((Xm * Ym[:, 1]), m, n)), colorbar = true),
                    plot([heatmap(reshape((x), size(A[1])), axis = false) for x in eachcol(-Xmg)]...),
                    size=(2500,1200)
                ) |> display

            Flux.Optimise.update!(optX, Xm, Xmg)
            Flux.Optimise.update!(optY, Ym, Ymg)
            # apply(optX, vec(Xmg), vec(Xm))
            # apply(optY, vec(Ymg), vec(Ym))

            normalize_mass!(Xm)
            normalize_mass!(Ym)

        end
    catch e
        e isa InterruptException || rethrow(e)
    end

    # @show size(Xm), size(Ym)

    m = length(Xm)
    # Xm = reshape(res.minimizer[1:m], size(Xm))
    # Ym = reshape(res.minimizer[m+1:end], size(Ym))

    [reshape(x, size(A[1])) for x in eachcol(Xm)],
    Ym,
    losses

end
