"""
    interactive_heatmap(data, sounds::Vector{SomethingPlayable}, f = wavplay)

Plots a heatmap where each cell is clickable. Clicking a cell causes the sound corresponding to the cell to be played. The vector of sounds can be either a vector of vectors, or a vector of file paths (basically anything that one can call `wavplay` on). The Function exectured can be changed from `wavplay` to any other.
"""
function interactive_heatmap(data, filesv, f=wavplay)
    data = data'
    limits = Makie.FRect(1,1,size(data)...)
    scene = Makie.heatmap(data, limits=limits)
    Observables.on(scene.events.mousebuttons) do val
        AbstractPlotting.Mouse.left ∈ val || return
        x,y = Makie.mouseposition(scene)
        x = ceil(Int, x)
        file = filesv[x]
        @info "playing file $file at position $x"
        f(file)
    end
    scene
end


"""
    interactive_scatter(X, Y, data, [spectrogram_markers]; f=wavplay, kwargs...)

Plot a scatter plot where each point can be clicked and `f` is executed on the corresponding entry of `data`.

# Arguments:
- `X`: x-coordinates
- `Y`: y-coordinates
- `data`: a vector of the same length as `x,y` with additional data
- `spectrogram_markers` This optional vector of matrices can be supplied in order to display a heatmap (spectrogram) as maker. See `SpectralDistances.normalize_spectrogram`.
- `kwargs`: are sent to `Makie.scatter`
"""
function interactive_scatter(X, Y, data; f=wavplay, kwargs...)
    tree = NearestNeighbors.KDTree([X'; Y'])
    scene = Makie.scatter(X,Y; kwargs...)
    Observables.on(scene.events.mousebuttons) do val
        AbstractPlotting.Mouse.left ∈ val || return
        x,y = Makie.mouseposition(scene)
        ind, _ = knn(tree, [x,y], 1)
        file = data[ind[]]
        @info "playing file $file at position $x, $y"
        f(file)
    end
    scene
end


function interactive_scatter(X, Y, data, spectrogram_markers; f=wavplay, kwargs...)

    specX = [(p=m; Gray.(Float32.((p .- minimum(p)) ./ (maximum(p)-minimum(p))))) for m in spectrogram_markers]
    # specX = [(p=m; Gray.(Float32.(p))) for m in spectrogram_markers]
    tree = NearestNeighbors.KDTree([X'; Y'])
    scene = Makie.Scene(resolution=(1600, 1000))
    for inds in Iterators.partition(1:length(X), 400)
        Makie.scatter!(scene, X[inds],Y[inds]; marker=specX[inds], kwargs...)
    end
    Observables.on(scene.events.mousebuttons) do val
        AbstractPlotting.Mouse.left ∈ val || return
        x,y = Makie.mouseposition(scene)
        ind, _ = knn(tree, [x,y], 1)
        file = data[ind[]]
        @info "playing file $file at position $x, $y"
        f(file)
    end
    display(scene)
end
