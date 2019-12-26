"""
    interactive_heatmap(data, sounds::Vector{SomethingPlayable})

Plots a heatmap where each cell is clickable. Clicking a cell causes the sound corresponding to the cell to be played. The vector of sounds can be either a vector of vectors, or a vector of file paths (basically anything that one can call `wavplay` on).
"""
function interactive_heatmap(data, filesv)
    data = data'
    limits = Makie.FRect(1,1,size(data)...)
    scene = Makie.heatmap(data, limits=limits)
    on(scene.events.mousebuttons) do val
        AbstractPlotting.Mouse.left ∈ val || return
        x,y = Makie.mouseposition(scene)
        x = ceil(Int, x)
        file = filesv[x]
        @info "playing file $file at position $x"
        wavplay(file)
    end
    scene
end


function interactive_scatter(X, Y, filesv; kwargs...)
    tree = NearestNeighbors.KDTree([X'; Y'])
    # limits = Makie.FRect(1,1,size(data)...)
    scene = Makie.scatter(X,Y; kwargs...)
    on(scene.events.mousebuttons) do val
        AbstractPlotting.Mouse.left ∈ val || return
        x,y = Makie.mouseposition(scene)
        ind, _ = knn(tree, [x,y], 1)
        file = filesv[ind[]]
        @info "playing file $file at position $x"
        wavplay(file)
    end
    scene
end
