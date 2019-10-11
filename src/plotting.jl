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
