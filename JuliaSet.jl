### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 42c404aa-e8b0-11eb-24a1-3d8971e86c0b
begin
	using Pkg
	Pkg.activate(@__DIR__)
	using AMDGPU
	using Colors, ImageShow, PlutoUI
end

# ╔═╡ 852ee289-9ecb-426f-ab6b-cd336b2028d7
md"Let's draw a Julia Set with Julia and GPUs! Shamelessly borrowed from https://nextjournal.com/sdanisch/julia-gpu-programming"

# ╔═╡ eba05f9e-33d1-49c3-b200-17d505af22a1
function juliaset(z0, maxiter)
    c = ComplexF32(-0.5, 0.75)
    z = z0
    for i in 1:maxiter
        abs2(z) > 4f0 && return (i - 1) % UInt8
        z = z * z + c
    end
    return maxiter % UInt8 # % is used to convert without overflow check
end

# ╔═╡ 706a7d64-19ab-4a78-972f-cc7df1be78e3
md"This slider adjusts the size of the image"

# ╔═╡ 58644700-f889-4fdf-89ed-7304b5ebfc51
@bind N Slider(100:50:2^12)

# ╔═╡ f767778b-e7e2-46c1-bc00-18182563c9ea
md"This slider adjusts how many iterations to compute before stopping"

# ╔═╡ 45d49c9a-f483-4149-aa5b-9b76b200cee8
@bind iter Slider(1:5:100)

# ╔═╡ 35914818-0a4c-4f69-bee3-c3427b57ed6f
begin
	w, h = N, N
	q = [ComplexF32(r, i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5]
	q_gpu = ROCArray(q)
	result = AMDGPU.zeros(UInt8, size(q))
	result .= juliaset.(q_gpu, iter)
	cmap = colormap("Blues", iter + 1)
	color_lookup(val, cmap) = cmap[val + 1]
	color_lookup.(Array(result), (cmap,))
end

# ╔═╡ Cell order:
# ╠═42c404aa-e8b0-11eb-24a1-3d8971e86c0b
# ╠═852ee289-9ecb-426f-ab6b-cd336b2028d7
# ╠═eba05f9e-33d1-49c3-b200-17d505af22a1
# ╠═706a7d64-19ab-4a78-972f-cc7df1be78e3
# ╠═58644700-f889-4fdf-89ed-7304b5ebfc51
# ╠═f767778b-e7e2-46c1-bc00-18182563c9ea
# ╠═45d49c9a-f483-4149-aa5b-9b76b200cee8
# ╠═35914818-0a4c-4f69-bee3-c3427b57ed6f
