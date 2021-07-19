### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 3d20921a-e8a7-11eb-119c-f55d1a197148
begin
	using Pkg
	Pkg.activate(@__DIR__)
	using AMDGPU
	using ImageFiltering, ColorTypes, FixedPointNumbers, TensorCore
	using FileIO, ImageMagick
	using ImageShow
	using PlutoUI
end

# ╔═╡ ca1ff428-f3f9-490c-9da7-dcafef674782
lilly = FileIO.load("Lilly_hat.jpg")

# ╔═╡ 025a666c-d066-48d5-9788-9cb4e9634783
begin
	lilly_gpu = ROCArray(map(RGB{Float32}, lilly))
	# Let's be careful not to try rendering a GPU array!
	Array(lilly_gpu)
end

# ╔═╡ 1ba58648-2433-44ad-9d8f-e9269a383dc2
md"Let's do some simple image operations on this image. Let's start with a basic negative:"

# ╔═╡ 2c3f32e2-5b16-47e0-9483-3ef7c8907efc
begin
	lilly_negative = RGB(1) .- lilly_gpu
	Array(lilly_negative)
end

# ╔═╡ 699fd845-797e-4f0d-ae76-a8cee06d1de7
md"Cool! We can also adjust the brightness of the image pretty easily:"

# ╔═╡ d28ec333-e3fd-40ba-849b-4f93accacf91
begin
	lilly_darker = lilly_gpu .* 0.5
	Array(lilly_darker)
end

# ╔═╡ 0fa198b7-74c5-49db-b0ca-69efbbdc3395
md"Array operations work well for some things, but to get at more complicated operations, we sometimes need to write our operations as GPU kernels directly. Let's implement a basic translation:"

# ╔═╡ 01b99862-652f-4235-88f3-040b9c734bb3
function translate_kernel(out, inp, translation)
    x_idx = (blockDim().x * (blockIdx().x - 1)) + threadIdx().x
    y_idx = (blockDim().y * (blockIdx().y - 1)) + threadIdx().y

    x_outidx = x_idx + translation[1]
    y_outidx = y_idx + translation[2]

    if (1 <= x_outidx <= size(out,1)) &&
       (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
    return
end

# ╔═╡ aeb70d82-c647-43c6-918e-e50ea0bacf0c
function exec_gpu(f, sz, args...)
	wait(@roc groupsize=(32,32) gridsize=sz f(args...))
end	

# ╔═╡ a6d50928-c419-4599-92f5-51649c039e03
begin
	lilly_moved = similar(lilly_gpu)
	lilly_moved .= RGB(0)
	exec_gpu(translate_kernel, size(lilly_gpu), lilly_moved, lilly_gpu, (-500, 1000))
	Array(lilly_moved)
end

# ╔═╡ 2ab9753a-8406-4622-a140-eac11380d3d7
md"Great, now let's do a scale operation:"

# ╔═╡ bf12763b-d41f-467e-aa5e-80aa15202980
function scale_kernel(out, inp, scale)
    x_idx = (blockDim().x * (blockIdx().x - 1)) + threadIdx().x
    y_idx = (blockDim().y * (blockIdx().y - 1)) + threadIdx().y

    x_outidx = floor(Int, x_idx * scale[1])
    y_outidx = floor(Int, y_idx * scale[2])

    if (1 <= x_outidx <= size(out,1)) &&
       (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
    return
end

# ╔═╡ dc2d6965-7766-4e12-a0a8-a263e81f19bd
begin
	lilly_scaled = similar(lilly_gpu)
	lilly_scaled .= RGB(0)
	exec_gpu(scale_kernel, size(lilly_gpu), lilly_scaled, lilly_gpu, (0.5, 0.2))
	Array(lilly_scaled)
end

# ╔═╡ 8730e4f8-ead5-4e88-9e5d-8e5cfe213eee
md"Finally, let's rotate this puppy:"

# ╔═╡ 11aa59de-5918-4043-a625-173b0c55fce3
function rotate_kernel(out, inp, angle)
    x_idx = (blockDim().x * (blockIdx().x - 1)) + threadIdx().x
    y_idx = (blockDim().y * (blockIdx().y - 1)) + threadIdx().y

    x_centidx = x_idx - (size(inp,1)÷2)
    y_centidx = y_idx - (size(inp,2)÷2)
    x_outidx = round(Int, (x_centidx*cos(angle)) + (y_centidx*-sin(angle)))
    y_outidx = round(Int, (x_centidx*sin(angle)) + (y_centidx*cos(angle)))
    x_outidx += (size(inp,1)÷2)
    y_outidx += (size(inp,2)÷2)

    if (1 <= x_outidx <= size(out,1)) &&
       (1 <= y_outidx <= size(out,2))
        out[x_outidx, y_outidx] = inp[x_idx, y_idx]
    end
    return
end

# ╔═╡ 136196a6-e972-4f6b-95a5-faecd4214f73
begin
	lilly_rotated = similar(lilly_gpu)
	lilly_rotated .= RGB(0)
	exec_gpu(rotate_kernel, size(lilly_gpu), lilly_rotated, lilly_gpu, deg2rad(145))
	Array(lilly_rotated)
end

# ╔═╡ b7174da1-b875-46f3-8e85-26e2e379a3a0
md"Awesome! Those 3 operations are foundational to image processing, and they were easy to implement in Julia! However, there are plenty of other useful operators. How about we implement a Gaussian filter so we can get a blurry puppy?"

# ╔═╡ 36dc1d4b-6299-4f68-8e1d-91d7f739308c
md"We can use ImageFiltering to generate the kernels for us on the CPU, and then we just need to massage them into something that'll work on the GPU:"

# ╔═╡ 9b81e95c-6ffd-406b-8bf5-9c935f57a4e2
begin
	gaussian_k = Kernel.gaussian(15)
	gaussian_kG = ROCArray(map(x->RGB{Float32}(Gray(x)), gaussian_k.parent))
	gaussian_offsets = abs.(gaussian_k.offsets) .- 1
end;

# ╔═╡ 646b1d7f-ad81-408f-958f-ed4c31cc4e61
md"And now we need to write out GPU kernel. We're going to implement a correlation (often incorrectly called a convolution) operation, which we can use to apply a filtering kernel to an input image:"

# ╔═╡ 6f47a3f9-f836-485b-b5a2-ce44d0e4c216
function corr_kernel(out, inp, kern, offsets)
    x_idx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    y_idx = threadIdx().y + (blockDim().y * (blockIdx().y - 1))

    out_T = eltype(out)

    if (1 <= x_idx <= size(out,1)) && (1 <= y_idx <= size(out,2))
        x_toff, y_toff = offsets

        # create our accumulator
        acc = zero(out_T)

        # iterate in column-major order for efficiency
        for y_off in -y_toff:1:y_toff, x_off in -x_toff:1:x_toff
            y_inpidx, x_inpidx = y_idx+y_off, x_idx+x_off
            if (1 <= y_inpidx <= size(inp,2)) && (1 <= x_inpidx <= size(inp,1))
                y_kernidx, x_kernidx = y_off+y_toff+1, x_off+x_toff+1
                acc += hadamard(inp[x_inpidx, y_inpidx],
                                kern[x_kernidx, y_kernidx])
            end
        end
        out[x_idx, y_idx] = acc
    end

    nothing
end

# ╔═╡ e93d4dfb-5a28-4c15-abb7-e2980d143e1e
begin
	lilly_blurry = similar(lilly_gpu)
	exec_gpu(corr_kernel, size(lilly_gpu), lilly_blurry, lilly_gpu, gaussian_kG, gaussian_offsets)
	Array(lilly_blurry)
end

# ╔═╡ 593e5e91-f604-4420-9f38-772d51e8b0dc
md"There are plenty of interesting features in this image. Let's use a Sobel filter to see what the edges in the image look like. Note that a Sobel filtering operation actually uses two filter kernels, but we can get a good idea with just one of them."

# ╔═╡ 00deb882-7fb3-4bfe-aad9-1131c9ef017b
begin
	sobel_k = Kernel.sobel()
	sobel_kG = ROCArray(map(RGB{Float32}, sobel_k[1].parent))
	sobel_offsets = abs.(sobel_k[1].offsets) .- 1

	lilly_sobel = similar(lilly_gpu)
	exec_gpu(corr_kernel, size(lilly_gpu), lilly_sobel, lilly_gpu, sobel_kG, sobel_offsets)
	# Post-process the Sobel gradients into something comprehendable by humans
	lilly_sobel = map(x->mapc(y->y > 0 ? 1.0 : 0.0, x), lilly_sobel)
	Array(lilly_sobel)
end

# ╔═╡ Cell order:
# ╠═3d20921a-e8a7-11eb-119c-f55d1a197148
# ╠═ca1ff428-f3f9-490c-9da7-dcafef674782
# ╠═025a666c-d066-48d5-9788-9cb4e9634783
# ╟─1ba58648-2433-44ad-9d8f-e9269a383dc2
# ╠═2c3f32e2-5b16-47e0-9483-3ef7c8907efc
# ╟─699fd845-797e-4f0d-ae76-a8cee06d1de7
# ╠═d28ec333-e3fd-40ba-849b-4f93accacf91
# ╟─0fa198b7-74c5-49db-b0ca-69efbbdc3395
# ╠═01b99862-652f-4235-88f3-040b9c734bb3
# ╠═aeb70d82-c647-43c6-918e-e50ea0bacf0c
# ╠═a6d50928-c419-4599-92f5-51649c039e03
# ╠═2ab9753a-8406-4622-a140-eac11380d3d7
# ╠═bf12763b-d41f-467e-aa5e-80aa15202980
# ╠═dc2d6965-7766-4e12-a0a8-a263e81f19bd
# ╠═8730e4f8-ead5-4e88-9e5d-8e5cfe213eee
# ╠═11aa59de-5918-4043-a625-173b0c55fce3
# ╠═136196a6-e972-4f6b-95a5-faecd4214f73
# ╠═b7174da1-b875-46f3-8e85-26e2e379a3a0
# ╠═36dc1d4b-6299-4f68-8e1d-91d7f739308c
# ╠═9b81e95c-6ffd-406b-8bf5-9c935f57a4e2
# ╠═646b1d7f-ad81-408f-958f-ed4c31cc4e61
# ╠═6f47a3f9-f836-485b-b5a2-ce44d0e4c216
# ╠═e93d4dfb-5a28-4c15-abb7-e2980d143e1e
# ╠═593e5e91-f604-4420-9f38-772d51e8b0dc
# ╠═00deb882-7fb3-4bfe-aad9-1131c9ef017b
