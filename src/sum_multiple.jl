using CUDA

function reduce_grid_atomic_shmem(op, a::AbstractArray{T}, b) where {T}
    elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    offset = (block-1) * elements

    # shared mem to buffer memory loads
    shared = @cuStaticSharedMem(T, (2048,))
    @inbounds shared[thread] = a[offset+thread]
    @inbounds shared[thread+blockDim().x] = a[offset+thread+blockDim().x]

    # parallel reduction of values in a block
    d = 1
    while d < elements
        sync_threads()
        index = 2 * d * (thread-1) + 1
        @inbounds if index <= elements && offset+index+d <= length(a)
            shared[index] = op(shared[index], shared[index+d])
        end
        d *= 2
    end

    # atomic reduction
    if thread == 1
        @atomic b[] = op(b[], shared[1])
    end

    return
end

NVTX.@range function my_sum(a::AbstractArray{T}) where {T}
    b = CUDA.zeros(T, 1)

    kernel = @cuda launch=false reduce_grid_atomic_shmem(+, a, b)

    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(a))
    blocks = cld(length(a), threads*2)

    @cuda threads=threads blocks=blocks reduce_grid_atomic_shmem(+, a, b)

    CUDA.@allowscalar b[]
end

NVTX.@range function my_multiple_sums(a::AbstractArray{T}) where {T}
    n = size(a)[end]
    dims = [axes(a)...][begin:end-1]
    sums = Array{T}(undef, (fill(1, ndims(a)-1)..., n))
    for x in 1:size(a,3)
        y = view(a, dims..., x)
        sums[x] = my_sum(y)
    end
    sums
end

function main()
    a = CUDA.rand(1024, 1024, 10)
    @assert my_multiple_sums(a) ≈ Array(sum(a; dims=(1,2)))

    CUDA.@profile begin
        my_multiple_sums(a)
        my_multiple_sums(a)
    end
end

isinteractive() || main()
