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

function my_sum(a::AbstractArray{T}, b::CuArray) where {T}
    kernel = @cuda launch=false reduce_grid_atomic_shmem(+, a, b)

    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(a))
    blocks = cld(length(a), threads*2)

    @cuda threads=threads blocks=blocks reduce_grid_atomic_shmem(+, a, b)
end

function my_multiple_sums(a::AbstractArray{T}) where {T}
    n = size(a)[end]
    dims = [axes(a)...][begin:end-1]
    sums = CUDA.zeros(T, (fill(1, ndims(a)-1)..., n))
    for x in 1:size(a,3)
        y = view(a, dims..., x)
        my_sum(y, view(sums, fill(1, ndims(a)-1)..., x))
    end
    sums
end

function main()
    a = CUDA.rand(1024, 1024, 10)
    @assert Array(my_multiple_sums(a)) ≈ Array(sum(a; dims=(1,2)))

    CUDA.@profile begin
        NVTX.@range "sum" CUDA.@sync my_multiple_sums(a)
        NVTX.@range "sum" CUDA.@sync my_multiple_sums(a)
    end
end

isinteractive() || main()
