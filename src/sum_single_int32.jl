using CUDA

function reduce_grid_atomic_shmem(op, a::AbstractArray{T}, b) where {T}
    elements = Int32(blockDim().x) * Int32(2)
    thread = Int32(threadIdx().x)
    block = Int32(blockIdx().x)
    offset = (block-Int32(1)) * elements

    # shared mem to buffer memory loads
    shared = @cuStaticSharedMem(T, (2048,))
    @inbounds shared[thread] = a[offset+thread]
    @inbounds shared[thread+blockDim().x] = a[offset+thread+blockDim().x]

    # parallel reduction of values in a block
    d = Int32(1)
    while d < elements
        sync_threads()
        index = Int32(2) * d * (thread-Int32(1)) + Int32(1)
        @inbounds if index <= elements && offset+index+d <= length(a)
            shared[index] = op(shared[index], shared[index+d])
        end
        d *= Int32(2)
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

function main()
    a = CUDA.rand(1024, 1024)
    @assert my_sum(a) ≈ sum(a)

    CUDA.@profile begin
        my_sum(a)
        my_sum(a)
    end
end

isinteractive() || main()
