using CUDA

function reduce_grid_atomic_shmem(op, a::AbstractArray{T}, b) where {T}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x
    offset = (block-1) * threads

    # shared mem to buffer memory loads
    shared = @cuStaticSharedMem(T, (1024,))
    @inbounds shared[thread] = a[offset+thread]

    # parallel reduction of values in a block
    d = 1
    while d < threads
        sync_threads()
        index = 2 * d * (thread-1) + 1
        @inbounds if index <= threads
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

function my_sum(a::AbstractArray{T}) where {T}
    b = CUDA.zeros(T, 1)

    kernel = @cuda launch=false reduce_grid_atomic_shmem(+, a, b)

    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(a))
    blocks = cld(length(a), threads)

    @cuda threads=threads blocks=blocks reduce_grid_atomic_shmem(+, a, b)

    CUDA.@allowscalar b[]
end

function main()
    a = CUDA.rand(1024, 1024)
    @assert my_sum(a) ≈ sum(a)
end

isinteractive() || main()
