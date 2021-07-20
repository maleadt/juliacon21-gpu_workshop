using Pkg
Pkg.activate(@__DIR__)

using KernelAbstractions

if GPU_PKG_NAME == "CUDA"
    using CUDA, CUDAKernels
    const GPUMOD = CUDA
    const GpuArray = CuArray
    const GpuBackend = CuDevice()
elseif GPU_PKG_NAME == "AMDGPU"
    using AMDGPU, ROCKernels
    const GPUMOD = AMDGPU
    const GpuArray = ROCArray
    const GpuBackend = ROCDevice()
elseif GPU_PKG_NAME == "oneAPI"
using oneAPI
    const GPUMOD = oneAPI
    const GpuArray = ZeArray
    const GpuBackend = CPU()
end

using ImageShow
