# JuliaCon 2021 - GPU Workshop material

This repository contains the notebooks and other material for the
[GPU workshop at JuliaCon 2021](https://www.youtube.com/watch?v=aKRv-W9Eg8g).

* `deep_dive`: a notebook that explains the different GPU programming models,
  array programming and kernel programming, and demonstrates what they can
  and cannot be used for using a series of examples. Depending on the exact
  back-end, different tools are shown to facilitate GPU development.
* `case_studies`: more hands-on demonstrations of the GPU programming
  functionality to implement specific applications and algorithms.
* `kernelabstractions`: an in-depth demonstration of KernelAbstractions.jl,
  an alternative way to program GPUs in Julia using a vendor-agnostic
  kernel programming abstraction
* `enzyme`: a sneak peek at the new LLVM-based autodifferentiation support,
  demonstrated on a parallel GPU kernel.
