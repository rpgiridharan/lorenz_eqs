#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:02:57 2025

@author: ogurcan
"""

import numpy as np
import cupy as cp
import matplotlib.pylab as plt
from juliacall import Main as jl

t0 = 0.0
t1 = 50.0
dtstep = 0.01

params = {
    'sigma': 10.0,
    'rho': 28.0,
    'beta': 8/3
}

# Initial conditions using CuPy for GPU
y0_gpu = cp.array([1.0, 1.0, 1.0], dtype=cp.float32)

# Define the RHS function in Python for GPU arrays
def rhs_gpu(dY, Y, p, t):
    # Since we can't directly access CuPy's dtype from Julia's CuArray
    # Let's use a different approach to update the dY values
    
    # Convert GPU arrays to NumPy for calculations
    Y_np = cp.asnumpy(Y)
    x, y, z = Y_np[0], Y_np[1], Y_np[2]
    sigma, rho, beta = p['sigma'], p['rho'], p['beta']
    
    # Calculate the derivatives
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    # Update each element individually using indexing from NumPy
    # This avoids needing to access the dtype attribute
    dY_np = cp.asnumpy(dY)  # Get a copy we can modify
    dY_np[0] = dx
    dY_np[1] = dy
    dY_np[2] = dz
    
    # Manually copy each value back to the GPU array
    for i in range(len(dY_np)):
        # Use raw pointer manipulation to set the values
        # without relying on dtype attribute
        jl.seval(f"CUDA.@allowscalar dY[{i+1}] = {float(dY_np[i])}")
        
    return dY

# Get the pointer to the CuPy array's memory
y0_ptr = y0_gpu.data.ptr
y0_shape = y0_gpu.shape
y0_dtype = y0_gpu.dtype

# Pass parameters and pointers to Julia
jl.rhs_gpu_py = rhs_gpu
jl.y0_ptr = y0_ptr
jl.y0_shape = y0_shape
jl.y0_dtype_str = str(y0_dtype)
jl.tspan = (t0, t1)
jl.params_py = params

# Run the simulation in Julia using the CuPy array pointer
jl.seval("""
using DifferentialEquations
using PythonCall
using CUDA

# Create CUDA array from Python CuPy pointer
function ptr_to_cudaarray(ptr, shape, dtype_str)
    # Map dtype string to Julia type
    dtype = if dtype_str == "float32"
        Float32
    elseif dtype_str == "float64"
        Float64
    else
        error("Unsupported dtype: $dtype_str")
    end
    
    # Create CUDA array that shares memory with the CuPy array
    # Note: shape[0] is the size since we're working with 1D arrays
    cuda_ptr = CuPtr{dtype}(ptr)
    return unsafe_wrap(CuArray, cuda_ptr, shape)
end

# Get the CUDA array from CuPy pointer
y0_cu = ptr_to_cudaarray(y0_ptr, y0_shape, y0_dtype_str)

# Create a wrapper function that uses the Python RHS function with CUDA arrays
function rhs_cuda!(du, u, p, t)
    # Need to use allowscalar because the Python function will
    # need to set individual elements
    CUDA.allowscalar(true)
    
    # Call the Python GPU function
    rhs_gpu_py(du, u, p, t)
    
    # Reset allowscalar to false for performance
    CUDA.allowscalar(false)
    return nothing
end

# Set up and solve the ODE problem
prob = ODEProblem(rhs_cuda!, y0_cu, tspan, params_py)
sol = solve(prob, TRBDF2(autodiff=false), abstol=1e-6, reltol=1e-6)

# Keep solution on GPU
ures_cu = mapreduce(permutedims, vcat, sol.u)
# Get a pointer to the GPU memory for Python to use
ures_ptr = pointer(ures_cu)
ures_shape = size(ures_cu)
ures_dtype = eltype(ures_cu)
ures_dtype_str = string(ures_dtype)
""")

# Create a CuPy array that shares memory with Julia's CUDA array
ures_shape = jl.ures_shape
ures_dtype = np.dtype(jl.ures_dtype_str.lower())
ures_gpu = cp.ndarray(
    shape=ures_shape,
    dtype=ures_dtype,
    memptr=cp.cuda.MemoryPointer(jl.ures_ptr, 0, False),
)

# Transfer solution to CPU only for plotting
xyz = cp.asnumpy(ures_gpu)

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz Attractor (GPU-accelerated)')
plt.show()