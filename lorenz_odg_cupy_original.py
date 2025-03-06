#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:02:57 2025

@author: ogurcan
"""

import cupy as cp
from juliacall import Main as jl
import matplotlib.pylab as plt

t0 = 0.0
t1 = 50.0
dtstep = 0.01

sigma = 10.0
rho = 28.0
beta = 8/3

y0 = cp.array([1.0, 1.0, 1.0])
dy0=cp.zeros_like(y0)

def rhs(dY,Y,t):
    x, y, z = Y
    dY[0] = sigma * (y - x)
    dY[1] = x * (rho - z) - y
    dY[2] = x * y - beta * z

jl.rhs_py = lambda dy,y,p,t : rhs(dy,y,t)
jl.y0_ptr=y0.data.ptr
jl.dy=dy0
jl.tspan=(t0,t1)

jl.seval("""
using DifferentialEquations
using PythonCall
using CUDA
y0_p=CuPtr{Float32}(pyconvert(UInt, y0_ptr))
""")

jl.y0_arr = jl.unsafe_wrap(jl.CuArray, jl.y0_p, y0.size)

jl.seval("""
prob = ODEProblem(rhs_py,y0_arr, tspan, nothing)
sol = solve(prob,Tsit5())
ures=mapreduce(permutedims, vcat, sol.u)
""")

xyz=jl.ures
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.5)