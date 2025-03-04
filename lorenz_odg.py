#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:09:30 2025

@author: ogurcan
"""

import numpy as np
from juliacall import Main as jl
import matplotlib.pyplot as plt

t0 = 0.0
t1 = 50.0
dtstep = 0.01

sigma = 10.0
rho = 28.0
beta = 8/3

y0 = np.array([1.0, 1.0, 1.0])
dy0=np.zeros_like(y0)

def rhs(dY,Y,t):
    x, y, z = Y
    dY[0] = sigma * (y - x)
    dY[1] = x * (rho - z) - y
    dY[2] = x * y - beta * z

jl.rhs_py = lambda dy,y,p,t : rhs(dy,y,t)
jl.y0=y0
jl.dy=dy0
jl.tspan=(t0,t1)

jl.seval("""
using DifferentialEquations
using PythonCall
rhs_py(dy,y0,Nothing,0.0)
prob = ODEProblem(rhs_py, y0, tspan, nothing)
sol = solve(prob,Tsit5())
ures=mapreduce(permutedims, vcat, sol.u)
""")

xyz=jl.ures
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.5)

# Set plot title and labels
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
