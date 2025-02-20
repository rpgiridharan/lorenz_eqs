import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from gensolver_gpu import GenSolver
import h5py as h5
import os

#%% Define parameters
# Create directories
datadir = "data/"
imdir = "images/"
os.makedirs(datadir, exist_ok=True)
os.makedirs(imdir, exist_ok=True)

# Time parameters
t0 = 0.0
t1 = 50.0
dtstep = 0.01

# System parameters
params = {
    'sigma': 10.0,
    'rho': 28.0,
    'beta': 8/3
}

# Define the RHS function for Lorenz equations
def rhs(Y, p, t):
    # Convert to numpy if coming from Julia
    if not isinstance(Y, cp.ndarray):
        Y = np.asarray(Y)
        x, y, z = Y
        dx = params['sigma'] * (y - x)
        dy = x * (params['rho'] - z) - y
        dz = x * y - params['beta'] * z
        return np.array([dx, dy, dz], dtype=np.float64)
    
    # Normal GPU computation
    x, y, z = Y
    dx = params['sigma'] * (y - x)
    dy = x * (params['rho'] - z) - y
    dz = x * y - params['beta'] * z
    return cp.array([dx, dy, dz], dtype=cp.float64)

# Initial conditions
initial_condition = cp.array([1.0, 1.0, 1.0])

#%% Solve system
output_file = os.path.join(datadir, 'data.h5')
fl = h5.File(output_file, 'w', libver='latest')
fl.swmr_mode = True

# Setup HDF5 datasets
fields = fl.create_group('fields')
fields.create_dataset('t', data=np.linspace(t0, t1, int((t1-t0)/dtstep)+1))
fields.create_dataset('xyz', shape=(int((t1-t0)/dtstep)+1, len(initial_condition)), dtype=np.float64)
fields['xyz'][0] = cp.asnumpy(initial_condition)

# Solvers: Tsit5, RK4, RK8, Euler
solver = GenSolver('RK4', rhs, t0, initial_condition, t1, dtstep=dtstep, params=params)
final_state = solver.run(fl)

fl.close()

#%% Plot results
with h5.File(os.path.join(datadir, 'data.h5'), 'r') as f:
    t = f['fields/t'][:]
    xyz = f['fields/xyz'][:]  # Data is already on CPU from HDF5

# Create 3D phase space plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.5, color='blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz Attractor')
plt.savefig(os.path.join(imdir, 'lorenz_attractor.png'), dpi=600, bbox_inches='tight')
plt.close(fig)

# Plot time series
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
ax1.plot(t, xyz[:, 0], color='red')
ax1.set_ylabel('x(t)')
ax2.plot(t, xyz[:, 1], color='green')
ax2.set_ylabel('y(t)')
ax3.plot(t, xyz[:, 2], color='purple')
ax3.set_ylabel('z(t)')
ax3.set_xlabel('t')
plt.tight_layout()
plt.savefig(os.path.join(imdir, 'lorenz_timeseries.png'), dpi=600, bbox_inches='tight')
plt.close(fig)

print("Plots saved")
