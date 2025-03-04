import numpy as np
from juliacall import Main as jl
import h5py as h5
import os

def rhs(dY, Y, params, t):
    x, y, z = Y
    dY[0] = params['sigma'] * (y - x)
    dY[1] = x * (params['rho'] - z) - y
    dY[2] = x * y - params['beta'] * z
    
class GenSolver:
    def __init__(self, solver_type, rhs, t0, initial_condition, t1, **kwargs):
        """
        Initialize the GPU-accelerated solver using Julia's CUDA support.
        
        Args:
            solver_type (str): Julia solver type (e.g., 'Tsit5').
            rhs (function): Right-hand side function that modifies dY in-place.
            t0 (float): Start time.
            initial_condition (array): Initial state.
            t1 (float): End time.
            **kwargs: Additional parameters:
            - dtstep: Time step for solver integration.
            - dtsave: Time step for data saving (can be different from dtstep).
            - datadir: Directory for data storage.
            - params: Parameters for the RHS function.
        """
        self.solver_type = solver_type
        self.rhs = rhs  # Store the Python RHS function
        self.t0 = t0
        self.t1 = t1
        
        # Convert initial condition to NumPy array for Julia transfer
        if not isinstance(initial_condition, np.ndarray):
            self.initial_condition = np.array(initial_condition)
        else:
            self.initial_condition = initial_condition
            
        # Set defaults for optional parameters
        self.dtstep = kwargs.get('dtstep', 0.01)
        self.dtsave = kwargs.get('dtsave', 0.05)
        self.datadir = kwargs.get('datadir', 'data/')
        self.params = kwargs.get('params', {})
        
        # Initialize Julia environment with CUDA support
        self._setup_julia()

    def _setup_julia(self):
        """Set up the Julia environment with required packages for GPU computation"""
        jl.seval("""
        using DifferentialEquations
        using PythonCall
        using CUDA
        using DiffEqGPU
        """)
        
        # Check if CUDA is available in Julia
        cuda_available = jl.seval("CUDA.functional()")
        if not cuda_available:
            raise RuntimeError("Julia CUDA support is not available. Please check your CUDA installation.")
        
        # Pass the parameters and RHS function directly to Julia
        jl.params_dict = self.params
        jl.rhs_py = self.rhs
        
        # Create a GPU-compatible wrapper function
        jl.seval("""
        # GPU-compatible wrapper: runs CPU version of the Python function
        function gpu_rhs_wrapper!(du, u, p, t) 
            # Convert GPU arrays to CPU
            u_cpu = Array(u)
            du_cpu = similar(u_cpu)
            
            # Call Python function on CPU
            rhs_py(du_cpu, u_cpu, params_dict, t)
            
            # Copy results back to GPU
            du .= du_cpu
            return nothing
        end
        """)

    def run(self):
        """Run the simulation with GPU acceleration in Julia"""
        # Pass initial state and parameters to Julia
        jl.Y0 = self.initial_condition
        jl.tspan = (self.t0, self.t1)
        jl.dtsave = self.dtsave
        
        # Calculate number of save points
        Nsave = int((self.t1 - self.t0)/self.dtsave) + 1
        
        # Create HDF5 file
        with h5.File(os.path.join(self.datadir, 'data.h5'), 'w') as fl:
            fields = fl.create_group('fields')
            # Create empty datasets that will be filled with results
            fields.create_dataset('t', shape=(Nsave,), dtype=np.float64)
            fields.create_dataset('xyz', shape=(Nsave, len(self.initial_condition)), dtype=np.float64)
            
            print("Setting up Julia GPU solver...")
            
            # Create a GPU-accelerated problem in Julia
            jl.seval("""
            # Convert initial condition to CUDA array (with appropriate precision)
            Y0_cuda = CuArray(Float32.(Y0))
            
            # Create the ODE problem using our GPU wrapper that calls Python
            prob_gpu = ODEProblem(gpu_rhs_wrapper!, Y0_cuda, tspan)
            
            # Pre-compile the solver with a short test run
            test_sol = solve(prob_gpu, """ + self.solver_type + """(), 
                            saveat=[tspan[1]], save_everystep=false)
            
            # Ensure compilation completes
            wait(CUDA.@sync test_sol)
            """)
            
            # Solve the full problem on GPU
            print("Solving with Julia GPU acceleration...")
            jl.seval("""
            # Define save points
            save_times = tspan[1]:dtsave:tspan[2]
            
            # Solve the ODE problem
            CUDA.@sync begin
                global sol_gpu = solve(prob_gpu, """ + self.solver_type + """(), 
                                      saveat=save_times, save_everystep=false)
            end
            
            # Extract solution back to CPU for Python
            sol_array = Array(sol_gpu)
            """)
            
            # Get all results at once
            results = np.array(jl.seval("sol_array"))
            save_times = np.array(jl.seval("save_times"))
            
            print("Saving results...")
            
            # Save all results to HDF5 file
            for i, t in enumerate(save_times):
                if i < Nsave:  # Safety check
                    fields['t'][i] = t
                    fields['xyz'][i] = results[:, i].T
            
            print(f"Solution completed with Julia GPU acceleration")
            final_state = results[:, -1]  # Final state
        
        print(f"Solution saved to {os.path.join(self.datadir, 'data.h5')}")
        return final_state
