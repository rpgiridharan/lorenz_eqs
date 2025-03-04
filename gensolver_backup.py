import numpy as np
from juliacall import Main as jl
import h5py as h5
import os
import matplotlib.pyplot as plt

class GenSolver:
    def __init__(self, solver_type, rhs, t0, initial_condition, t1, **kwargs):
        """
        Initialize the solver.
        
        Args:
            solver_type (str): Julia solver type (e.g., 'Tsit5').
            rhs (function): Right-hand side function modifying the first argument in-place.
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
        self.rhs = rhs
        self.t0 = t0
        self.t1 = t1
        self.initial_condition = initial_condition
        
        # Set defaults for optional parameters
        self.dtstep = kwargs.get('dtstep', 0.01)
        self.dtsave = kwargs.get('dtsave', 0.05)
        self.datadir = kwargs.get('datadir', 'data/')
        self.params = kwargs.get('params', {})

    def run(self):
        """Run the simulation"""
        # Setup Julia environment and functions - now with correct argument order
        jl.rhs_py = lambda dy, y, p, t: self.rhs(dy, y, self.params, t)
        
        jl.seval("""
        using DifferentialEquations
        using PythonCall
        """)
        
        # Initialize solution storage
        Y_current = self.initial_condition
        
        # Calculate number of save points
        Nsave = int((self.t1 - self.t0)/self.dtsave) + 1
        
        # Calculate number of solver steps
        Nt = int((self.t1 - self.t0)/self.dtstep) + 1
        t_eval = np.linspace(self.t0, self.t1, Nt)
        
        # Create HDF5 file
        with h5.File(os.path.join(self.datadir, 'data.h5'), 'w') as fl:
            fields = fl.create_group('fields')
            # Create empty datasets will be filled incrementally
            fields.create_dataset('t', shape=(Nsave,), dtype=np.float64)
            fields.create_dataset('xyz', shape=(Nsave, len(self.initial_condition)), dtype=np.float64)
            
            # Set initial values at index 0
            fields['t'][0] = t_eval[0]
            fields['xyz'][0] = Y_current
            fl.flush()
            
            # Initialize save counter
            save_idx = 1
            next_save_time = self.t0 + self.dtsave

            # Time stepping loop
            for i in range(1, Nt):                
                # Pass current state to Julia
                jl.tspan = (t_eval[i-1], t_eval[i])
                jl.Y_current = Y_current
                
                # Solve one time step
                jl.seval(f"""
                Y0_current = pyconvert(Vector{{Float64}}, Y_current)
                prob = ODEProblem(rhs_py, Y0_current, tspan, nothing)
                sol = solve(prob, {self.solver_type}())
                final_state = sol.u[end]
                """)
                
                # Get solution
                Y_current = np.array(jl.seval("final_state"))
                
                # Check if it's time to save data
                if t_eval[i] >= next_save_time - 1e-10:
                    fields['t'][save_idx] = t_eval[i]
                    fields['xyz'][save_idx] = Y_current
                    fl.flush()
                    print(f"Saved at t = {t_eval[i]:.3f}, point {save_idx}/{Nsave-1}")
                    save_idx += 1
                    next_save_time += self.dtsave
                
                if i % 100 == 0:
                    print(f"Completed time step {i}/{Nt-1}, t = {t_eval[i]:.3f}")
        
        print(f"Solution saved to {os.path.join(self.datadir, 'data.h5')}")
        return Y_current
