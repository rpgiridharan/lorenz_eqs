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
        
        # Calculate save times outside of the h5.File context
        t_save = np.linspace(self.t0, self.t1, Nsave)

        # Calculate number of solver steps
        Nt = int((self.t1 - self.t0)/self.dtstep) + 1
        
        # Create HDF5 file
        with h5.File(os.path.join(self.datadir, 'data.h5'), 'w') as fl:
            fields = fl.create_group('fields')
            fields.create_dataset('t', data=t_save)
            fields.create_dataset('xyz', shape=(Nsave, len(self.initial_condition)), dtype=np.float64)
            fields['xyz'][0] = Y_current
            
            # Time stepping loop
            next_save_idx = 1  # Index for the next save point
            next_save_time = self.t0 + self.dtsave  # Time for the next save
            
            for i in range(Nt-1):
                t = self.t0 + i * self.dtstep
                t_next = t + self.dtstep
                tspan_current = (t, t_next)
                
                # Pass current state to Julia
                jl.tspan = tspan_current
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
                
                # Save data when we reach a save time (exact match)
                if t_next == next_save_time:
                    fields['xyz'][next_save_idx] = Y_current
                    fl.flush()
                    next_save_idx += 1
                    next_save_time = self.t0 + next_save_idx * self.dtsave
                    print(f"Saved at t = {t_next:.3f}, point {next_save_idx-1}/{Nsave-1}")
                
                if (i+1) % 100 == 0:
                    print(f"Completed time step {i+1}/{Nt-1}, t = {t_next:.3f}")
        
        print(f"Solution saved to {os.path.join(self.datadir, 'data.h5')}")
        return Y_current
