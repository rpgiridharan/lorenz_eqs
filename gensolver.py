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
            - dtsave: Time step for data saving and printing.
            - datadir: Directory for data storage.
            - params: Parameters for the RHS function.
            - atol: Absolute tolerance for the solver.
            - rtol: Relative tolerance for the solver.
        """
        self.solver_type = solver_type
        self.rhs = rhs
        self.t0 = t0
        self.t1 = t1
        self.initial_condition = initial_condition
        
        # Set defaults for optional parameters
        self.dtsave = kwargs.get('dtsave', 0.05)
        self.datadir = kwargs.get('datadir', 'data/')
        self.params = kwargs.get('params', {})
        self.atol = kwargs.get('atol', 1e-6)
        self.rtol = kwargs.get('rtol', 1e-3)
        
        # Create HDF5 file
        self.fl = None
        self.fields = None
        self.save_idx = 0
        self.Nsave = int((self.t1 - self.t0)/self.dtsave) + 1
        
    def _initialize_storage(self):
        """Initialize HDF5 storage for simulation results"""
        self.fl = h5.File(os.path.join(self.datadir, 'data.h5'), 'w')
        self.fields = self.fl.create_group('fields')
        self.fields.create_dataset('t', shape=(self.Nsave,), dtype=np.float64)
        self.fields.create_dataset('xyz', shape=(self.Nsave, len(self.initial_condition)), 
                                  dtype=np.float64)
        # Set initial values
        self.save_idx = 0
            
    def save_state(self, t, u):
        """Save the current state - called by Julia callback"""
        if self.save_idx < self.Nsave:
            self.fields['t'][self.save_idx] = t
            self.fields['xyz'][self.save_idx] = u
            self.fl.flush()
            print(f"Saved at t = {t:.3f}, point {self.save_idx}/{self.Nsave-1}")
            self.save_idx += 1
            return True
        return False

    def run(self):
        """Run the simulation using Julia's callback for saving"""
        # Initialize storage
        self._initialize_storage()
        
        # Initialize Julia environment
        jl.rhs_py = lambda dy, y, p, t: self.rhs(dy, y, self.params, t)
        
        # Create Python callback that will be called from Julia
        jl.py_save_callback = self.save_state
        
        # Pass parameters to Julia
        jl.dtsave = self.dtsave
        jl.atol = self.atol
        jl.rtol = self.rtol
        jl.solver_type = self.solver_type
        
        jl.seval("""
        using DifferentialEquations
        using PythonCall
        """)
        
        # Setup callback for saving at fixed intervals
        jl.seval("""
        # Create a periodic callback that will trigger exactly at the specified intervals
        saveat_cb = PeriodicCallback(dtsave) do integrator
            # Save the current state
            py_save_callback(integrator.t, integrator.u)
        end
        
        # We also need a callback for the initial point
        function init_save_affect!(integrator)
            py_save_callback(integrator.t, integrator.u)
        end
        
        # Callback that triggers at the beginning
        init_cb = DiscreteCallback(
            (u, t, integrator) -> t == integrator.sol.prob.tspan[1], 
            init_save_affect!,
            save_positions=(false, false)
        )
        
        # Combine callbacks into a callback set
        cb_set = CallbackSet(init_cb, saveat_cb)
        """)
        
        # Set the initial condition and solve
        jl.Y_current = self.initial_condition
        jl.t0 = self.t0
        jl.t1 = self.t1
        
        jl.seval("""
        tspan = (t0, t1)
        prob = ODEProblem(rhs_py, Y_current, tspan, nothing)
        
        # Select solver based on solver_type
        solver = getproperty(Main, Symbol(solver_type))()
        
        # Solve with callback and specified tolerances
        sol = solve(prob, solver, callback=cb_set, 
                  save_everystep=false, save_start=false, save_end=false,
                  abstol=atol, reltol=rtol)
                  
        final_state = sol.u[end]
        """)
        
        # Get final state
        final_state = np.array(jl.seval("final_state"))
        
        # Close the HDF5 file
        self.fl.close()
        
        print(f"Solution saved to {os.path.join(self.datadir, 'data.h5')}")
        return final_state
