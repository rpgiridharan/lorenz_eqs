import cupy as cp
import numpy as np
from juliacall import Main as jl
import os

class GenSolver:
    def __init__(self, solver_type, rhs, t0, initial_condition, t1, **kwargs):
        """Initialize solver with parameters
        Args:
            solver_type (str): Julia solver type (e.g. 'Tsit5')
            rhs (function): Right-hand side function
            t0 (float): Initial time
            initial_condition (array): Initial state
            t1 (float): Final time
            **kwargs: Additional parameters including:
                - dtstep: time step for integration
                - params: dictionary of parameters for RHS function
        """
        self.solver_type = solver_type
        self.rhs = rhs
        self.t0 = t0
        self.t1 = t1
        self.initial_condition = initial_condition
        
        # Set defaults for optional parameters
        self.dtstep = kwargs.get('dtstep', 0.01)
        self.params = kwargs.get('params', {})
        
        # Calculate number of time steps
        self.Nt = int((t1 - t0)/self.dtstep) + 1
        
        # Initialize Julia
        self._setup_julia()

    def _setup_julia(self):
        """Setup Julia environment and functions"""
        # Pass parameters to Julia
        jl.rhs_py = self.rhs
        for key, value in self.params.items():
            setattr(jl, key, value)
        
        # Define Julia wrapper
        jl.seval("""
        using DifferentialEquations
        using PythonCall

        function rhs_wrapper!(dY, Y, p, t)
            Y_py = pyconvert(Py, Y)
            p_py = pyconvert(Py, p)
            t_py = pyconvert(Py, t)
            
            # Call Python function and get numpy array result
            result = rhs_py(Y_py, p_py, t_py)
            result_numpy = pyconvert(Vector{Float64}, result)
            
            copyto!(dY, result_numpy)
            return nothing
        end
        """)

    def run(self, fl):
        """Run the simulation
        Args:
            fl: Open HDF5 file handle with write access
        """
        # Initialize solution storage
        Y_current = self.initial_condition  # GPU array
        fields = fl['fields']
        
        # Time stepping loop
        for i in range(self.Nt-1):
            t = self.t0 + i * self.dtstep
            tspan_current = (t, t + self.dtstep)
            
            # Convert GPU array to CPU for Julia
            Y_current_cpu = cp.asnumpy(Y_current)
            
            # Pass current state to Julia
            jl.tspan = tspan_current
            jl.Y_current = Y_current_cpu
            
            # Solve one time step
            jl.seval(f"""
            Y0_current = pyconvert(Vector{{Float64}}, Y_current)
            prob = ODEProblem(rhs_wrapper!, Y0_current, tspan, nothing)
            sol = solve(prob, {self.solver_type}())
            final_state = sol.u[end]
            """)
            
            # Convert Julia result back to GPU array
            Y_current = cp.array(jl.seval("final_state"))
            fields['xyz'][i+1] = cp.asnumpy(Y_current)
            fl.flush()
            
            if (i+1) % 100 == 0:
                print(f"Completed time step {i+1}/{self.Nt-1}, t = {t + self.dtstep:.3f}")
        
        return Y_current

    def rhs(self, Y, p, t):
        """Convert GPU array to CPU before passing to Julia"""
        if isinstance(Y, cp.ndarray):
            Y = cp.asnumpy(Y)
        return Y  # Return CPU array for Julia
