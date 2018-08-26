### Library imports
from __future__ import division
from iDEA.input import InputSection, SystemSection
import numpy as np


### Run parameters
run = InputSection()
run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\)
run.time_dependence = False          #: Run time-dependent calculation
run.verbosity = 'default'            #: Output verbosity ('low', 'default', 'high')
run.save = True                      #: Save results to disk when they are generated


### System parameters
sys = SystemSection()
sys.grid = 1001                      #: Number of grid points (must be odd)
sys.xmax = 10.0                      #: Size of the system
sys.tmax = 1.0                       #: Total real time
sys.imax = 1001                      #: Number of real time iterations
sys.itmax = 2001                     #: Total imaginary time
sys.iimax = 1e5                      #: Imaginary time iterations
sys.acon = 1.0                       #: Smoothing of the Coloumb interaction
sys.interaction_strength = 1.0       #: Scales the strength of the Coulomb interaction

def v_ext(x):
    """Ground-state external potential
    """
    return 0.5*(0.25**2)*(x**2)
sys.v_ext = v_ext

def v_pert(x):
    """Perturbing potential (switched on at t=0)
    """
    return -0.01*x
sys.v_pert = v_pert


### Exact parameters
ext = InputSection()
ext.itol = 1e-12                     #: Tolerance of imaginary time propagation (Recommended: 1e-12)
ext.itol_solver = 1e-14              #: Tolerance of linear solver in imaginary time propagation (Recommended: 1e-14)
ext.rtol_solver = 1e-12              #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
