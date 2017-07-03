"""
A Dedalus script for a purely diffusive boundary layer on a slope in a 
stratified flow. The misalignment of the diffusive boundary layer (which 
enforces adiabatic boundary conditions and is set by molecular diffusion) 
and the gravitational potential drives a laminar baroclinic flow up the
slope. Phillips (1970) and Wunsch (1970) independently obtained an 
analytical solution for this flow, which is computed using the full 
nonlinear governing equations in this script. The coordinates are rotated 
such that x is in the cross slope direction (tangential to the slope) and
z is slope-normal.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 diffusive_boundary_layer_1d.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 diffusive_boundary_layer_1d.py snapshots/*.h5

The simulation should take a few process-minutes to run.

"""

# =============================================================================

import numpy as np
from mpi4py import MPI
import time
import math

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
# root = logging.root
# for h in root.handlers:
#     h.setLevel("INFO")
logger = logging.getLogger(__name__)

# Parameters
L, H = (128., 500.) # m, cross-slope distance (L) and height normal to the slope (H)
#dpdx = 1.0 # Pa (kg/(m s^2),N/m^2), constant along-channel pressure gradient
nu = 1e-3 # m^2/s, kinematic viscosity of water
N = 1e-3 # 1/s, typical buoyancy frequency for stratification in the abyss 
kappa = nu # m^2/s, thermometric diffusion
theta = np.pi/180. # radians, theta: slope angle
sint = math.sin(theta)
N2sint = (N**2.)*math.sin(theta)
noflux = -(N**2.)*math.cos(theta)
d=((4.*nu**2.)/((N**2.)*(math.sin(theta))**2.))**(1./4.) # analytical boundary layer thickness

# Create bases and domain
#x_basis = de.Fourier('x', 2**8, interval=(-L/2., L/2.), dealias=3/2) 
z_basis = de.Chebyshev('z', 1024, interval=(0., H), dealias=3/2) # "left" z bc is at z=0, "right" at z=Lz.
#domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
domain = de.Domain([z_basis], np.float64)

# 2D incompressible flow
problem = de.IVP(domain, variables=['u','b','uz','bz'])
problem.meta['u','b','uz','bz']['z']['dirichlet'] = True
problem.parameters['H'] = H
problem.parameters['nu'] = nu
problem.parameters['sint'] = sint
problem.parameters['N2sint'] = N2sint
problem.parameters['noflux'] = noflux
problem.add_equation("dt(u) - b*sint - nu*(dz(uz)) = 0")
problem.add_equation("dt(b) + u*N2sint - nu*(dz(bz)) = 0")
problem.add_equation("uz - dz(u) = 0") 
problem.add_equation("bz - dz(b) = 0") 

# boundary conditions
problem.add_bc("left(u) = 0") # no-slip condition at the bottom of the channel, z = 0
problem.add_bc("right(uz) = 0") # no-slip condition at the top of the channel, z = H
problem.add_bc("left(bz) = noflux") # impermiability condition at the bottom of the channel, z = 0  
problem.add_bc("right(bz) = 0")
#problem.add_bc("right(w) = 0", condition="nx != 0") # impermiability condition (at the wall)
#problem.add_bc("right(p) = 0", condition="nx == 0") 

# Build solver / temporal integration set up
solver = problem.build_solver(de.timesteppers.RK443) # 4th-order, 4 step. 
logger.info('Solver built')
solver.stop_sim_time = np.inf # s?
solver.stop_wall_time = 48. * 60. * 60. # 48 hr
solver.stop_iteration = np.inf
dt = 1.0 # s

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.2,
                     max_change=1.5, min_change=0.5, max_dt=5.0, threshold=0.05)
CFL.add_velocities(('u'))

# Initial conditions 
#x = domain.grid(0) 
z = domain.grid(0) 
u = solver.state['u']  
b = solver.state['b']
u['g'] = 2.0*nu/d*(np.tan(theta)**(-1.))*np.exp(-z/d)*np.sin(z/d)
b['g'] = (N**2.)*d*np.cos(theta)*np.exp(-z/d)*np.cos(z/d) 

# Random perturbations, initialized globally for same results in parallel
#gshape = domain.dist.grid_layout.global_shape(scales=1)
#slices = domain.dist.grid_layout.slices(scales=1)
#rand = np.random.RandomState(seed=42)
#noise = rand.standard_normal(gshape)[slices]
#zb, zt = z_basis.interval
#pert =  1e-10 * noise * (zt - z) * (z - zb)
#u['g'] = F * pert

# output data
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=500.0, max_writes=20)
snapshots.add_system(solver.state)

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u)*H / nu", name='Re')
flow.add_property("u", name='max_u')
#flow.add_property("abs(dx(u) + wz)", name='divergence')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('max Re = %f' %flow.max('Re'))
            logger.info('max u = %f' %flow.max('max_u'))
	    #logger.info('Energy = %f' %flow.volume_average('KE'))
            #logger.info('Divergence = %f' %flow.grid_average('divergence'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
