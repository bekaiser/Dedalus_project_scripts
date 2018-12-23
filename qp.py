# quick plots and quick stats of simulation output
# Bryan Kaiser
# 12/23/18

# make it open the previous stat files and plot all!

# includes buoyancy fluxes

snap_path = './snapshots/'
figure_path = './figures/'
stat_path = './statistics/'

Re = 840
stokes_flag = 1

import h5py
import numpy as np
import math as ma
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import sys
from scipy.fftpack import fft, dct
from decimal import Decimal
from scipy.stats import chi2
from scipy import signal
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from scipy.fftpack import fft, fftshift
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc


# =============================================================================
# functions

def cgq(u,z,Lz,Nz): # Chebyshev-Gauss Quadrature on a grid z = (0,Lz)
  w = np.pi/Nz # Chebyshev weight
  U = 0.
  for n in range(0,Nz-1):  
    U = w*u[Nz-1-n]*np.sqrt(1.-z[n]**2.) + U
  U = U*Lz/2.
  return U


# =============================================================================
# find time series length for wave averaging

onlyfiles = [f for f in listdir(snap_path) if isfile(join(snap_path, f))]
Nt = np.shape(onlyfiles)[0] # number of files (time steps)

# Get the output file number of the first file:
Nt0i = np.zeros([Nt])
for j in range(0,Nt):
 (a1,a2) = onlyfiles[j].split('_s')
 #print(a2)
 Nt0i[j] = int(a2.split('.')[0])
Nt0 = int(np.amin(Nt0i))
#print(Nt0,Nt0+Nt-1)

# import the grid
filename = snap_path + onlyfiles[0]
f1 = h5py.File(filename, 'r')
z = f1['/scales/z']['1.0'][:]
Nz = np.shape(z)[0]
x = f1['/scales/x']['1.0'][:]
Nx = np.shape(x)[0]
y = f1['/scales/y']['1.0'][:]
Ny = np.shape(y)[0]


# cell edges 
if z[Nz-1] > 25.:
 H = 30.
else:
 H = 20.
ze = np.zeros([int(Nz+1)]) 
for j in range(1,Nz):
  ze[j] = ( z[j] + z[j-1] ) / 2.0 
ze[Nz] = H

# Chebyshev nodes
z_Chby = np.cos((np.linspace(1., Nz, num=Nz)*2.-1.)/(2.*Nz)*np.pi) # Chebyshev nodes

t = np.zeros([Nt]) 
K = np.zeros([Nt]) 
P = np.zeros([Nt])
if stokes_flag != 1: 
 B = np.zeros([Nt]) 
E = np.zeros([Nt])
Ei = np.zeros([Nt])
W = np.zeros([Nt])

Ua = np.zeros([Nx,Ny,Nt])
Va = np.zeros([Nx,Ny,Nt])
Wa = np.zeros([Nx,Ny,Nt])
if stokes_flag != 1:
 Ba = np.zeros([Nx,Ny,Nt])

uzm = np.zeros([Nz,Nt])

for n in range(Nt0,Nt0+Nt):
  filename = snap_path + 'snapshots_s%i.h5' %(n)
  f = h5py.File(filename, 'r')
  t[n-Nt0] = f['/scales/sim_time'][:] 
  #print(filename)
  #print(np.shape(np.transpose(f['/tasks/tke1'][:,0,:][0,:])))

  u = f['/tasks/u'][:]
  uz = f['/tasks/uz'][:]
  v = f['/tasks/v'][:] # shape(u) = Nt,Nx,Ny,Nz
  w = f['/tasks/w'][:]
  if stokes_flag != 1:
   b = f['/tasks/b'][:]
  uzm[:,n-Nt0] = np.mean( np.mean( uz[0,:,:,:] , axis=0 ) , axis=0 )
  up =  u[0,:,:,:] - np.mean( np.mean( u[0,:,:,:] , axis=0 ) , axis=0 )
  vp =  v[0,:,:,:] - np.mean( np.mean( v[0,:,:,:] , axis=0 ) , axis=0 ) 
  wp =  w[0,:,:,:] - np.mean( np.mean( w[0,:,:,:] , axis=0 ) , axis=0 )
  if stokes_flag != 1:
   bp =  b[0,:,:,:] - np.mean( np.mean( b[0,:,:,:] , axis=0 ) , axis=0 )
  for i in range(0,Nx):
    for j in range(0,Ny):
      Ua[i,j,n-Nt0] = cgq( up[i,j,:] , z_Chby , H , Nz ) 
      Va[i,j,n-Nt0] = cgq( vp[i,j,:] , z_Chby , H , Nz ) 
      Wa[i,j,n-Nt0] = cgq( wp[i,j,:] , z_Chby , H , Nz ) 
      if stokes_flag != 1:
       Ba[i,j,n-Nt0] = cgq( bp[i,j,:] , z_Chby , H , Nz ) 

  K[n-Nt0] = cgq( np.transpose((f['/tasks/tke1'][:,0,:] +\
              f['/tasks/tke2'][:,0,:] +\
              f['/tasks/tke3'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  if stokes_flag != 1:
   B[n-Nt0] = cgq( np.transpose((f['/tasks/B1'][:,0,:] +\
              f['/tasks/B3'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  P[n-Nt0] = cgq( np.transpose((f['/tasks/P13'][:,0,:] +\
              f['/tasks/P23'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  E[n-Nt0] = cgq( np.transpose((f['/tasks/eps'][:,0,:])[0,:]) , z_Chby , H , Nz )
  Ei[n-Nt0] = cgq( np.transpose((f['/tasks/eps_iso'][:,0,:])[0,:]) , z_Chby , H , Nz )


dt = t[1:Nt]-t[0:(Nt-1)]
dKdt = np.zeros([Nt-2])
for m in range(0,Nt-2):
 dKdt[m] = (K[int(m+2)]-K[m])/(dt[m]+dt[m+1]) 

start_time = int(t[0])
end_time = int(t[Nt-1])

T = 44700.

plotname = figure_path +'tke_budget_%i_%i.png' %(start_time,end_time)
plottitle = 'z-integrated y-mean tke budget' 
fig = plt.figure()  
plt.plot(t/T,-E,'c',label="-e"); 
plt.plot(t/T,P,'g',label='P'); 
if stokes_flag != 1:
 plt.plot(t/T,B,'b',label='B'); 
 plt.plot(t/T,P+B-E,'k',label='P+B-e');
else:
 plt.plot(t/T,P-E,'k',label='P+B-e');
plt.plot(t[1:Nt-1]/T,dKdt,"--r",label='dk/dt'); 
plt.xlabel("t/T"); plt.legend(loc=1); plt.ylabel("m^3/s^3"); 
plt.title(plottitle);  
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path +'u_%i_%i.png' %(start_time,end_time)
plottitle = 'z-integrated x/y-mean u' 
fig = plt.figure()  
for i in range(0,Nx):
 for j in range(0,Ny):
   plt.plot(t/T,Ua[i,j,:],'grey');
plt.plot(t/T,np.mean(np.mean(Ua,axis=0),axis=0),'k'); 
plt.xlabel("t/T"); plt.legend(loc=1); plt.ylabel("m^2/s"); 
plt.title(plottitle);  
plt.savefig(plotname,format="png"); plt.close(fig);

nu = 2e-6
omg = 2.*np.pi/T
U0 = Re*np.sqrt(nu*omg/2.)

tauw = nu*uzm[0,:] # wall shear stress
ustar = np.amax(np.sqrt(abs(tauw))) # max friction velocity
utau = np.sqrt(abs(tauw))*(tauw/abs(tauw)) # friction velocity
dtau = np.sqrt(abs(tauw))/omg # bl thickness

A = 3.
phi0 = ma.asin(A*ustar/U0)

plotname = figure_path +'tauw_%i_%i.png' %(start_time,end_time)
plottitle = 'wall shear stress, Re=%.2f' %(Re)
fig = plt.figure()
plt.plot(omg*t/(np.pi*2.),tauw/U0**2.,'b',label=r"$\tau_w/U_0^2$");
plt.xlabel(r"$0.5\phi/\pi$",fontsize=13); 
plt.legend(loc=4,fontsize=14); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'dtau_%i_%i.png' %(start_time,end_time)
plottitle = 'boundary layer thickness, Re=%.2f' %(Re)
fig = plt.figure()
plt.plot(phi/(np.pi*2.),dtau/dl,'b',label=r"$\delta_\tau/\delta_l$");
plt.xlabel(r"$0.5\phi/\pi$",fontsize=13); 
plt.legend(loc=4,fontsize=14); 
plt.title(plottitle);
plt.savefig(plotname,format="png"); plt.close(fig);





h5_filename = stat_path + 'qstat_%i_%i.h5' %(start_time,end_time)
f2 = h5py.File(h5_filename, "w")
dset = f2.create_dataset('Nt', data=Nt, dtype='f8')
dset = f2.create_dataset('t', data=t, dtype='f8')
dset = f2.create_dataset('dKdt', data=dKdt, dtype='f8')
dset = f2.create_dataset('P', data=P, dtype='f8')
dset = f2.create_dataset('E', data=E, dtype='f8')
dset = f2.create_dataset('Ei', data=Ei, dtype='f8')
dset = f2.create_dataset('Ua', data=Ua, dtype='f8')
dset = f2.create_dataset('Va', data=Va, dtype='f8')
dset = f2.create_dataset('Wa', data=Wa, dtype='f8')
dset = f2.create_dataset('uzm', data=uzm, dtype='f8')
if stokes_flag != 1:
 dset = f2.create_dataset('Ba', data=Ba, dtype='f8')
 dset = f2.create_dataset('B', data=B, dtype='f8')

print('\nTKE budget plotted and stats computed and written to file' + h5_filename + '.\n')


