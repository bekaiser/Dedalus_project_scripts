# quick plots and quick stats of simulation output
# Bryan Kaiser
# 12/7/18

# make it open the previous stat files and plot all!

# includes buoyancy fluxes

snap_path = './snapshots/'
figure_path = './figures/'
stat_path = './statistics/'

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

# fix! Find lowest number snapshots file, loop over that 

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
B = np.zeros([Nt]) 
E = np.zeros([Nt])
Ei = np.zeros([Nt])
W = np.zeros([Nt])

Ua = np.zeros([Nx,Ny,Nt])
Va = np.zeros([Nx,Ny,Nt])
Wa = np.zeros([Nx,Ny,Nt])
Ba = np.zeros([Nx,Ny,Nt])

"""
for n in range(Nt0,Nt0+Nt):
  filename = snap_path + 'snapshots_s%i.h5' %(n)
  f = h5py.File(filename, 'r')
  print(filename)
"""

#print('data loop:')
for n in range(Nt0,Nt0+Nt):
  filename = snap_path + 'snapshots_s%i.h5' %(n)
  f = h5py.File(filename, 'r')
  t[n-Nt0] = f['/scales/sim_time'][:] 
  #print(filename)
  #print(np.shape(np.transpose(f['/tasks/tke1'][:,0,:][0,:])))

  u = f['/tasks/u'][:]
  v = f['/tasks/v'][:] # shape(u) = Nt,Nx,Ny,Nz
  w = f['/tasks/w'][:]
  b = f['/tasks/b'][:]
  up =  u[0,:,:,:] - np.mean( np.mean( u[0,:,:,:] , axis=0 ) , axis=0 )
  vp =  v[0,:,:,:] - np.mean( np.mean( v[0,:,:,:] , axis=0 ) , axis=0 ) 
  wp =  w[0,:,:,:] - np.mean( np.mean( w[0,:,:,:] , axis=0 ) , axis=0 )
  bp =  b[0,:,:,:] - np.mean( np.mean( b[0,:,:,:] , axis=0 ) , axis=0 )
  for i in range(0,Nx):
    for j in range(0,Ny):
      Ua[i,j,n-Nt0] = cgq( up[i,j,:] , z_Chby , H , Nz ) 
      Va[i,j,n-Nt0] = cgq( vp[i,j,:] , z_Chby , H , Nz ) 
      Wa[i,j,n-Nt0] = cgq( wp[i,j,:] , z_Chby , H , Nz ) 
      Ba[i,j,n-Nt0] = cgq( bp[i,j,:] , z_Chby , H , Nz ) 

  #uz = f['/tasks/uz'][:] 

  """
  um[ct,:] = np.mean( np.mean( u[m,:,:,:] , axis=0 ) , axis=0 )
  vm[ct,:] = np.mean( np.mean( v[m,:,:,:] , axis=0 ) , axis=0 )
  wm[ct,:] = np.mean( np.mean( w[m,:,:,:] , axis=0 ) , axis=0 )
  bm[ct,:] = np.mean( np.mean( b[m,:,:,:] , axis=0 ) , axis=0 ) + (N**2.0)*np.cos(tht)*z
  #uzm[ct,:] = np.mean( np.mean( uz[m,:,:,:] , axis=0 ) , axis=0 )
  #bzm[ct,:] = np.mean( np.mean( bz[m,:,:,:] , axis=0 ) , axis=0 ) + (N**2.0)*np.cos(tht)*np.ones(np.shape(z))

  up =  u[m,:,:,:] - np.mean( np.mean( u[0,:,:,:] , axis=0 ) , axis=0 )
  vp =  v[m,:,:,:] - np.mean( np.mean( v[0,:,:,:] , axis=0 ) , axis=0 ) 
  wp =  w[m,:,:,:] - np.mean( np.mean( w[0,:,:,:] , axis=0 ) , axis=0 )

  for i in range(0,Nx):
    for j in range(0,Ny):
      U[n-Nt0,i,j] = cgq( up[0,i,j,:] , z_Chby , H , Nz ) 
      V[n-Nt0,i,j] = cgq( vp[0,i,j,:] , z_Chby , H , Nz ) 
      W[n-Nt0,i,j] = cgq( wp[0,i,j,:] , z_Chby , H , Nz ) 
      Ba[n-Nt0,i,j] = cgq( bp[0,i,j,:] , z_Chby , H , Nz ) 
  """

  K[n-Nt0] = cgq( np.transpose((f['/tasks/tke1'][:,0,:] +\
              f['/tasks/tke2'][:,0,:] +\
              f['/tasks/tke3'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  B[n-Nt0] = cgq( np.transpose((f['/tasks/B1'][:,0,:] +\
              f['/tasks/B3'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  P[n-Nt0] = cgq( np.transpose((f['/tasks/P13'][:,0,:] +\
              f['/tasks/P23'][:,0,:])[0,:]) , z_Chby , H , Nz ) 
  E[n-Nt0] = cgq( np.transpose((f['/tasks/eps'][:,0,:])[0,:]) , z_Chby , H , Nz )
  Ei[n-Nt0] = cgq( np.transpose((f['/tasks/eps_iso'][:,0,:])[0,:]) , z_Chby , H , Nz )


#print(np.shape(t),Nt)
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
plt.plot(t/T,B,'b',label='B'); 
#plt.plot(t/T,K5i+K6i,'m',label='div(T)'); 
plt.plot(t/T,P+B-E,'k',label='P+B-e');
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
#plt.plot(t/T,B,'b',label='B'); 
#plt.plot(t/T,K5i+K6i,'m',label='div(T)'); 
#plt.plot(t/T,P+B-E,'k',label='P+B-e');
#plt.plot(t[1:Nt-1]/T,dKdt,"--r",label='dk/dt'); 
plt.xlabel("t/T"); plt.legend(loc=1); plt.ylabel("m^2/s"); 
plt.title(plottitle);  
plt.savefig(plotname,format="png"); plt.close(fig);







#start_time = int(t[0])
#end_time = int(t[Nt-1])

h5_filename = stat_path + 'qstat_%i_%i.h5' %(start_time,end_time)
f2 = h5py.File(h5_filename, "w")
dset = f2.create_dataset('Nt', data=Nt, dtype='f8')
dset = f2.create_dataset('t', data=t, dtype='f8')
dset = f2.create_dataset('dKdt', data=dKdt, dtype='f8')
dset = f2.create_dataset('P', data=P, dtype='f8')
dset = f2.create_dataset('E', data=E, dtype='f8')
dset = f2.create_dataset('Ei', data=Ei, dtype='f8')
dset = f2.create_dataset('B', data=B, dtype='f8')
dset = f2.create_dataset('Ua', data=Ua, dtype='f8')
dset = f2.create_dataset('Va', data=Va, dtype='f8')
dset = f2.create_dataset('Wa', data=Wa, dtype='f8')
dset = f2.create_dataset('Ba', data=Ba, dtype='f8')

print('\nTKE budget plotted and stats computed and written to file' + h5_filename + '.\n')


