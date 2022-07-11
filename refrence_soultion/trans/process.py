import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py


# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/grid/x'][:]
    t = f['tally/grid/t'][:]

dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
dt    = t[1:] - t[:-1]
K     = len(dt)
J     = len(x_mid)

data = np.load('azurv1_pl.npz')
phi_x_ref = data['phi_x']
phi_t_ref = data['phi_t']
phi_ref = data['phi']

#data2 = np.load('output.npz')
my_sf = np.load('t0.npy')#data['arr_0']
my_xx = np.linspace(-20,20,81)#len(scalar_flux))
# =============================================================================
# Animate results
# =============================================================================



with h5py.File('output.h5', 'r') as f:
    phi      = f['tally/flux/mean'][:]
    phi_sd   = f['tally/flux/sdev'][:]
    phi_x    = f['tally/flux-x/mean'][:]
    phi_x_sd = f['tally/flux-x/sdev'][:]
    phi_t    = f['tally/flux-t/mean'][:]
    phi_t_sd = f['tally/flux-t/sdev'][:]
for k in range(K):
    phi[k]      /= (dx*dt[k])
    phi_sd[k]   /= (dx*dt[k])
    phi_x[k]    /= (dt[k])
    phi_x_sd[k] /= (dt[k])
    phi_t[k]    /= (dx)
    phi_t_sd[k] /= (dx)
phi_t[K]    /= (dx)
phi_t_sd[K] /= (dx)


for i in range(1):
    plt.figure(1)
    #plt.plot(x_mid,phi[i,:],'-b', my_xx)
    #ref_plot = 
    scalaing_factor = 1 #max(my_sf[:,i]) / max(phi_ref[i,:])
    
    plt.plot(my_xx, my_sf[:,i],'-k')
    plt.plot(x_mid, phi_ref[i,:], '--r')
    plt.plot()
    plt.ylim([0,1.25])
    plt.xlim([-22,22])
    plt.title(i)
    plt.show()

print(np.max(phi_ref))
#for i in range(20):
#    my_sf[:,i] = my_sf[:,i]/max(my_sf[:,1])

# Flux - average
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-22, 22), ylim=(0, 1.25))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{k,j}$')
#line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="AZURV1")
line3, = ax.plot([], [],'-k',label="TNT")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):
    scalaing_factor = max(my_sf[:,k]) / max(phi_ref[k,:])
    
    #line1.set_data(x_mid,phi[k,:])
    #ax.collections.clear()
    #ax.fill_between(x_mid,phi[k,:]-phi_sd[k,:],phi[k,:]+phi_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x_mid,phi_ref[k,:])
    line3.set_data(my_xx, my_sf[:,k])
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line2, line3, text        #line1, 
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=3)

simulation.save('slab_reactor.gif')

#anim.save('wave.mp4', dpi=200, fps=30, writer='ffmpeg')
'''
# Flux - x
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{k}(x)$')
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
line3, = ax.plot([], [],'-k',label="TNT.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    line1.set_data(x,phi_x[k,:])
    ax.collections.clear()
    ax.fill_between(x,phi_x[k,:]-phi_x_sd[k,:],phi_x[k,:]+phi_x_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x,phi_x_ref[k,:])
    line3.set_data(my_xx, my_sf[k,:])
    text.set_text(r'$t \in [%.1f,%.1f]$ s'%(t[k],t[k+1]))
    return line1, line2, line3, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()

# Flux - t
fig = plt.figure(figsize=(6,4))
ax = plt.axes(xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax.grid()
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Flux')
ax.set_title(r'$\bar{\phi}_{j}(t)$')
line1, = ax.plot([], [],'-b',label="MC")
line2, = ax.plot([], [],'--r',label="Ref.")
line3, = ax.plot([], [],'-k',label="TNT.")
text   = ax.text(0.02, 0.9, '', transform=ax.transAxes)
ax.legend()        
def animate(k):        
    k += 1
    line1.set_data(x_mid,phi_t[k,:])
    ax.collections.clear()
    ax.fill_between(x_mid,phi_t[k,:]-phi_t_sd[k,:],phi_t[k,:]+phi_t_sd[k,:],alpha=0.2,color='b')
    line2.set_data(x_mid,phi_t_ref[k-1,:])
    line3.set_data(my_xx, my_sf[k,:])
    text.set_text(r'$t = %.1f$ s'%(t[k]))
    return line1, line2, line3, text        
simulation = animation.FuncAnimation(fig, animate, frames=K)
writervideo = animation.FFMpegWriter(fps=6)
plt.show()
'''
