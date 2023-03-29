import numpy as np
import matplotlib.pyplot as plt
from rb import Rayleigh_functions as RayFunc
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


def RayleighK(low,high,plot=False):
    """Code for Rayleigh number vs wave vector

    Args:
        low (float): lower limit
        high (float): higher limit
        plot (bool, optional): For plotting. Defaults to False.
    """
    if (plot==True):
        fig1=plt.figure(1)
        k,Ra=RayFunc.WaveRayleigh(low,high).RaSimple()
        Ra_real = RayFunc.WaveRayleigh(low,high).RaReal()
        kc=k[np.argmin(Ra)]
        Rc=np.min(Ra)
        print("Critical Rayleigh number: ", Rc)
        print("Critical wavenumber: ", kc)
        plt.xlabel('Wave vector K')
        plt.ylabel('Rayleigh numbers')
        plt.plot(k,Ra,color='red')
        plt.plot(k,Ra_real,color='black')
        plt.ylim(0,5000)
        plt.scatter(kc,Rc,color='black',alpha=.9)
        plt.text(4,3500,f'km={np.round(kc,3)},\n Rc={np.round(Rc,2)} ' ,ha='center', va='center',fontsize=10,  bbox=dict(facecolor='red', alpha=0.5) )
        plt.legend(["Linear","Non-linear"])
        plt.title('Critical Rayleigh')
        plt.tight_layout()
        plt.show()


def StaticSim(length,n,θn,low,high, plotStatic=False):
    """For plotting the static simulation

    Args:
        length (int): Domain length
        n (int): number of basis functions
        low (float): lower limit
        high (float): higher limit
        plotStatic (bool, optional): For plotting. Defaults to False.
    """
    k,Ra=RayFunc.WaveRayleigh(low,high).RaSimple()
    km = k[np.argmin(Ra)+1]
    Rm = Ra[np.argmin(Ra)+1]
    U,V,W,UZ,VZ,WZ,θxy,θxz,X,Y,XZ,ZX = RayFunc.StaticRayleigh(length,n,θn).Ansatz(Rm,km)
    
    if(plotStatic==True):  
        
        # Plot XZ  
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 6))
        stream = ax1.streamplot(X[:,0:45], Y[:,0:45], U[:,0:45], V[:,0:45], color='k', linewidth=0.5, density=1.)#, arrowstyle='->', arrowsize=0.5)
        temp = ax2.contourf(X[:,0:45], Y[:,0:45], θxy[:,0:45], cmap='inferno', levels=np.linspace(-1, 1, 50))
        # temp = ax2.contourf(X[:,0:23], Y[:,0:23], Theta[:,0:23], cmap='inferno', levels=np.linspace(-1, 1, 21))
        color=plt.colorbar(temp)
        fig.suptitle('Convective pattern at XY plane')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        fig.tight_layout()
        
        # Plot XY
        fig1, (ax3, ax4) = plt.subplots(1,2,figsize=(12, 6))
        stream = ax3.streamplot(XZ[:,0:45], ZX[:,0:45], UZ[:,0:45], WZ[:,0:45], color='k', linewidth=0.5, density=1.)#, arrowstyle='->', arrowsize=0.5)
        temp = ax4.contourf(XZ[:,0:45], ZX[:,0:45], θxz[:,0:45], cmap='inferno', levels=np.linspace(-1, 1, 50))
        # temp = ax4.contourf(X[:,0:23], Y[:,0:23], Theta[:,0:23], cmap='inferno', levels=np.linspace(-1, 1, 21))
        color=plt.colorbar(temp)
        fig1.suptitle('Convective pattern at XZ plane')
        ax3.set_xlabel('x')
        ax3.set_ylabel('z')
        ax4.set_xlabel('x')
        ax4.set_ylabel('z')
        fig1.tight_layout()
        
        plt.show()
       
    
def DynamicSim(Ra,Nx,Ny,Nz,Lx,Ly,Lz,n,timescale,θn,dt,save_every, anim=False,plots=True):
    """For simulating the time marching algorithm for non-linear system

    Args:
        Ra (float): Rayleigh number
        Nx (float): Number of nodes along x
        Ny (float): Number of nodes along y
        Nz (float): Number of nodes along z
        Lx (int): Domain length along x
        Ly (int): Domain length along y
        Lz (int): Domain length along z
        n (int): number of basis functions
        timescale (int): timescale
        dt (float): timestep
        save_every (int): save every for plotting purpose
        anim (bool, optional): For Thermal plot. Defaults to False.
        plots (bool, optional): For velocity plot. Defaults to True.
    """
    
    global θxy,u_xy,v_xy,w_xy,u,v,w,anim_im,anim_im1, anim_u,anim_v,anim_w
    
    
    # Initialize Variables to plot (Fourier space)
    θxy=np.zeros((Nx, Ny), dtype=np.complex128)
    u_xy=np.copy(θxy)
    v_xy=np.copy(θxy)
    w_xy=np.copy(θxy)
    θxy = np.fft.fftshift(np.fft.fft2(np.random.rand(Nx, Ny)))
    RayDyn = RayFunc.DynamicRayleigh(Ra,Nx,Ny,Nz,Lx,Ly,Lz,n,timescale,θn,dt)
    
    #region Plot_definition
    # Theta Plot definition
    fig_3,(ax3,ax4) = plt.subplots(1,2,figsize=(8, 4))
    ax3.set_title('θ(r,t)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax4.set_title('θ_(kx,ky)')
    ax4.set_xlabel('kx')
    ax4.set_ylabel('ky')
    anim_im = ax3.imshow(np.real(np.fft.ifft2(θxy)), interpolation='bilinear', origin='lower', cmap='inferno', animated=True, aspect='auto')
    anim_im1 = ax4.imshow(np.real(θxy), interpolation='bilinear', origin='lower', cmap='inferno', animated=True, aspect='auto')
    plt.colorbar(anim_im,ax=ax3)
    fig_3.tight_layout()
    fig_3.suptitle(f'Temp(θ) vs Time(t) at Ra={Ra}')    
    
    # Velocity plot definition
    fig_4,(ax5,ax6,ax7) = plt.subplots(1,3,figsize=(14, 5))
    ax5.set_title('u(r,t)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax6.set_title('v(r,t)')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax7.set_title('w(r,t)')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    fig_4.suptitle(f'Velocity marching at Ra={Ra}')
    u = np.real(np.fft.ifft2(u_xy))
    v = np.real(np.fft.ifft2(v_xy))
    w = np.real(np.fft.ifft2(w_xy))
    anim_u = ax5.imshow(u, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
    anim_v = ax6.imshow(v, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
    anim_w = ax7.imshow(w, interpolation='bilinear', origin='lower', cmap='plasma', animated=True, aspect='auto')
    plt.colorbar(anim_u,ax=ax6)
    fig_4.tight_layout()    
    #endregion 
    
    # Plot theta variant
    def updateθ(t):
        print(f'{t+1}//{timescale}', end="\r")
        global θxy,u_xy,v_xy,w_xy,anim_im,anim_im1               
        θxy,u_xy,v_xy,w_xy = RayDyn.theta_dot(θxy,u_xy,v_xy,w_xy)
        u,v,w,u_xy,v_xy,w_xy = RayDyn.velocity(θxy,u_xy,v_xy,w_xy)
        anim_im.set_data(np.real(np.fft.ifft2(θxy)))
        anim_im1.set_data(np.fft.fftshift(np.real(θxy)))
        anim_im.autoscale()
        anim_im1.autoscale()
        fig_3.suptitle(f'Temp(θ) vs Time(t) at Ra={Ra} and t={t*dt}') 
        if (t%save_every==0):
            fig_3.savefig(f'plots/Rayleigh/temp/t_{t}.png')
    
    # Plot velocity variant
    def update_velocity(t):
        print(f'{t+1}//{timescale}', end="\r")  
        global θxy,u_xy,v_xy,w_xy,u,v,w,anim_u,anim_v,anim_w               
        θxy,u_xy,v_xy,w_xy = RayDyn.theta_dot(θxy,u_xy,v_xy,w_xy)
        u,v,w,u_xy,v_xy,w_xy = RayDyn.velocity(θxy,u_xy,v_xy,w_xy)
        anim_u.set_data(u)
        anim_v.set_data(v)
        anim_w.set_data(w)
        anim_u.autoscale()
        anim_v.autoscale()
        anim_w.autoscale()
        fig_4.suptitle(f'Velocity marching at Ra={Ra}and t={t*dt}')
        if (t%save_every==0):
            fig_4.savefig(f'plots/Rayleigh/velocity/t_{t}.png')
      
    if(anim==True):
        ani_t = FuncAnimation(fig_3, updateθ, repeat=True,frames=70, interval=500) 
        ani_t.save('plots/Rayleigh/temp/temp_ani.gif')   
    elif(plots==True):
        ani_v = FuncAnimation(fig_4, update_velocity, frames=range(timescale), interval=1, repeat=False, cache_frame_data=True)        
    # plt.show() 
            
    
    