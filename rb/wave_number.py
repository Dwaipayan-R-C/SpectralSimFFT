import numpy as np

class WaveRayleigh:    
    def __init__(self, l_lim,h_lim):
        self.low = l_lim
        self.high = h_lim
        self.k = np.linspace(0.8, 7, 100)         
        
    def RaReal(self):
        return (28*self.k**6+952*self.k**4+20832*self.k**2+141120)/(27*self.k**2)

    def RaSimple(self):
        return [self.k,[(np.pi**2 + k**2)**3 / k**2 for k in self.k]]

class StaticRayleigh:
    def __init__(self,length,n,θn):
        self.length = length        
        self.n = n
        self.kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.length,self.length/(self.length-1)))
        self.ky=self.kx.copy()
        self.Kx,self.Ky=np.meshgrid(self.kx, self.ky)
        self.x,self.y,self.z=np.linspace(0, self.length, self.length),np.linspace(0, self.length, self.length),np.linspace(0, 1, self.length)
        self.X,self.Y=np.meshgrid(self.x,self.y)
        self.XZ,self.ZX=np.meshgrid(self.x,self.z)
        self.θn=θn
           
    def Ansatz(self,R,km):
        kxc = 2 * np.pi / self.length * km
        kzc = 2 * np.pi / self.length * self.n
        wn=R* kzc**2 / (-self.n**2 * np.pi**2 - kzc**2)**2
        wz = wn * np.sin(self.n * self.ZX*np.pi)
        θ_xz = self.θn*np.sin(np.pi * self.n * self.ZX)
        θ_xy = self.θn*np.sin(np.pi * self.n * 0.5)
        U = -np.real(wz * np.sin(kxc * self.X))
        V = np.real(wz * np.cos(kxc * self.Y))
        W = np.real(wz * np.cos(kxc * self.X))
        UZ = -np.real(wz * np.sin(kxc * self.XZ))
        VZ = np.real(wz * np.cos(kxc * self.XZ))
        WZ = np.real(wz * np.cos(kxc * self.XZ))
        θxy = np.real(θ_xy * np.cos(kxc * self.X))
        θxz = np.real(θ_xz * np.cos(kxc * self.XZ))
        return [U,V,W,UZ,VZ,WZ,θxy,θxz,self.X,self.Y,self.XZ,self.ZX]
    
class DynamicRayleigh:
        
    def __init__(self,Rayleigh,Nx,Ny,Nz,Lx,Ly,Lz,n,timescale,θn,dt):
        self.Rayleigh = Rayleigh 
        self.Nx=Nx
        self.Nz=Nz
        self.Ny=Ny
        self.Lx=Lx
        self.Ly=Ly
        self.Lz=Lz
        self.timescale=timescale
        self.n=n
        self.θn=θn        
        self.Kx, self.Ky = np.meshgrid(2 * np.pi * np.fft.fftfreq(self.Nx, Lx / Nx),
                                       2 * np.pi * np.fft.fftfreq(self.Ny, Ly / Ny))
        self.K_2=self.Kx**2+self.Ky**2
        self.dt = dt
        
    def theta_dot(self,θxy,u_xy,v_xy,w_xy):
        
        # Real space θ
        # All the variables until now is in Fourier space as we define it in wave vectors 
        θ_rp = np.real(np.fft.ifftshift(θxy))        
        
        # Calculate ∇θ(r,t)
        grad_θx, grad_θy = np.gradient(θ_rp)
        
        # Real space velocities
        u_rp = np.real(np.fft.ifft2(u_xy))
        v_rp = np.real(np.fft.ifft2(v_xy))
        w_rp = np.real(np.fft.ifft2(w_xy))
        
        # Calculate u(r,t)
        # Calculate N in x and y directions
        N_rp = u_rp * grad_θx + v_rp * grad_θy

        # Calculate N(x,y,z,t) in Fourier space
        N_z_rp = w_rp * np.fft.ifft2(-1j * self.Kx * θxy).real
        N_rp += N_z_rp
        N_xy = np.fft.fft2(N_rp)
        
        # Calculate Time marching of θxy using N_xy = u(r,t)⋅∇θ(r,t) 
        θxy += self.dt*(-(28/3 + self.K_2) * θxy - (7/63) * w_xy - 1/140*N_xy)
        return θxy,u_xy,v_xy,w_xy
        
    def velocity(self,θxy,u_xy,v_xy,w_xy):
        
        # Masking to avoid zero division
        mask = (self.K_2**2 > 0) & (self.K_2 < 1e3)
        
        # Calculate x,y,z component of velocity here
        u_xy[mask] = (self.Rayleigh * 1j * self.Kx[mask] * θxy[mask]) / ((self.K_2[mask]**2) / 2 + 12 * self.K_2[mask])  
        v_xy[mask] = (self.Rayleigh * 1j * self.Ky[mask] * θxy[mask]) / ((self.K_2[mask]**2) / 2 + 12 * self.K_2[mask])    
        w_xy[mask] = -((self.Rayleigh*self.K_2[mask] * θxy[mask])/30)/(((self.K_2[mask]**2) / 140) + (2 * self.K_2[mask] / 15) + 4 )

        # velocity in real space
        u = np.real(np.fft.ifft2(u_xy))
        v = np.real(np.fft.ifft2(v_xy))
        w = np.real(np.fft.ifft2(w_xy)) 
        
        return u,v,w,u_xy,v_xy,w_xy
    
    