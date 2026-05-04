import numpy as np
from numba import njit

gamma0 = 1.76e11
hbar = 1.054571817e-34
muB = 9.2740100657e-24
q_e = 1.60217663e-19
mu0 = 4*np.pi*1e-7


@njit
def self_demag_xx(dx,dy,dz):
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    r = np.sqrt(dx2 + dy2 +dz2)
    mpxy, mpxz = (dx-dy) * (dx+dy), (dx-dz) * (dx+dz)

    acc = 0

    acc += -4 * (2*dx2*dx - dy2*dy - dz2*dz)
    acc +=  4 * (dx2 + mpxy) * np.sqrt(dx2 + dy2)
    acc +=  4 * (dx2 + mpxz) * np.sqrt(dx2 + dz2)
    acc += -4 * (dy2 + dz2) * np.sqrt(dy2 + dz2)
    acc += -4 * r * (mpxy + mpxz)

    acc +=  24 * dx * dy * dz *np.atan2( dy * dz, dx * r)
    acc +=  12 * (dz + dy) * dx2 * np.log(dx)

    acc +=  12 * dz * dy2 *np.log( (np.sqrt(dy2 + dz2) + dz)/dy)
    acc += -12 * dz * dx2 *np.log( (np.sqrt(dx2 + dz2) + dz))
    acc +=  12 * dz * mpxy * np.log(r + dz)
    acc +=  -6 * dz * mpxy * np.log(dx2 + dy2) 

    acc +=  12 * dy * dz2 *np.log( (np.sqrt(dy2 + dz2) + dy)/dz)
    acc += -12 * dy * dx2 *np.log( (np.sqrt(dx2 + dy2) + dy))
    acc +=  12 * dy * mpxz * np.log(r + dy)
    acc +=   -6 * dy * mpxz * np.log(dx2 + dz2) 

    return acc/(12 * np.pi * dx * dy * dz)


@njit
def newell_f_function(x,y,z):

    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    R2 = x2 + y2 + z2
    if R2<0:
        raise ValueError("How exactly did you manage to get a negative distance?")
    R = np.sqrt(R2)
    acc = 0.0
    if z > 0:
        acc+=(2 * (2*x2 - y2 -z2) * R)
        temp = x * y * z
        if temp > 0:
            acc+= (-12 * temp * np.atan2( y * z, x * R))
        
        temp2 = x2 + z2
        if y>0 and temp2>0:
            dummy = np.log( ((y + R)**2) / temp2)
            acc+= ( 3 * y * z2 * dummy)
            acc+= ( -3 * y * x2 * dummy)
            

        temp3 = x2 + y2
        if temp3>0:
            dummy = np.log( ((z + R)**2) / temp3)
            acc+=( 3 * z * y2 * dummy)
            acc +=( -3 * z * x2 * dummy)
    else:
        if x == y:
            K =  -2.45981439737106805379          
            acc +=(K * x2 * x)
            
        else:
            acc +=(2 * (2*x2 - y2) * R)

            if y>0 and x>0:
                acc +=(-6 * y * x2 * np.log( (y + R) / x))


    return acc/12

@njit
def get_Nxx(x,y,z, dx, dy, dz):
    if x==0 and y==0 and z==0:
        return self_demag_xx(dx,dy,dz)
    
    res = 0
    res +=-1*(newell_f_function(x+dx,y+dy,z+dz)+newell_f_function(x+dx,y-dy,z+dz)+newell_f_function(x+dx,y-dy,z-dz)+
	  newell_f_function(x+dx,y+dy,z-dz)+newell_f_function(x-dx,y+dy,z-dz)+newell_f_function(x-dx,y+dy,z+dz)+
	  newell_f_function(x-dx,y-dy,z+dz)+newell_f_function(x-dx,y-dy,z-dz))
    res+=2*(newell_f_function(x,y-dy,z-dz)+newell_f_function(x,y-dy,z+dz)+newell_f_function(x,y+dy,z+dz)+newell_f_function(x,y+dy,z-dz)+
	 newell_f_function(x+dx,y+dy,z)+newell_f_function(x+dx,y,z+dz)+newell_f_function(x+dx,y,z-dz)+newell_f_function(x+dx,y-dy,z)+
	 newell_f_function(x-dx,y-dy,z)+newell_f_function(x-dx,y,z+dz)+newell_f_function(x-dx,y,z-dz)+newell_f_function(x-dx,y+dy,z))
    res +=-4*(newell_f_function(x,y-dy,z)+newell_f_function(x,y+dy,z)+newell_f_function(x,y,z-dz)+
	  newell_f_function(x,y,z+dz)+newell_f_function(x+dx,y,z)+newell_f_function(x-dx,y,z))
    res += 8*newell_f_function(x,y,z);
    return res/(4 * np.pi * dx * dy * dz)

@njit
def get_Nyy(x,y,z,dx,dy,dz):
    return get_Nxx(y,z,x,dy,dz,dx)

@njit
def get_Nzz(x,y,z,dx,dy,dz):
    return get_Nxx(z,x,y,dz,dx,dy)

@njit
def newell_g_function(x, y, z):
    result_sign = 1.0
    if x<0:
        result_sign*=-1
    if y<0:
        result_sign*=-1
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    x2, y2, z2 = x**2, y**2, z**2
    R2 = x2 + y2 + z2
    if R2<0:
        raise ValueError("How exactly did you manage to get a negative distance?")
    R = np.sqrt(R2)

    res = 0
    res+=(-2 * x * y * R)
    
    if z>0:
        res+=( -z * (z2 * np.atan2(x*y,z*R)))
        
        res+=( -3 * z * (y2 * np.atan2(x*z,y*R) + x2 * np.atan2(y*z,x*R)))
        

        temp1 = x2 + y2
        if temp1>0:
            res+=(6 * x * y * z * np.log( (z + R) / np.sqrt(temp1) ) )
            
        temp2 = y2 +z2
        if temp2>0:
            res+=(y * (3*z2 - y2) * np.log( (x + R) / np.sqrt(temp2)) )
            
        temp3 = x2 +z2
        if temp3>0:
            res+=(x * (3*z2 - x2) * np.log( (y + R) / np.sqrt(temp3)) )
            
    else:
        if y>0:
            res+=( - y * y2 * np.log( (x+R) / y))
            
        if x>0:
            res+=( - x * x2 * np.log( (y+R) / x))
            

    return result_sign *res /6


@njit
def get_Nxy(x,y,z,dx,dy,dz):
    res = 0
    res += -1*(newell_g_function(x-dx,y-dy,z-dz)+newell_g_function(x-dx,y-dy,z+dz)+newell_g_function(x+dx,y-dy,z+dz)+newell_g_function(x+dx,y-dy,z-dz)+
	newell_g_function(x+dx,y+dy,z-dz)+newell_g_function(x+dx,y+dy,z+dz)+newell_g_function(x-dx,y+dy,z+dz)+newell_g_function(x-dx,y+dy,z-dz))

    res += 2*(newell_g_function(x,y+dy,z-dz)+newell_g_function(x,y+dy,z+dz)+newell_g_function(x,y-dy,z+dz)+newell_g_function(x,y-dy,z-dz)+
       newell_g_function(x-dx,y-dy,z)+newell_g_function(x-dx,y+dy,z)+newell_g_function(x-dx,y,z-dz)+newell_g_function(x-dx,y,z+dz)+
       newell_g_function(x+dx,y,z+dz)+newell_g_function(x+dx,y,z-dz)+newell_g_function(x+dx,y-dy,z)+newell_g_function(x+dx,y+dy,z))

    res += -4*(newell_g_function(x-dx,y,z)+newell_g_function(x+dx,y,z)+newell_g_function(x,y,z+dz)+newell_g_function(x,y,z-dz)+
	newell_g_function(x,y-dy,z)+newell_g_function(x,y+dy,z))

    res += 8*newell_g_function(x,y,z)

    return res/(4 * np.pi * dx * dy * dz)

@njit
def get_Nyz(x,y,z,dx,dy,dz):
    return get_Nxy(y,z,x,dy,dz,dx)

@njit
def get_Nxz(x,y,z,dx,dy,dz):
    return get_Nxy(z,x,y,dz,dx,dy)


def get_N(x,y,z,dx,dy,dz):
    return np.array([
       [get_Nxx(x,y,z,dx, dy, dz), get_Nxy(x,y,z,dx, dy, dz), get_Nxz(x,y,z,dx, dy, dz)],
       [get_Nxy(x,y,z,dx, dy, dz), get_Nyy(x,y,z,dx, dy, dz), get_Nyz(x,y,z,dx, dy, dz)],
       [get_Nxz(x,y,z,dx, dy, dz), get_Nyz(x,y,z,dx, dy, dz), get_Nzz(x,y,z,dx, dy, dz)] 
    ])


@njit
def get_real_kernel(Nx, Ny, Nz, dx, dy, dz):
    shape = (2*Nx, 2*Ny, 2*Nz)

    Nxx = np.zeros(shape)
    Nyy = np.zeros(shape)
    Nzz = np.zeros(shape)
    Nxy = np.zeros(shape)
    Nxz = np.zeros(shape)
    Nyz = np.zeros(shape)

    cx, cy, cz = Nx, Ny, Nz   # center of FFT grid

    for i in range(-Nx+1, Nx):
        for j in range(-Ny+1, Ny):
            for k in range(-Nz+1, Nz):
                ii = i + cx
                jj = j + cy
                kk = k + cz
                x = i * dx
                y = j * dy
                z = k * dz
                Nxx[ii,jj,kk] = get_Nxx(x,y,z,dx,dy,dz)
                Nyy[ii,jj,kk] = get_Nyy(x,y,z,dx,dy,dz)
                Nzz[ii,jj,kk] = get_Nzz(x,y,z,dx,dy,dz)
                Nxy[ii,jj,kk] = get_Nxy(x,y,z,dx,dy,dz)
                Nxz[ii,jj,kk] = get_Nxz(x,y,z,dx,dy,dz)
                Nyz[ii,jj,kk] = get_Nyz(x,y,z,dx,dy,dz)
    return Nxx, Nyy, Nzz, Nxy, Nxz, Nyz  

@njit
def numba_fft(arr):
    return np.fft.fftn(arr)

@njit
def numba_ifftshift(arr):
    return np.fft.ifftshift(arr)
