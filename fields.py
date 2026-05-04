import numpy as np
from numba import njit
import utilities as utils
mu0 = 4*np.pi*1e-7

def make_field_external(B0, ux, uy, uz):
    if B0 is None:
        return utils.make_noField(1)

    bx,by,bz = B0*ux, B0*uy, B0*uz

    if utils.is_constant(bx) and utils.is_constant(by) and utils.is_constant(bz):
        bx0 = float(bx.flat[0])
        by0 = float(by.flat[0])
        bz0 = float(bz.flat[0])

        @njit(inline="always")
        def field_external(i):
            return bx0, by0, bz0, bx0, by0, bz0
    else: 
        @njit(inline = "always")
        def field_external(i):
            return bx[i],by[i],bz[i],bx[i],by[i],bz[i]
    return field_external


def make_field_uniaxial(Bk,kx0,ky0,kz0,onS):
    if Bk is None:
        return utils.make_noField(7)


    if utils.is_constant(Bk) and utils.is_constant(kx0) and utils.is_constant(kz0) and utils.is_constant(ky0):
        Bk_i = float(Bk[0])
        kxi,kyi,kzi = float(kx0[0]),float(ky0[0]),float(kz0[0])
        @njit(inline = "always")
        def field_anisotropy_uniaxial(Sx,Sy,Sz,Lx,Ly,Lz,i):
    
            BL = Bk_i*(        kxi*Lx + kyi*Ly + kzi*Lz)
            BS = Bk_i*(onS[i]*(kxi*Sx + kyi*Sy + kzi*Sz))
            return BS*kxi,BS*kyi,BS*kzi,BL*kxi,BL*kyi,BL*kzi  

    else:
        @njit(inline = "always")
        def field_anisotropy_uniaxial(Sx,Sy,Sz,Lx,Ly,Lz,i):
            Bk_i = Bk[i]
            kxi,kyi,kzi = kx0[i],ky0[i],kz0[i]

            BL = Bk_i*(        kxi*Lx + kyi*Ly + kzi*Lz)
            BS = Bk_i*(onS[i]*(kxi*Sx + kyi*Sy + kzi*Sz))
            return BS*kxi,BS*kyi,BS*kzi,BL*kxi,BL*kyi,BL*kzi  
    return field_anisotropy_uniaxial


def make_field_exchange(Bex):
    if Bex is None:
        return utils.make_noField(4, outNum=3)
    if utils.is_constant(Bex):
        Bex_i = float(Bex[0])
        @njit(inline = "always")
        def field_exchange(lap_x, lap_y, lap_z,i):
            return Bex_i*lap_x,Bex_i*lap_y,Bex_i*lap_z
    else:
        @njit(inline = "always")
        def field_exchange(lap_x, lap_y, lap_z,i):
            return Bex[i]*lap_x,Bex[i]*lap_y,Bex[i]*lap_z
    return field_exchange


def make_field_oersted(BOe, ux = None, uy = None, uz = None):
    if ux is None:
        ux = np.zeros_like(BOe)
    if uy is None:
        uy = np.ones_like(BOe)
    if uz is None:
        uz = np.zeros_like(BOe)
    
    
    if BOe is None:
        return utils.make_noField(2)
    bx, by, bz = BOe*ux, BOe*uy, BOe*uz

    if utils.is_constant(bx) and utils.is_constant(by) and utils.is_constant(bz):
        bx0 = float(bx.flat[0])
        by0 = float(by.flat[0])
        bz0 = float(bz.flat[0])

        @njit(inline="always")
        def field_oersted(I,i):
            return bx0*I, by0*I, bz0*I, bx0*I, by0*I, bz0*I
    else: 
        @njit(inline = "always")
        def field_oersted(I, i):
            return bx[i]*I, by[i]*I, bz[i]*I, bx[i]*I, by[i]*I, bz[i]*I
    return field_oersted


def make_field_spinorbit(Bso, Ms_S, Ms_L, gL, gS):
    BS = Bso * Ms_L / gL
    BL = Bso * Ms_S / gS

    if Bso is None:
        return utils.make_noField(7)

    @njit(inline = "always")
    def field_spinorbit(Sx, Sy, Sz, Lx, Ly, Lz, i):
        BSi = BS[i]
        BLi = BL[i]
        return BSi*Lx,BSi*Ly,BSi*Lz, BLi*Sx, BLi*Sy, BLi*Sz
    return field_spinorbit


def make_field_fieldlike(Bfl, ux, uy, uz):
    #Note that this works for both the spin and the orbital contributions, as long as you are careful with the inputs
    bx, by, bz = Bfl*ux, Bfl*uy, Bfl*uz
    @njit(inline = "always")
    def field_fieldlike(I,i):
        return I*bx[i],I*by[i],I*bz[i]
    return field_fieldlike


def make_field_dampinglike(Bdl, ux, uy, uz):
    #Note that this works for both the spin and the orbital contributions, as long as you are careful with the inputs
    Bx, By, Bz = Bdl * ux, Bdl * uy, Bdl * uz 
    @njit(inline = "always")
    def field_dampinglike(I, Jx, Jy, Jz, i):
        bx, by, bz = Bx[i]*I, By[i]*I, Bz[i]*I
        cx, cy, cz = bz*Jy - by*Jz,-bz*Jx + bx*Jz,by*Jx - bx*Jy
        return cx, cy, cz
    return field_dampinglike

def make_field_demag_from_tensor(Nx, Ny, Nz, MS, ML):
    Bs = mu0 * MS
    Bl = mu0 * ML
    @njit(inline = "never")
    def update_field_demag(Sx, Sy, Sz, Lx, Ly, Lz):
        Mx = Bs * Sx + Bl * Lx
        My = Bs * Sy + Bl * Ly
        Mz = Bs * Sz + Bl * Lz

        return - Mx * Nx, - My * Ny, - Mz * Nz
    return update_field_demag


def make_field_demag(Kxx, Kyy, Kzz, Kxy, Kxz, Kyz, MS, ML, Nx, Ny, Nz, convolution = True):
    if not convolution:
        return make_field_demag_from_tensor(Nx, Ny, Nz, MS, ML)
    shape = Kxx.shape


    @njit(inline = "never")
    def update_field_demag(Sx, Sy, Sz, Lx, Ly, Lz):

        Mx_k = np.empty(shape, dtype=np.complex128)
        My_k = np.empty(shape, dtype=np.complex128)
        Mz_k = np.empty(shape, dtype=np.complex128)

        Mx_R = np.zeros(shape, dtype=np.float64)
        My_R = np.zeros(shape, dtype=np.float64)
        Mz_R = np.zeros(shape, dtype=np.float64)

        Bk1 = np.empty(shape, dtype=np.complex128)
        Bk2 = np.empty(shape, dtype=np.complex128)
        Bk3 = np.empty(shape, dtype=np.complex128)

        # optional: reusable outputs
        Bx = np.empty(shape, dtype=np.complex128)
        By = np.empty(shape, dtype=np.complex128)
        Bz = np.empty(shape, dtype=np.complex128)

        
        Mx_R[:Nx, :Ny, :Nz] = (MS*Sx + ML*Lx).reshape(Nx,Ny,Nz)
        My_R[:Nx, :Ny, :Nz] = (MS*Sy + ML*Ly).reshape(Nx,Ny,Nz)
        Mz_R[:Nx, :Ny, :Nz] = (MS*Sz + ML*Lz).reshape(Nx,Ny,Nz)

        Mx_k = np.fft.fftn(Mx_R)
        My_k = np.fft.fftn(My_R)
        Mz_k = np.fft.fftn(Mz_R)

        # --- Bx ---
        Bk1[:] = np.multiply(Kxx, Mx_k)
        Bk1+= Kxy * My_k
        Bk1+= Kxz * Mz_k
        Bx = np.fft.ifftn(Bk1).real

        # --- By ---
        Bk2[:] = np.multiply(Kxy, Mx_k)
        Bk2+= Kyy * My_k
        Bk2+= Kyz * Mz_k
        By = np.fft.ifftn(Bk2).real


        # --- Bz ---
        Bk3[:] = np.multiply(Kxz, Mx_k)
        Bk3+= Kyz * My_k
        Bk3+= Kzz * Mz_k
        Bz = np.fft.ifftn(Bk3).real


        return (
            -Bx[:Nx, :Ny, :Nz].ravel(),
            -By[:Nx, :Ny, :Nz].ravel(),
            -Bz[:Nx, :Ny, :Nz].ravel(),
        )

    return update_field_demag



