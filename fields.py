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


## DEPRACATED
def make_field_exchange(A, Ms, dx = 0, dy = 0 , dz = 0, Nx = 1, Ny=1, Nz = 1):
    if A is None and Ms is None:
        return utils.make_noField(4, outNum=3)
    Bex = 2 * A / Ms
    if utils.is_constant(Bex):
        Bex_i = float(Bex[0])
        @njit(inline = "always")
        def field_exchange(lap_x, lap_y, lap_z,i):
            return Bex_i*lap_x,Bex_i*lap_y,Bex_i*lap_z
        return field_exchange


def make_field_exchange_sublattices(A, Ms, Nx, Ny, Nz, dx, dy, dz, reciever = "S", source = "L", bc = "Neumann"):
        two_over_dx2 = 2/dx**2
        two_over_dy2 = 2/dy**2
        two_over_dz2 = 2/dz**2
        two_over_Ms = 2/Ms
        
        iy = Nx
        iz = Nx * Ny

        if reciever == "S":
            x1 = 0
            y1 = 1
            z1 = 2
            m = 0
        else:
            x1 = 0+3
            y1 = 1+3
            z1 = 2+3
            m = 1

        if source == "S":
            x2 = 0
            y2 = 1
            z2 = 2
        else:
            x2 = 0+3
            y2 = 1+3
            z2 = 2+3

        if bc == "Neumann":
            @njit(inline="always")
            def idx(i, n):
                if i < 0:
                    return 0
                elif i >= n:
                    return n - 1
                return i

        elif bc == "Periodic":
            @njit(inline="always")
            def idx(i, n):
                if i < 0:
                    return n - 1
                elif i >= n:
                    return 0
                return i
        
        if bc == "pass":
            def update_field_exchange(J,Bex):
                pass

        @njit
        def update_field_exchange(J, B_ex):
            
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        

                        ip = idx(i+1,Nx)
                        im = idx(i-1,Nx)

                        jp = idx(j+1,Ny)
                        jm = idx(j-1,Ny)

                        kp = idx(k+1,Nz)
                        km = idx(k-1,Nz)

                        p = i + j*iy + k*iz

                        ppx = ip + j*iy + k*iz
                        pmx = im + j*iy + k*iz

                        ppy = i + jp*iy + k*iz
                        pmy = i + jm*iy + k*iz

                        ppz = i + j*iy + kp*iz
                        pmz = i + j*iy + km*iz


                        
                        A_c = A[p]
                        A_px = A[ppx]
                        A_mx = A[pmx]
                        A_py = A[ppy]
                        A_my = A[pmy]
                        A_pz = A[ppz]
                        A_mz = A[pmz]
                        pref = two_over_Ms[m,p] * A_c / mu0
                        #actual cell
                        Sx,Sy,Sz = J[x2,p],J[y2,p],J[z2,p]   
                        # X neighbours
                        Sx_px,Sy_px,Sz_px = J[x2,ppx],J[y2,ppx],J[z2,ppx]
                        Sx_mx,Sy_mx,Sz_mx = J[x2,pmx],J[y2,pmx],J[z2,pmx] 
                        # Y neighbours
                        Sx_py,Sy_py,Sz_py = J[x2,ppy],J[y2,ppy],J[z2,ppy]
                        Sx_my,Sy_my,Sz_my = J[x2,pmy],J[y2,pmy],J[z2,pmy]
                        # Z neighbours
                        Sx_pz,Sy_pz,Sz_pz = J[x2,ppz],J[y2,ppz],J[z2,ppz] 
                        Sx_mz,Sy_mz,Sz_mz = J[x2,pmz],J[y2,pmz],J[z2,pmz]
                        
                        B_ex[x1,p] += utils.exchange_stencil(pref, A_c, A_px, A_py, A_pz, A_mx, A_my, A_mz, Sx_px, Sx, Sx_py, Sx_pz, Sx_mx, Sx_my, Sx_mz, two_over_dx2, two_over_dy2, two_over_dz2)
                        B_ex[y1,p] += utils.exchange_stencil(pref, A_c, A_px, A_py, A_pz, A_mx, A_my, A_mz, Sy_px, Sy, Sy_py, Sy_pz, Sy_mx, Sy_my, Sy_mz, two_over_dx2, two_over_dy2, two_over_dz2)
                        B_ex[z1,p] += utils.exchange_stencil(pref, A_c, A_px, A_py, A_pz, A_mx, A_my, A_mz, Sz_px, Sz, Sz_py, Sz_pz, Sz_mx, Sz_my, Sz_mz, two_over_dx2, two_over_dy2, two_over_dz2)
        
        return update_field_exchange
    #else:
    #    #deprecated, wrong physical behaviour
    #    @njit(inline = "always")
    #    def field_exchange(lap_x, lap_y, lap_z,i):
    #        return Bex[i]*lap_x,Bex[i]*lap_y,Bex[i]*lap_z


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



