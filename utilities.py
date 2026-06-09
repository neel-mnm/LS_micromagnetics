import numpy as np
from numba import njit
import classes as mmcls
import factories as mmfct



def is_constant(arr):
    return np.allclose(arr, arr.flat[0])

def harmonic_mean(A,B):
    return 2*A*B/(A+B)

@njit(inline = "always")
def exchange_stencil(pref, A_c, A_px, A_py, A_pz, A_mx, A_my, A_mz, Sx_px, Sx, Sx_py, Sx_pz, Sx_mx, Sx_my, Sx_mz, two_over_dx2, two_over_dy2, two_over_dz2):
    return pref * (
                                (A_px*(Sx_px-Sx)/(A_px+A_c) + A_mx*(Sx_mx-Sx)/(A_mx+A_c))*two_over_dx2 +
                                (A_py*(Sx_py-Sx)/(A_py+A_c) + A_my*(Sx_my-Sx)/(A_my+A_c))*two_over_dy2 +
                                (A_pz*(Sx_pz-Sx)/(A_pz+A_c) + A_mz*(Sx_mz-Sx)/(A_mz+A_c))*two_over_dz2 
                            )


def make_noField(varNum, outNum = 6):
    match varNum: 
        case 4:
            out4 = np.zeros(outNum)
            @njit(inline = "always")
            def no_field(lap_x, lap_y, lap_z,i):
                return out4
            
        case 2:
            out2 = np.zeros(outNum)
            @njit(inline = "always")
            def no_field(I, i):
                return out2
            
        case 1:
            out1 = np.zeros(outNum)
            @njit(inline = "always")
            def no_field(i):
                return out1
        
        case 5:
            out5 = np.zeros(outNum)
            @njit(inline = "always")
            def no_field(Sx, Sy, Sz, I, i):
                return out5
        
        case 7:
            out7 = np.zeros(outNum)
            @njit(inline = "always")
            def no_field(Sx, Sy, Sz, Lx, Ly, Lz, i):
                return out7
    return no_field

def build_no_effect(shape):
    out = np.zeros(shape)
    @njit(inline = "always")
    def noEffect(t):
        """
        Default function for the current (i.e. no current on any node, which are stored on an array of shape=shape)
        """
        return out
    return noEffect

def jitter(J, eps=1e-5):
    """
    Function to randomly push out of the initial state the mangnetization
    """
    Jj = J.copy()
    N = J.shape[1]

    for i in range(N):
        # small random perturbation
        dx = eps * (np.random.rand() )
        dy = eps * (np.random.rand() )
        dz = eps * (np.random.rand() )

        Jj[0,i] += dx
        Jj[1,i] += dy
        Jj[2,i] += dz
        Jj[3,i] += dx
        Jj[4,i] += dy
        Jj[5,i] += dz

        # renormalize S
        s_norm = np.sqrt(Jj[0,i]**2 + Jj[1,i]**2 + Jj[2,i]**2)
        Jj[0,i] /= s_norm
        Jj[1,i] /= s_norm
        Jj[2,i] /= s_norm

        # renormalize L
        l_norm = np.sqrt(Jj[3,i]**2 + Jj[4,i]**2 + Jj[5,i]**2)
        Jj[3,i] /= l_norm
        Jj[4,i] /= l_norm
        Jj[5,i] /= l_norm

    return Jj

@njit
def RK4_stream_integrator(J0, t0, tf, dt, func, save_every=100, normalize_every = -1):

    n_step = int((tf - t0) / dt)
    N = J0.shape[1]

    k1 = np.zeros_like(J0)
    k2 = np.zeros_like(J0)
    k3 = np.zeros_like(J0)
    k4 = np.zeros_like(J0)

    laplacian = np.zeros((6, N))
    B = np.zeros_like(J0)

    y = J0.copy()
    t = t0

    # ALWAYS allocate SAME TYPES
    if save_every != -1:
        n_save = n_step // save_every + 1
        J_save = np.empty((6, N, n_save), dtype=np.float64)
        t_save = np.empty(n_save, dtype=np.float64)

        save_idx = 0
        J_save[:, :, save_idx] = y
        t_save[save_idx] = t
        save_idx += 1
    else:
        # still allocate arrays, but fixed shape
        J_save = np.empty((6, N, 2), dtype=np.float64)
        t_save = np.empty(2, dtype=np.float64)




    # MAIN LOOP
    for i in range(n_step):

        func(t, y, B, k1, laplacian)
        func(t + 0.5 * dt, y + 0.5 * dt * k1, B, k2, laplacian)
        func(t + 0.5 * dt, y + 0.5 * dt * k2, B, k3, laplacian)
        func(t + dt, y + dt * k3, B, k4, laplacian)


        y += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        #print(t)

        if normalize_every != -1 and i % normalize_every == 0:
        
            # S normalization
            for j in range(y.shape[1]):
                sx = y[0, j]
                sy = y[1, j]
                sz = y[2, j]
        
                sn = (sx*sx + sy*sy + sz*sz) ** 0.5 + 1e-12
        
                y[0, j] = sx / sn
                y[1, j] = sy / sn
                y[2, j] = sz / sn
        
            # L normalization
            for j in range(y.shape[1]):
                lx = y[3, j]
                ly = y[4, j]
                lz = y[5, j]
        
                ln = (lx*lx + ly*ly + lz*lz) ** 0.5 + 1e-12
        
                y[3, j] = lx / ln
                y[4, j] = ly / ln
                y[5, j] = lz / ln      

        if save_every != -1:
            if i % save_every == 0:
                J_save[:, :, save_idx] = y
                t_save[save_idx] = t
                save_idx += 1
    
    if save_every == -1:
        J_save[:,:,0] = J0
        J_save[:,:,1] = y
        t_save[0] = 0

    print(t_save, J_save)

    return t_save, J_save

@njit
def RK4_integrator(J0, t0, tf, dt, func, normalize_every = -1):

    n_step = int((tf - t0) / dt)
    N = J0.shape[1]

    k1 = np.empty_like(J0)
    k2 = np.empty_like(J0)
    k3 = np.empty_like(J0)
    k4 = np.empty_like(J0)

    laplacian = np.empty((3, N))
    B = np.empty_like(J0)

    y = J0.copy()
    t = t0

    # ALWAYS allocate SAME TYPES
    n_save = n_step 
    J_save = np.empty((6, N, n_save), dtype=np.float64)
    t_save = np.empty(n_save, dtype=np.float64)

    save_idx = 0
    J_save[:, :, save_idx] = y
    t_save[save_idx] = t
    save_idx += 1

    J_save = np.empty((6, N, 2), dtype=np.float64)
    t_save = np.empty(2, dtype=np.float64)




    # MAIN LOOP
    for i in range(n_step):

        func(t, y, B, k1, laplacian)
        func(t + 0.5 * dt, y + 0.5 * dt * k1, B, k2, laplacian)
        func(t + 0.5 * dt, y + 0.5 * dt * k2, B, k3, laplacian)
        func(t + dt, y + dt * k3, B, k4, laplacian)

        y += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt

        if normalize_every != -1 and i % normalize_every == 0:
        
            # S normalization
            for j in range(y.shape[1]):
                sx = y[0, j]
                sy = y[1, j]
                sz = y[2, j]
        
                sn = (sx*sx + sy*sy + sz*sz) ** 0.5 + 1e-12
        
                y[0, j] = sx / sn
                y[1, j] = sy / sn
                y[2, j] = sz / sn
        
            # L normalization
            for j in range(y.shape[1]):
                lx = y[3, j]
                ly = y[4, j]
                lz = y[5, j]
        
                ln = (lx*lx + ly*ly + lz*lz) ** 0.5 + 1e-12
        
                y[3, j] = lx / ln
                y[4, j] = ly / ln
                y[5, j] = lz / ln      

        
        J_save[:, :, save_idx] = y
        t_save[save_idx] = t
        save_idx += 1
    
    return t_save, J_save


def timeEvol(J0, system: mmcls.MicromagneticSystem, fmrFieldFunction, tf, dt = 1e-10, t0 = 0, dynamics = "full", save_every = 100, stream = False, normalize_every = -1):
    LLGs = mmfct.get_LLGs(system, dynamics, fmrFieldFunction)
    n_step = int((tf-t0)/dt)
    N = J0.shape[1]

    estimated_bytes = 6 * N * (n_step+1) * 8

    if estimated_bytes > 1e9 or stream:  # ~1 GB threshold
        print("Using streaming RK4")
        return RK4_stream_integrator(J0, t0, tf, dt, LLGs, save_every=save_every, normalize_every=normalize_every)
    else:
        print("Using full RK4")
        return RK4_integrator(J0, t0, tf, dt, LLGs)