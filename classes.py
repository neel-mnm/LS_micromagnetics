from demag_kernel import *
import matplotlib.pyplot as plt
import utilities as mmutils


class MicromagneticMesh:
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dz = Lz / Nz

        self.N = Nx * Ny * Nz
        self.shape = (Nx, Ny, Nz)

        self.stride_x = 1
        self.stride_y = Nx
        self.stride_z = Nx*Ny

        self.dim = (Nx>1) + (Ny>1) + (Nz>1)

    
    def set_values_on_mesh(self, A, parameterName):
        Nx, Ny, Nz = self.shape
        N = self.N
        x = np.arange(Nx) * self.dx
        y = np.arange(Ny) * self.dy
        z = np.arange(Nz) * self.dz

        if np.isscalar(A):
            print(f"Set uniform {parameterName}")
            return np.full(N, A, dtype=np.float64)
        
        elif isinstance(A, np.ndarray) and A.shape == (N,):
            print(f"Set {parameterName} from flattened mesh")
            return A.astype(np.float64)

        elif isinstance(A, np.ndarray) and A.shape == (Nx, Ny, Nz):
            print(f"Set {parameterName} from grid")
            return A.ravel().astype(np.float64)
        
        elif callable(A):
            print(f"Set {parameterName} from function")
            values = A(x[:,None, None], y[None, :, None], z[None, None, :])
            try:
                values = np.broadcast_to(values, (Nx, Ny, Nz))
            except ValueError:
                raise ValueError("Function must be broadcastable to shape (Nx, Ny, Nz)")
            return values.ravel().astype(np.float64)
        
        else:
            raise TypeError(f"Invalid input type for {parameterName}")
    
    def _1d_laplacian_updater(self):
        """
        Factory function to obtain the laplacian in the specific 1d mesh considered. 
        Note that the numba-compiled output function has no "return" value as memory-wise it's faster to in-place change 
        """
        if self.Nx>1:
            dx = self.dx
        elif self.Ny>1:
            dx = self.dy
        elif self.Nz>1:
            dx = self.dz
        else:
            raise ValueError("Wrong dimensionality")
        one_over_dx2 = 1.0 / (dx*dx)
        two_over_dx2 = 2.0 / (dx*dx)
        N = self.N
        @njit(inline = "always")
        def update_laplacian(laplacian,J):
            for i in range(1,N-1):
                laplacian[0,i] = (J[0,i+1]-2.0*J[0,i]+J[0,i-1]) * one_over_dx2
                laplacian[1,i] = (J[1,i+1]-2.0*J[1,i]+J[1,i-1]) * one_over_dx2
                laplacian[2,i] = (J[2,i+1]-2.0*J[2,i]+J[2,i-1]) * one_over_dx2


            laplacian[0,0] =   (J[0,1]-J[0,0]) *     two_over_dx2
            laplacian[1,0] =   (J[1,1]-J[1,0]) *     two_over_dx2
            laplacian[2,0] =   (J[2,1]-J[2,0]) *     two_over_dx2
            laplacian[0,N-1] = (J[0,N-2]-J[0,N-1]) * two_over_dx2
            laplacian[1,N-1] = (J[1,N-2]-J[1,N-1]) * two_over_dx2
            laplacian[2,N-1] = (J[2,N-2]-J[2,N-1]) * two_over_dx2
        return update_laplacian
    
    def _2d_laplatian_updater(self):
        """
        Factory function to obtain the laplacian in the specific 2d mesh considered. 
        Note that the numba-compiled output function has no "return" value as memory-wise it's faster to in-place change 
        """

        if (self.Nx>1) and (self.Ny>1) and not (self.Nz>1):
            dx = self.dx
            dy = self.dy
            stride_x = self.stride_x
            stride_y = self.stride_y
            Nx = self.Nx
            Ny = self.Ny
             
        elif not (self.Nx>1) and (self.Ny>1) and (self.Nz>1):
            dx = self.dy
            dy = self.dz
            stride_x = self.stride_y
            stride_y = self.stride_z
            Nx = self.Ny
            Ny = self.Nz
            
        elif (self.Nx>1) and not (self.Ny>1) and (self.Nz>1):
            dx = self.dx
            dy = self.dz
            stride_x = self.stride_x
            stride_y = self.stride_z
            Nx = self.Nx
            Ny = self.Nz

        else:
            raise ValueError("Wrong dimensionality")
        

        one_over_dx2 = 1/(dx * dx)
        one_over_dy2 = 1/(dy * dy)


        @njit(inline = "always")
        def update_laplacian(laplacian,J):
            for y in range(1,Ny-1):
                for x in range(1, Nx-1):

                    i = x * stride_x + y*stride_y
                    # neighbors with Neumann BC
                    x_m = x-1 if x > 0 else 1
                    x_p = x+1 if x < Nx-1 else Nx-2
                    y_m = y-1 if y > 0 else 1
                    y_p = y+1 if y < Ny-1 else Ny-2
    
                    i_mx = x_m*stride_x + y*stride_y
                    i_px = x_p*stride_x + y*stride_y
                    i_my = x*stride_x + y_m*stride_y
                    i_py = x*stride_x + y_p*stride_y

                    laplacian[0,i] = (J[0, i_px]-2.0*J[0,i]+J[0,i_mx]) * one_over_dx2 + (J[0,i_py]-2.0*J[0,i]+J[0,i_my]) * one_over_dy2
                    laplacian[1,i] = (J[1, i_px]-2.0*J[1,i]+J[1,i_mx]) * one_over_dx2 + (J[1,i_py]-2.0*J[1,i]+J[1,i_my]) * one_over_dy2
                    laplacian[2,i] = (J[2, i_px]-2.0*J[2,i]+J[2,i_mx]) * one_over_dx2 + (J[2,i_py]-2.0*J[2,i]+J[2,i_my]) * one_over_dy2


        return update_laplacian
    
    def _3d_laplatian_updater(self):
        if not (self.Nx>1 and self.Ny>1 and self.Nz>1):
            raise ValueError("Wrong dimensionality")
        dx = self.dx
        dy = self.dy
        dz = self.dz
        stride_x = self.stride_x
        stride_y = self.stride_y
        stride_z = self.stride_z
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        one_over_dx2 = 1/(dx * dx)
        one_over_dy2 = 1/(dy * dy)
        one_over_dz2 = 1/(dz * dz)

        @njit(inline = "always")
        def update_laplacian(laplacian, J):
            for z in range(Nz):
                for y in range(Ny):
                    for x in range(Nx):
                        i = x*stride_x + y*stride_y + z*stride_z

                        # --- mirrored indices (Neumann BC) ---
                        x_m = x-1 if x > 0 else 1
                        x_p = x+1 if x < Nx-1 else Nx-2

                        y_m = y-1 if y > 0 else 1
                        y_p = y+1 if y < Ny-1 else Ny-2

                        z_m = z-1 if z > 0 else 1
                        z_p = z+1 if z < Nz-1 else Nz-2

                        i_mx = x_m*stride_x + y*stride_y + z*stride_z
                        i_px = x_p*stride_x + y*stride_y + z*stride_z

                        i_my = x*stride_x + y_m*stride_y + z*stride_z
                        i_py = x*stride_x + y_p*stride_y + z*stride_z

                        i_mz = x*stride_x + y*stride_y + z_m*stride_z
                        i_pz = x*stride_x + y*stride_y + z_p*stride_z
                        # --- unrolled for speed ---
                        laplacian[0, i] = (
                            (J[0, i_px] - 2.0*J[0, i] + J[0, i_mx]) * one_over_dx2 +
                            (J[0, i_py] - 2.0*J[0, i] + J[0, i_my]) * one_over_dy2 +
                            (J[0, i_pz] - 2.0*J[0, i] + J[0, i_mz]) * one_over_dz2
                        )

                        laplacian[1, i] = (
                            (J[1, i_px] - 2.0*J[1, i] + J[1, i_mx]) * one_over_dx2 +
                            (J[1, i_py] - 2.0*J[1, i] + J[1, i_my]) * one_over_dy2 +
                            (J[1, i_pz] - 2.0*J[1, i] + J[1, i_mz]) * one_over_dz2
                        )

                        laplacian[2, i] = (
                            (J[2, i_px] - 2.0*J[2, i] + J[2, i_mx]) * one_over_dx2 +
                            (J[2, i_py] - 2.0*J[2, i] + J[2, i_my]) * one_over_dy2 +
                            (J[2, i_pz] - 2.0*J[2, i] + J[2, i_mz]) * one_over_dz2
                        )
        return update_laplacian
    
    def _show_on_mesh(self, dataValues = None, z=0):
        """
        Shows the mesh and, if specified, the datavalues on it, plotting along the cross section at z=z
 
        """
        Nx, Ny, Nz = self.shape

        if z < 0 or z >= self.Lz:
            raise ValueError("z index out of bounds")
        nz = int(z/self.dz)
        
        if dataValues is None:
            print("Showing raw mesh")
            x = np.arange(Nx) * self.dx
            y = np.arange(Ny) * self.dy
            fig, ax = plt.subplots()
            for xi in x:
                ax.plot([xi]*Ny,y, color = "black")
            for yi in y:
                ax.plot(x, [yi]*Nx, color = "black")
           
            ax.set_title(f"Mesh stucture for z = {z}")
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_aspect("equal")

            plt.show()

        if dataValues.ndim ==1:
            field = dataValues.reshape((Nx,Ny,Nz))[:,:,nz]
            vmin = np.min(dataValues)
            vmax = np.max(dataValues)
            plt.imshow(field.T, origin="lower", vmin = vmin, vmax = vmax)
            plt.xlabel("y")
            plt.xlabel("x")
            plt.colorbar()
            plt.show()

    def get_laplacian_updater(self):
        if self.dim == 1:
            return self._1d_laplacian_updater()
        elif self.dim == 2:
            return self._2d_laplatian_updater()
        elif self.dim == 3:
            return self._3d_laplatian_updater()
        else:
            raise ValueError(f"Laplatian not defined for {self.dim}-dimensional systems")

    def _get_real_demag_kernel(self):

        Nx, Ny, Nz = self.shape
        dx, dy, dz = self.dx, self.dy, self.dz

        return get_real_kernel(Nx, Ny, Nz, dx, dy, dz)  

    def _get_reciprocal_demag_kernel(self):
        Nxx, Nyy, Nzz, Nxy, Nxz, Nyz  = self._get_real_demag_kernel()
        Kxx = numba_fft(numba_ifftshift(Nxx))*mu0
        Kyy = numba_fft(numba_ifftshift(Nyy))*mu0
        Kzz = numba_fft(numba_ifftshift(Nzz))*mu0
        Kxy = numba_fft(numba_ifftshift(Nxy))*mu0
        Kxz = numba_fft(numba_ifftshift(Nxz))*mu0
        Kyz = numba_fft(numba_ifftshift(Nyz))*mu0

        return Kxx, Kyy, Kzz, Kxy, Kxz, Kyz
           

        
class MicromagneticSystem:
    def __init__(self, mesh:MicromagneticMesh):
        self.mesh = mesh
        self.x = np.arange(self.mesh.Nx) * self.mesh.dx
        self.y = np.arange(self.mesh.Ny) * self.mesh.dy
        self.z = np.arange(self.mesh.Nz) * self.mesh.dz
        self.set_magnetocrystalline_on_spin(0)
        self.set_spin_g_factor(2)
        self.set_orbital_g_factor(1)
        self.set_demag_field(True)
        self.set_spin_damping(0)
        self.set_orbital_damping(0)
        self.Ms = np.zeros((2, self.mesh.N))
        self.set_bc("Neumann")

    def set_bc(self, bc):
        self.bc = bc
        print("Set boundary conditions to ", bc)

    def set_spin_magnetization(self, Ms):
        """
        Define spin saturation magnetization over the mesh.
        The only input Ms can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.Ms[0,:] = self.mesh.set_values_on_mesh(Ms, "Spin Magnetization")

    def set_orbital_magnetization(self, Ms):
        """
        Define orbital saturation magnetization over the mesh.
        The only input Ms can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.Ms[1,:] = self.mesh.set_values_on_mesh(Ms, "Orbital Magnetization")

    def set_external_field(self, Bext, ux, uy, uz):
        """
        Define the external field over the mesh.
        The inputs Bext, ux, uy, uz are:
           -the external field strength
           -the x, y, and z components of the unit vector
            
        All three can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """

        self.B0 = self.mesh.set_values_on_mesh(Bext, "external field amplitude")
        self.uBx = self.mesh.set_values_on_mesh(ux, "external field x-component")
        self.uBy = self.mesh.set_values_on_mesh(uy, "external field y-component")
        self.uBz = self.mesh.set_values_on_mesh(uz, "external field z-component")

    def set_magnetocrystalline_on_spin(self, onS):
        """
        Define whether the magnetocrystalline anisotropy terms  over the mesh.
        The only input onS can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.onS = self.mesh.set_values_on_mesh(onS, "magnetocrystalline anisotropy behaviour on S")
    
    def set_uniaxial_field(self, Bk, ux, uy, uz):
        """
        Define the uniaxial anisotropy field over the mesh.
        The inputs Bk, ux, uy, uz, and are:
           -the uniaxial anisotropy field strength
           -the x, y, and z components of the unit vector
                       
        All three can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """

        self.Bk = self.mesh.set_values_on_mesh(Bk, "uniaxial field amplitude")
        self.ukx = self.mesh.set_values_on_mesh(ux, "uniaxial anisotropy field x-component")
        self.uky = self.mesh.set_values_on_mesh(uy, "uniaxial anisotropy field y-component")
        self.ukz = self.mesh.set_values_on_mesh(uz, "uniaxial anisotropy field z-component")
        print("By default, magnetocrystalline anisotropy only acts on L. Remember to change it using set_magnetocrystalline_on_spin to change this")

    def set_oersted_field(self, BOe, ux=0, uy=1, uz=0):
        """
        Define the Oersted field over the mesh.
        The inputs BOe, ux, uy, uz are:
           -the Oersted field strength
           -the x, y, and z components of the unit vector
            
        All three can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """

        self.BOe = self.mesh.set_values_on_mesh(BOe, "Oersted field amplitude")
        self.jc_x = self.mesh.set_values_on_mesh(ux, "Oersted field x-component")
        self.jc_y = self.mesh.set_values_on_mesh(uy, "Oersted field y-component")
        self.jc_z = self.mesh.set_values_on_mesh(uz, "Oersted field z-component")

    def set_spin_accumulation(self, sx, sy, sz):
        """
        Define the spin accumulation direction over the mesh.
        The inputs ux, uy, uz are:
           -the x, y, and z components of the unit vector
            
        All three can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.s_x = self.mesh.set_values_on_mesh(sx, "Spin accumulation x-component")
        self.s_y = self.mesh.set_values_on_mesh(sy, "Spin accumulation y-component")
        self.s_z = self.mesh.set_values_on_mesh(sz, "Spin accumulation z-component")
    
    def set_orbital_accumulation(self, lx, ly, lz):
        """
        Define the orbital accumulation direction over the mesh.
        The inputs ux, uy, uz are:
           -the x, y, and z components of the unit vector
            
        All three can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.l_x = self.mesh.set_values_on_mesh(lx, "Orbital accumulation x-component")
        self.l_y = self.mesh.set_values_on_mesh(ly, "Orbital accumulation y-component")
        self.l_z = self.mesh.set_values_on_mesh(lz, "Orbital accumulation z-component")

    def set_fieldlike_spin(self, Bfl):
        """
        Define spin fieldlike torque over the mesh.
        The only input Bfl can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "s_x"):
            return ValueError("Missing spin accumulation. Use 'set_spin_accumulation' first")
        self.Bfl_s = self.mesh.set_values_on_mesh(Bfl, "Spin fieldLike torque")
    
    def set_fieldlike_orbital(self, Bfl):
        """
        Define orbital fieldlike torque over the mesh.
        The only input Bfl can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "l_x"):
            return ValueError("Missing orbital accumulation. Use 'set_orbital_accumulation' first")
        self.Bfl_l = self.mesh.set_values_on_mesh(Bfl, "Orbital fieldLike torque")

    def set_dampinglike_spin(self, Bdl):
        """
        Define spin dampinglike torque over the mesh.
        The only input Bfl can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "s_x"):
            return ValueError("Missing spin accumulation. Use 'set_spin_accumulation' first")
        self.Bdl_s = self.mesh.set_values_on_mesh(Bdl, "Spin fieldLike torque")
    
    def set_dampinglike_orbital(self, Bdl):
        """
        Define orbital dampinglike torque over the mesh.
        The only input Bfl can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "l_x"):
            return ValueError("Missing orbital accumulation. Use 'set_orbital_accumulation' first")
        self.Bdl_l = self.mesh.set_values_on_mesh(Bdl, "Orbital fieldLike torque")

    def set_spin_g_factor(self, gS):
        """
        Define the spin g-factor over the mesh.
        The only input Ms can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.gS = self.mesh.set_values_on_mesh(gS, "Spin g-factor")

    def set_orbital_g_factor(self, gL):
        """
        Define the orbital g-factor over the mesh.
        The only input can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        self.gL = self.mesh.set_values_on_mesh(gL, "Orbital g-factor")
    
    def set_spin_damping(self, alpha_S):
        self.alphaS = self.mesh.set_values_on_mesh(alpha_S, "spin Gilbert damping")
    
    def set_orbital_damping(self, alpha_L):
        self.alphaL = self.mesh.set_values_on_mesh(alpha_L, "orbital Gilbert damping")

    def set_spinorbit_field(self,Bso):
        """
        Define the local spin-orbit interaction over the mesh. This means the interaction between L and S within the same node.
        Calling this function requires to have already defined both an orbital magnetization, a spin magnetization, and the g-factors, though the g-factors default to 2 and 1 for sspin andorbital respectively
        The only input can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "Ms") or not hasattr(self, "Ms") or not hasattr(self, "gS") or not hasattr(self, "gL"):
            return ValueError("Magnetizations or g-factors not defined")
        self.Bso = self.mesh.set_values_on_mesh(Bso, "spin-orbit field")

    def set_SS_exchange(self, A):
        """
        Define the nearest neighbour exchange interaction over the mesh.
        This term only acts on the spin degrees of freedom and calling this function requires having defined a spin magnetization. 
        The only input can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "Ms"):
            return ValueError("Spin magnetization not defined")
        self.A_SS = self.mesh.set_values_on_mesh(A, "first neighbor SS exchange")

    def set_LS_exchange(self, A):
        """
        Define the nearest neighbour exchange interaction over the mesh.
        This term only acts on the spin degrees of freedom and calling this function requires having defined a spin magnetization. 
        The only input can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "Ms"):
            return ValueError("Spin magnetization not defined")
        self.A_LS = self.mesh.set_values_on_mesh(A, "first neighbor LS exchange")

    def set_LL_exchange(self, A):
        """
        Define the nearest neighbour exchange interaction over the mesh.
        This term only acts on the spin degrees of freedom and calling this function requires having defined a spin magnetization. 
        The only input can either be:
          - a scalar
          - a vector of the same shape as the mesh
          - a flattened vector the same length as the flattened mesh 
          - a function of exactly 3 variables (x,y,z). Note that this needs to be mesh compatible  
        """
        if not hasattr(self, "Ms"):
            return ValueError("Spin magnetization not defined")
        self.A_LL = self.mesh.set_values_on_mesh(A, "first neighbor LL exchange")

    def set_demag_field(self, on: bool = True):
        self.usedemag = on
        if on:
            print("Turned on convolution-based demagnetizing field, call set_demag_field(False) to change this")
        else:
            print("Turned off convolution-based demagnetizing field, call set_demag_field(True) to change this")

    def set_demag_tensor(self, Nxx, Nyy, Nzz):
        self.set_demag_field(False)
        self.N_xx = self.mesh.set_values_on_mesh(Nxx, "demag tensor x-component")
        self.N_yy = self.mesh.set_values_on_mesh(Nyy, "demag tensor y-component")
        self.N_zz = self.mesh.set_values_on_mesh(Nzz, "demag tensor z-component")
        print("Demag tensor now defined locally, mesh point by mesh point. Careful")


    def set_initial_state(self, Sx, Sy, Sz, Lx, Ly, Lz):
        Sx = self.mesh.set_values_on_mesh(Sx, "Initial spin x")
        Sy = self.mesh.set_values_on_mesh(Sy, "Initial spin y")
        Sz = self.mesh.set_values_on_mesh(Sz, "Initial spin z")
        Lx = self.mesh.set_values_on_mesh(Lx, "Initial orbital x")
        Ly = self.mesh.set_values_on_mesh(Ly, "Initial orbital y")
        Lz = self.mesh.set_values_on_mesh(Lz, "Initial orbital z")
        self.J = np.array([Sx, Sy, Sz, Lx, Ly, Lz])

    def jitter_initial_state(self,eps=1e-3):
        self.J = mmutils.jitter(self.J, eps)


    def time_evolution(self, tf, dt, t0 = 0, dynamics = "full", save_every = 100, stream = True, normalize_every = 10, fmrFieldFunction = None):
        return mmutils.timeEvol(self.J,self,fmrFieldFunction=fmrFieldFunction, tf = tf, dt = dt, t0=t0, dynamics=dynamics, save_every=save_every, stream = stream, normalize_every=normalize_every)
        