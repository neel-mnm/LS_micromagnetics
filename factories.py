
from fields import *
import utilities as mmutils
from classes import *
from numba.core.registry import CPUDispatcher

def make_effective_field(system : MicromagneticSystem):

    print("Building effective field")

    if system.mesh.dim != 0:
        update_laplacian = system.mesh.get_laplacian_updater()


        if hasattr(system, "A") and hasattr(system, "Ms_S"):
            Bex = 2*system.A/(system.Ms_S)
            update_field_exchange = make_field_exchange(Bex)
        else:
            if not hasattr(system, "A"):
                print("System has undefined exchange stiffness. Setting exchange to zero")
            if not hasattr(system, "Ms_S"):
                print("System has undefined magnetization, can't define exchange field. Defaulting to zero")
            update_field_exchange = make_field_exchange(None)
        
    if hasattr(system, "B0"):
        print("Building external field")
        uBx, uBy, uBz = system.uBx, system. uBy, system.uBz
        B0 = system.B0
        update_field_external = make_field_external(B0,uBx,uBy,uBz)
        print("External field built")
    else:
        update_field_external = make_field_external(None, 0, 0, 0)


    if hasattr(system, "Bk"):
        kx0, ky0, kz0 = system.ukx,system.uky,system.ukz
        Bk = system.Bk
        aniS = system.onS
        update_field_uniaxial = make_field_uniaxial(Bk, kx0, ky0, kz0, aniS)
    else:
        update_field_uniaxial = make_field_uniaxial(None, 0,0,0,0)


    if hasattr(system, "BOe"):
        uBOx, uBOy, uBOz = system.jc_x,system.jc_y,system.jc_z
        BOe = system.BOe
        update_field_oersted = make_field_oersted(BOe,uBOx,uBOy,uBOz)
    else:
        update_field_oersted = make_field_oersted(None,0,0,0)
    
    if hasattr(system, "Bso"):
        Bso = system.Bso
        Ms_S = system.Ms_S
        Ms_L = system.Ms_L
        gL = system.gL
        gS = system.gS
        update_field_spinorbit = make_field_spinorbit(Bso, Ms_S, Ms_L, gL, gS)
    else:
        update_field_spinorbit = utils.make_noField(7)

    if hasattr(system, "Bfl_s"):
        sx, sy, sz = system.s_x,system.s_y,system.s_z
        Bfl_s = system.Bfl_s
        update_fieldlike_spin = make_field_fieldlike(Bfl_s, sx, sy, sz)
    else:
        update_fieldlike_spin = utils.make_noField(2, outNum=3)

    if hasattr(system, "Bfl_l"):
        lx, ly, lz = system.l_x,system.l_y,system.l_z
        Bfl_l = system.Bfl_l
        update_fieldlike_orbital = make_field_fieldlike(Bfl_l, lx, ly, lz)
    else:
        update_fieldlike_orbital = utils.make_noField(2, outNum=3)

    if hasattr(system, "Bdl_s"):
        sx, sy, sz = system.s_x,system.s_y,system.s_z
        Bdl_s = system.Bdl_s
        update_dampinglike_spin = make_field_dampinglike(Bdl_s, sx, sy, sz)
    else:
        update_dampinglike_spin = utils.make_noField(5, outNum=3)

    if hasattr(system, "Bdl_l"):
        lx, ly, lz = system.l_x,system.l_y,system.l_z
        Bdl_l = system.Bdl_l
        update_dampinglike_orbital = make_field_dampinglike(Bdl_l, lx, ly, lz)
    else:
        update_dampinglike_orbital = utils.make_noField(5, outNum=3)

    if hasattr(system, "Nxx"):
        if hasattr(system, "Ms_S"):
            MsS = system.Ms_S 
        else:
            MsS = np.zeros_like(system.Nxx)
            
        if hasattr(system, "Ms_L"):
            MsL = system.Ms_L 
        else:
            MsL = np.zeros_like(system.Nxx)
        Nxx = system.N_xx
        Nyy = system.N_yy
        Nzz = system.N_zz
        update_field_demag = make_field_demag(None, None, None, None, None, None, MsS,MsL,Nxx,Nyy,Nzz, convolution=False)
    elif system.usedemag:
        print("Building demag field")
        if hasattr(system, "Ms_S"):
            MsS = system.Ms_S 
        else:
            MsS = np.zeros(system.mesh.N)
            
        if hasattr(system, "Ms_L"):
            MsL = system.Ms_L 
        else:
            MsL = np.zeros(system.mesh.N)
        Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = system.mesh._get_reciprocal_demag_kernel()
        Nx, Ny, Nz = system.mesh.shape
        update_field_demag = make_field_demag(Kxx, Kyy, Kzz, Kxy, Kxz, Kyz, MsS, MsL,Nx, Ny, Nz)
        print("Demag field built")
    N = system.mesh.N



    if system.mesh.dim == 0:
        @njit(inline = "always")
        def field_effective(B, I, J, laplacian):
            hx=hy=hz=0.0
            kx=ky=kz=0.0
            Sx,Sy,Sz = J[0],J[1],J[2]
            Lx,Ly,Lz = J[3],J[4],J[5]   

            bx, by, bz  = update_field_demag(Sx, Sy, Sz, Lx, Ly, Lz) 
            hx+=bx.item(); hy+=by.item(); hz+=bz.item();kx+=bx.item(); ky+=by.item(); kz+=bz.item()
            
            bx,by,bz,cx,cy,cz = update_field_external(0)
            hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz 
            
            bx,by,bz,cx,cy,cz=update_field_uniaxial(Sx,Sy,Sz,Lx,Ly,Lz, 0)
            hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz 
            
            bx,by,bz,cx,cy,cz = update_field_oersted(I,0)
            hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz
            
            bx,by,bz,cx,cy,cz=update_field_spinorbit(Sx,Sy,Sz,Lx,Ly,Lz, 0)
            hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz

            hx+=bx; hy+=by; hz+=bz
            bx,by,bz = update_fieldlike_spin(I,0)
            hx+=bx; hy+=by; hz+=bz
            
            cx, cy, cz = update_fieldlike_orbital(I,0)
            kx+=cx; ky+=cy; kz+=cz
            
            bx,by,bz = update_dampinglike_spin(I,Sx,Sy,Sz,0)
            hx+=bx; hy+=by; hz+=bz
            
            cx, cy, cz = update_dampinglike_orbital(I,Lx,Ly,Lz,0)
            kx+=cx; ky+=cy; kz+=cz
            
            B[0]=hx
            B[1]=hy
            B[2]=hz
            B[3]=kx
            B[4]=ky
            B[5]=kz
        return field_effective
    else:
        @njit(inline = "always")
        def field_effective(B, I, J, laplacian):
            update_laplacian(laplacian,J)
            demX, demY, demZ = update_field_demag(J[0,:],J[1,:],J[2,:],J[3,:],J[4,:],J[5,:])

            for i in range(N):
            
                Ii = I[i]

                hx=hy=hz=0.0
                kx=ky=kz=0.0

                hx +=demX[i]
                hy +=demY[i]
                hz +=demZ[i]
                
                kx +=demX[i]
                ky +=demY[i]
                kz +=demZ[i]

                Sx,Sy,Sz = J[0,i],J[1,i],J[2,i]
                Lx,Ly,Lz = J[3,i],J[4,i],J[5,i]            



                bx,by,bz,cx,cy,cz = update_field_external(i)
                hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz 


                bx,by,bz,cx,cy,cz=update_field_uniaxial(Sx,Sy,Sz,Lx,Ly,Lz, i)
                hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz 


                bx,by,bz,cx,cy,cz = update_field_oersted(Ii,i)
                hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz


                bx,by,bz,cx,cy,cz=update_field_spinorbit(Sx,Sy,Sz,Lx,Ly,Lz, i)
                hx+=bx; hy+=by; hz+=bz;kx+=cx; ky+=cy; kz+=cz


                lapX, lapY, lapZ = laplacian[0,i],laplacian[1,i],laplacian[2,i]
                bx,by,bz=update_field_exchange(lapX, lapY, lapZ,i)
                hx+=bx; hy+=by; hz+=bz


                bx,by,bz = update_fieldlike_spin(Ii,i)
                hx+=bx; hy+=by; hz+=bz


                cx, cy, cz = update_fieldlike_orbital(Ii,i)
                kx+=cx; ky+=cy; kz+=cz

                bx,by,bz = update_dampinglike_spin(Ii,Sx,Sy,Sz,i)
                hx+=bx; hy+=by; hz+=bz


                cx, cy, cz = update_dampinglike_orbital(Ii,Lx,Ly,Lz,i)
                kx+=cx; ky+=cy; kz+=cz


                B[0,i]=hx
                B[1,i]=hy
                B[2,i]=hz
                B[3,i]=kx
                B[4,i]=ky
                B[5,i]=kz
        return field_effective
    

def get_LLGs(system:MicromagneticSystem, dynamics = "full", currentFunc = None):

    alphaS = system.alphaS
    alphaL = system.alphaL


    gS = system.gS
    gL = system.gL
    N = system.mesh.N

    gamma_base = -muB / hbar
    prefactors_S = gS * gamma_base / (1 + alphaS**2)
    prefactors_L = gL * gamma_base / (1 + alphaL**2)

    update_field_func = make_effective_field(system)

    fullMode = dynamics=="full"
    dampingMode = dynamics=="damping"
    precessionMode = dynamics=="precession"

    if currentFunc is None:
        currentFunc = mmutils.build_no_effect(N)
        
    if not isinstance(currentFunc, CPUDispatcher):
        raise TypeError(f"{currentFunc.__name__} must be a numba compiled function")

    print(f"Building LLGs with {dynamics} dynamics")

    if fullMode:
        @njit
        def LLG(t,J, B, dJdt, laplatian):
            I = currentFunc(t)
            update_field_func(B,I,J, laplatian)
            for i in range(N):




                Sx = J[0,i]
                Sy = J[1,i]
                Sz = J[2,i]
                Lx = J[3,i]
                Ly = J[4,i]
                Lz = J[5,i]

                BSx = B[0,i]
                BSy = B[1,i]
                BSz = B[2,i]
                BLx = B[3,i]
                BLy = B[4,i]
                BLz = B[5,i]

                pref_S = prefactors_S[i]
                pref_L = prefactors_L[i]
                alpha_S = alphaS[i]
                alpha_L = alphaL[i]



                px = Sy*BSz -  BSy*Sz
                py = -Sx*BSz +  BSx*Sz
                pz = Sx*BSy -  BSx*Sy


                dx = Sy*pz -  py*Sz
                dy = -Sx*pz +  px*Sz
                dz = Sx*py -  px*Sy

                sx = px + alpha_S * dx
                sy = py + alpha_S * dy
                sz = pz + alpha_S * dz

                dJdt[0,i]=pref_S * sx
                dJdt[1,i]=pref_S * sy
                dJdt[2,i]=pref_S * sz



                px = Ly*BLz -  BLy*Lz
                py = -Lx*BLz +  BLx*Lz
                pz = Lx*BLy -  BLx*Ly


                dx = Ly*pz -  py*Lz
                dy = -Lx*pz +  px*Lz
                dz = Lx*py -  px*Ly

                lx = px + alpha_L * dx
                ly = py + alpha_L * dy
                lz = pz + alpha_L * dz



                dJdt[3,i]=pref_L * lx
                dJdt[4,i]=pref_L * ly
                dJdt[5,i]=pref_L * lz


            #return dJdt


    elif dampingMode:
        @njit
        def LLG(t,J, B, dJdt, laplatian):
            I = currentFunc(t)
            update_field_func(B,I,J, laplatian)
            for i in range(N):




                Sx = J[0,i]
                Sy = J[1,i]
                Sz = J[2,i]
                Lx = J[3,i]
                Ly = J[4,i]
                Lz = J[5,i]

                BSx = B[0,i]
                BSy = B[1,i]
                BSz = B[2,i]
                BLx = B[3,i]
                BLy = B[4,i]
                BLz = B[5,i]

                pref_S = prefactors_S[i]
                pref_L = prefactors_L[i]
                alpha_S = alphaS[i]
                alpha_L = alphaL[i]



                px = Sy*BSz -  BSy*Sz
                py = -Sx*BSz +  BSx*Sz
                pz = Sx*BSy -  BSx*Sy




                dx = Sy*pz -  py*Sz
                dy = -Sx*pz +  px*Sz
                dz = Sx*py -  px*Sy

                sx = alpha_S * dx
                sy = alpha_S * dy
                sz = alpha_S * dz



                dJdt[0,i]=pref_S * sx
                dJdt[1,i]=pref_S * sy
                dJdt[2,i]=pref_S * sz



                px = Ly*BLz -  BLy*Lz
                py = -Lx*BLz +  BLx*Lz
                pz = Lx*BLy -  BLx*Ly




                dx = Ly*pz -  py*Lz
                dy = -Lx*pz +  px*Lz
                dz = Lx*py -  px*Ly

                lx = alpha_L * dx
                ly = alpha_L * dy
                lz = alpha_L * dz


                dJdt[3,i]=pref_L * lx
                dJdt[4,i]=pref_L * ly
                dJdt[5,i]=pref_L * lz


    else:
        @njit
        def LLG(t,J, B, dJdt, laplatian):
            I = currentFunc(t)
            update_field_func(B,I,J, laplatian)
            for i in range(N):




                Sx = J[0,i]
                Sy = J[1,i]
                Sz = J[2,i]
                Lx = J[3,i]
                Ly = J[4,i]
                Lz = J[5,i]

                BSx = B[0,i]
                BSy = B[1,i]
                BSz = B[2,i]
                BLx = B[3,i]
                BLy = B[4,i]
                BLz = B[5,i]

                pref_S = prefactors_S[i]
                pref_L = prefactors_L[i]




                px = Sy*BSz -  BSy*Sz
                py = -Sx*BSz +  BSx*Sz
                pz = Sx*BSy -  BSx*Sy

                sx = px
                sy = py
                sz = pz

                dJdt[0,i]=pref_S * sx
                dJdt[1,i]=pref_S * sy
                dJdt[2,i]=pref_S * sz



                px = Ly*BLz -  BLy*Lz
                py = -Lx*BLz +  BLx*Lz
                pz = Lx*BLy -  BLx*Ly


                lx = px
                ly = py
                lz = pz



                dJdt[3,i]=pref_L * lx
                dJdt[4,i]=pref_L * ly
                dJdt[5,i]=pref_L * lz


            #return dJdt
    print(f"LLG equations built")
    return LLG


