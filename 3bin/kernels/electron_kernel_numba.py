import math
import cupy as cp
import numba
from numba import cuda

# Kernel for electron movement and collisions
@cuda.jit
def electron_kernel_1D(Nc, d_data_ar, d_tosum_ar, d_collct_ar, 
                       d_E_ar, d_R_vec, d_currxbins, d_forE_xbins, 
                       nn, dt_ratio, dt, d_curr_count,
                       d_g0_ar, d_g2_ar, d_gas_v_ar,):

    L = 2.54
    q_e = 1.602*10**(-19) # charge of electron (C)
    m_e = 9.10938*10**(-31) # mass of electron (kg)
    m_n = 6.6*10**(-26) # mass of neutral particles (argon)
    j_ev = 6.241509*(10**18) # joules to eV conversion factor
   
    # Collision energy losses (eV) 
    e_ion_full = 15.76

    num_steps = dt_ratio

    epsilon = 0.01

    # Setting up threads for particles 
    tidx = cuda.threadIdx.x
    bidx = cuda.blockIdx.x

    block_dimx = cuda.blockDim.x
    grid_dimx = cuda.gridDim.x

    start = tidx + bidx * block_dimx
    stride = block_dimx * grid_dimx
    for mystep in range(num_steps):
    
        # Looping and doing all the particles 
        for i in range(start, Nc, stride):
            d_w = d_tosum_ar[i,0]
            nz_ind = d_w > 0
            
            d_xx = d_data_ar[i,0] 
            d_xy = d_data_ar[i,1] 
            d_xz = d_data_ar[i,2] 
            
            d_vx = d_data_ar[i,3] * nz_ind + epsilon * (1 - nz_ind) 
            d_vy = d_data_ar[i,4] * nz_ind + epsilon * (1 - nz_ind)
            d_vz = d_data_ar[i,5] * nz_ind + epsilon * (1 - nz_ind)

            ## Electron velocity / energy (for non-zero gas temp.)
            adj_vx = d_vx - d_gas_v_ar[i,0]
            adj_vy = d_vy - d_gas_v_ar[i,1]
            adj_vz = d_vz - d_gas_v_ar[i,2]

            e_el = adj_vx**2 + adj_vy**2 + adj_vz**2
            v_mag = math.sqrt(e_el)

            v_inc_x = adj_vx / v_mag
            v_inc_y = adj_vy / v_mag
            v_inc_z = adj_vz / v_mag

            # conver to eV
            e_el = 0.5*m_e*j_ev*e_el
            log_e_el = math.log10(e_el)

            ## G0 - ELASTIC - REAL DATA
            a1 = -0.02704763
            b1 = -0.23720051
            c1 = -19.67900951
            a2 = -0.08847237
            b2 = -0.6084786
            c2 = -20.24111992
            a3 = -0.37608274
            b3 = -1.8167778
            c3 = -21.51414308
            a4 = -1.4874467
            b4 = -4.82420619
            c4 = -23.55745478
            a5 = -0.9870356
            b5 = -4.2206026
            c5 = -23.44715988
            a6 = 14.28063581
            b6 = 18.92275458
            c6 = -14.70556113
            a7 = -2.12069169
            b7 = 0.7555105
            c7 = -19.71943671
            a8 = -0.3585636
            b8 = 0.79246666
            c8 = -19.71260274
            a9 = 1.25262128
            b9 = -0.40002029
            c9 = -19.48909003
            a10 = -2.28332905
            b10 = 4.89566076
            c10 = -21.46789173
            a11 = -1.47508661
            b11 = 2.82263476
            c11 = -20.1928456
            a12 = -0.11090525
            b12 = -0.77506081
            c12 = -17.81752307
            
            i1 = 0.000780396
            i2 = 0.006098667
            i3 = 0.0476269
            i4 = 0.1015574
            i5 = 0.1943527
            i6 = 0.2995781
            i7 = 1.097156
            i8 = 2.339524
            i9 = 4.988691
            i10 = 10.63765
            i11 = 22.68322
            log_sig_g0 = (e_el < i1) * (((a1*log_e_el + b1)* log_e_el + c1)) + \
                         (i1 <= e_el < i2) * (((a2*log_e_el + b2)* log_e_el + c2)) + \
                         (i2 <= e_el < i3) * (((a3*log_e_el + b3)* log_e_el + c3)) + \
                         (i3 <= e_el < i4) * (((a4*log_e_el + b4)* log_e_el + c4)) + \
                         (i4 <= e_el < i5) * (((a5*log_e_el + b5)* log_e_el + c5)) + \
                         (i5 <= e_el < i6) * (((a6*log_e_el + b6)* log_e_el + c6)) + \
                         (i6 <= e_el < i7) * (((a7*log_e_el + b7)* log_e_el + c7)) + \
                         (i7 <= e_el < i8) * (((a8*log_e_el + b8)* log_e_el + c8)) + \
                         (i8 <= e_el < i9) * (((a9*log_e_el + b9)* log_e_el + c9)) + \
                         (i9 <= e_el < i10) * (((a10*log_e_el + b10)* log_e_el + c10)) + \
                         (i10 <= e_el < i11) * (((a11*log_e_el + b11)* log_e_el + c11)) + \
                         (i11 <= e_el) * (((a12*log_e_el + b12)* log_e_el + c12))
            sig_g0 = 10**log_sig_g0

            ## G2 - IONIZATION - REAL DATA
            a1 = -2149239.99337822
            b1 = 5150451.29014456
            c1 = -3085665.64874862
            a2 = -227229.83440255
            b2 = 545171.14839945
            c2 = -327016.69508374
            a3 = -20173.68563579
            b3 = 48599.13254342
            c3 = -29290.96100603
            a4 = -1838.90888364
            b4 = 4491.58838596
            c4 = -2763.82866252
            a5 = -186.98923927
            b5 = 477.04784782
            c5 = -324.75732578
            a6 = -24.48843901
            b6 = 69.43965607
            c6 = -69.145292411
            a7 = -4.33101714
            b7 = 14.47124519
            c7 = -31.66224458
            a8 = -0.68345667
            b8 = 2.6053539
            c8 = -21.99823511            
            
            
            i1 = 15.77742
            i2 = 15.81964
            i3 = 15.96196
            i4 = 16.44162
            i5 = 18.05837
            i6 = 23.50771
            i7 = 41.87498
             
            log_sig_g2 = (e_ion_full < e_el < i1) * (((a1*log_e_el + b1)* log_e_el + c1)) + \
                         (i1 <= e_el < i2) * (((a2*log_e_el + b2)* log_e_el + c2)) + \
                         (i2 <= e_el < i3) * (((a3*log_e_el + b3)* log_e_el + c3)) + \
                         (i3 <= e_el < i4) * (((a4*log_e_el + b4)* log_e_el + c4)) + \
                         (i4 <= e_el < i5) * (((a5*log_e_el + b5)* log_e_el + c5)) + \
                         (i5 <= e_el < i6) * (((a6*log_e_el + b6)* log_e_el + c6)) + \
                         (i6 <= e_el < i7) * (((a7*log_e_el + b7)* log_e_el + c7)) + \
                         (i7 <= e_el) * (((a8*log_e_el + b8)* log_e_el + c8))
            sig_g2 = 10**log_sig_g2
            if (e_el < e_ion_full):
                sig_g2 = 0.

            sig_g1 = 0.
            sig_g3 = 0.

            # Scale by heavy density
            sig_g0 *= nn 
            sig_g2 *= nn

            sig_g1 = 0.
            sig_g3 = 0.

            # Currently only elastic + ionization 
            sig_tot = sig_g0 + sig_g2
            P_coll = 1 - math.exp(-dt*v_mag*sig_tot)

            # Determine which collision occurs 
            pcst = P_coll *(1./sig_tot)
            sig_range_g0 = sig_g0 * pcst
            sig_range_g2 = sig_range_g0 + sig_g2 * pcst

            coll_indicator_g2 = (sig_range_g0 < d_R_vec[i,0] < sig_range_g2)

            ## G0: ELASTIC
            if (d_R_vec[i,0] < sig_range_g0):
                d_collct_ar[i,0] = d_tosum_ar[i,0]
                d_g0_ar[i] = 1.
                ## Original electron deflection direction (vscat)      
                cos_chi = 2*d_R_vec[i,1] - 1
                chi = math.acos(cos_chi)
                phi = 2*math.pi*d_R_vec[i,2]

                v_mag_new = v_mag * math.sqrt(1 - (2*m_e/m_n))
                d_vx = v_mag_new * math.sin(chi) * math.cos(phi)
                d_vy = v_mag_new * math.sin(chi) * math.sin(phi)
                d_vz = v_mag_new * math.cos(chi)

            ## G2 WITH SPLITTING FOR REGULAR HIGH ENERGIES
            elif (coll_indicator_g2):
                ## Original electron deflection direction (vscat)      
                cos_chi = 2*d_R_vec[i,1] - 1
                chi = math.acos(cos_chi)
                phi = 2*math.pi*d_R_vec[i,2]
            
                write_ind = cuda.atomic.add(d_curr_count, 0, 1)
                write_ind += Nc 
                
                # Energy splitting
                e_ej = abs(0.5*(e_el - e_ion_full))
                e_scat = e_ej 
               
                ## Original electron gets split
                v_mag_new = math.sqrt(2*e_scat / (j_ev*m_e))
                d_vx = v_mag_new * math.sin(chi) * math.cos(phi)
                d_vy = v_mag_new * math.sin(chi) * math.sin(phi)
                d_vz = v_mag_new * math.cos(chi)

                d_tosum_ar[i,1] = 1.
                d_collct_ar[i,2] = 1.
                d_g2_ar[i] = 1.

                # Ejected particle exit angle
                cos_chi = 2*d_R_vec[i,3] - 1
                chi = math.acos(cos_chi)
                phi = 2*math.pi*d_R_vec[i,5]

                t_vx = v_mag_new * math.sin(chi) * math.cos(phi)
                t_vy = v_mag_new * math.sin(chi) * math.sin(phi)
                t_vz = v_mag_new * math.cos(chi)

                # Write new particle data
                d_data_ar[write_ind,3] = t_vx
                d_data_ar[write_ind,4] = t_vy
                d_data_ar[write_ind,5] = t_vz
                d_data_ar[write_ind,0] = d_xx
                d_data_ar[write_ind,1] = d_xy
                d_data_ar[write_ind,2] = d_xz
                en_x = t_vx*t_vx
                en_y = t_vy*t_vy
                en_z = t_vz*t_vz
                d_tosum_ar[write_ind,3] = 0.5*m_e*j_ev*(en_x + en_y + en_z)
                d_tosum_ar[write_ind,0] = d_tosum_ar[i,0]

            ## X-ADV
            d_xx += 100 * dt * d_vx
            d_xy += 100 * dt * d_vy
            d_xz += 100 * dt * d_vz      

            # V_ADV
            index = d_forE_xbins[i]
            d_vx -= dt * (q_e / m_e) * (100*d_E_ar[index])

            ## BOUNDARY CONDITIONS 
            is_valid = (0 < d_xx < L and d_w > 0)

            d_tosum_ar[i,0]= (is_valid) * d_w            
            d_data_ar[i,0] = (is_valid) * d_xx + (1-is_valid) * 0.00000001# don't kill for cuda reduction purposes with empty spot (long story)
            d_data_ar[i,1] = (is_valid) * d_xy 
            d_data_ar[i,2] = (is_valid) * d_xz  
           
            d_vx = (is_valid) * d_vx
            d_vy = (is_valid) * d_vy
            d_vz = (is_valid) * d_vz

            # For temperature calculations 
            en_x = d_vx*d_vx
            en_y = d_vy*d_vy
            en_z = d_vz*d_vz
            d_tosum_ar[i,3] = 0.5*m_e*j_ev*(en_x + en_y + en_z)

            # Putting the v back in
            d_data_ar[i,3] = d_vx
            d_data_ar[i,4] = d_vy
            d_data_ar[i,5] = d_vz



def electron_kernel(
    Nc, d_data_ar, d_tosum_ar, d_collct_ar,
    d_E_ar, cp_R_vec, d_R_vec,
    d_currxbins, d_forE_xbins,
    nn, dt_ratio, dt,
    d_curr_count, d_g0_ar, d_g2_ar, d_gas_v_ar,
    num_blocks, threads_per_block,
):

    cp_R_vec[0:Nc,:] = cp.random.rand(Nc, 7)

    electron_kernel_1D[num_blocks, threads_per_block](
        Nc, d_data_ar, d_tosum_ar, d_collct_ar, d_E_ar, d_R_vec, d_currxbins, d_forE_xbins, nn, dt_ratio, dt, d_curr_count, d_g0_ar, d_g2_ar, d_gas_v_ar)
