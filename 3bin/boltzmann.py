import cupy as cp
import numba
from numba import cuda
import numpy as np
import math
import sys
from scipy.linalg import cholesky_banded

try:
    from kernels import reduction
except:
    from .kernels import reduction


rank = 0
NUM_GPUS = 1
print("NUM GPUS = ", NUM_GPUS)

cuda.select_device(rank % NUM_GPUS)
cp.cuda.runtime.setDevice(rank % NUM_GPUS)

class BoltzmannDSMC:
    def __init__(self, num_steps, name, restart_cyclenum, Nc, gridpts, bnf, num_nodes, pyk, resample_freq, in_dt, in_spc):

        self.reduction: Callable
        self.EF_kernel: Callable
        self.electron_kernel: Callable
        self.heavies_kernel_fluid: Callable
        self.post_processing: Callable
        self.data_write: Callable
        self.bin_particles: Callable
        self._init_kernels(pyk)

        self.pyk: bool = pyk

        # used for indexing in some places for clarity
        self.xx: int = 0
        self.xy: int = 1
        self.xz: int = 2
        self.vx: int = 3
        self.vy: int = 4
        self.vz: int = 5
        self.wt: int = 0
        self.en: int = 3

        ## Heavy parameters
        P = 0.1
        self.pressure = P
        self.nn = 3.22e16 * P
        self.nn_Di = 2.07e18
        self.nn_mui = 4.65e19
        self.nn_Ds = 2.42e18
        self.D_i = self.nn_Di / self.nn
        self.D_s = self.nn_Ds / self.nn
        self.mu_i = self.nn_mui / self.nn
        self.epsilon = 5.5263e5  # permitivity
        self.m_e = 9.10938*10**(-31)
        self.j_ev = 6.241509*(10**18)

        ## 1-D electric field
        self.freq = 13.56e6  
        self.tau = 1.0 / self.freq
        self.V0 = 100

        ## Base numerical stuff findme
        self.N: int = gridpts - 1
        self.big_N: int = bnf*(gridpts)
        self.L: float = 2.54
        self.tau: float = 1.0 / self.freq  # period
        self.dx: float = self.L / self.N
        
        ## Time
        self.steps_per_cycle: int = in_spc
        self.dt_big: float = (self.tau / in_dt) 
        self.dt_ratio: int = 1
        self.dt_el: float = self.dt_big / self.dt_ratio
        self.dt_col = self.dt_el
        self.curr_t: float = 0.0

        ## Particles
        self.Nc_list = []
        self.Na_list = []
        self.Nmax_list = []
        self.Nnew_list = []
        self.Nmax_fac = 100
        Nsplit = int(Nc/NUM_GPUS) # taking input
        for ng in range(NUM_GPUS):
            self.Nc_list.append(Nsplit)
            self.Na_list.append(Nsplit)
            self.Nmax_list.append(int(self.Nmax_fac*Nsplit))
            self.Nnew_list.append(0)
         
        ## Setup work arrays
        self.E_ar = np.zeros(self.N)
        self.Ji_ar = np.zeros(self.N)
        self.Js_ar = np.zeros(self.N)
        self.ni_rhs = np.zeros(self.N + 1)
        self.ns_rhs = np.zeros(self.N + 1)
        self.V_ar = np.zeros(self.N - 1)
        self.V_rhs = np.zeros(self.N - 1)
        self.ne_ar = np.zeros(self.N + 1)
        self.ni_ar = np.ones(self.N + 1)
        self.ns_ar = np.zeros(self.N + 1)
        self.nrg_ar = np.zeros(self.N + 1) 
        self.counter_g0_ar = np.zeros(self.N + 1) 
        self.counter_g1_ar = np.zeros(self.N + 1) 
        self.counter_g2_ar = np.zeros(self.N + 1) 
        self.counter_g3_ar = np.zeros(self.N + 1) 
        self.Te_ar = np.zeros(self.N + 1) 
       
        ## For P2C sum algorithm
        self.data_out_list = []
        self.data_out_np_list = []
        self.data_out_list2 = []
        self.data_out_np_list2 = []
        self.data_out_list3 = []
        self.data_out_np_list3 = []
        self.temp_x_list = []
        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            self.data_out_list.append(cp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list.append(np.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_list2.append(cp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list2.append(np.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_list3.append(cp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list3.append(np.zeros((self.big_N, 4), dtype=np.float64))

            self.temp_x_list.append(cp.zeros(self.Nmax_list[ng]))

        ## For MPI (currently disabled)
        self.ni_src = np.zeros(self.N + 1)
        self.ns_src = np.zeros(self.N + 1)
        self.ne_ar_mpi = np.zeros(self.N + 1)
        self.ni_src_mpi = np.zeros(self.N + 1)
        self.ns_src_mpi = np.zeros(self.N + 1)
        self.nrg_ar_mpi = np.zeros(self.N + 1)

        ## GPU Tuning
        self.threads_per_block: int = 64
        self.num_blocks: int = 1024

        ## Initialize for voltage solve
        self.Vc_diag, self.Vc_lower_diag, self.V_tempy = self._chol_EF(self.N, self.dx)

       
        ## 3 bin setup for this test problem
        self.target_ppc_LE_ar = int(0.7*self.Nc_list[0]/self.N) * cp.ones(self.N+1).astype(int)
        self.target_ppc_ME_ar = int(0.2*self.Nc_list[0]/self.N) * cp.ones(self.N+1).astype(int)
        self.target_ppc_HE_ar = int(0.1*self.Nc_list[0]/self.N) * cp.ones(self.N+1).astype(int)
        self.ttLE = self.target_ppc_LE_ar[1]
        self.ttME = self.target_ppc_ME_ar[1]
        self.ttHE = self.target_ppc_HE_ar[1]
        
        ## Particle initizliation
        IC_flag = 1 # =0 is not used, =1 is todd soln, =3 is milinda soln

        if (IC_flag == 1 and restart_cyclenum > 0):
            print("CONFUSING INPUT SETTINGS. TODD FLAG AND RESTART BOTH ON. EXITING")
            sys.exit()

        elif (restart_cyclenum > 0):
            print("RESTARTING WITH DATA FROM CYCLE # ", restart_cyclenum)
            
            self.restart_cyclenum = restart_cyclenum
            sstr = name
            cyclestr = '_cycle' + str(restart_cyclenum) + '.dat'
        
            # Restart mapping rank data
            for res_rank in range(num_nodes):

                if (rank == res_rank):            
                    cyclestrmpi = '_cycle' + str(restart_cyclenum) + '_mpi' + str(res_rank) + '.dat'
                    nistr = sstr + 'ni' + cyclestr
                    xstr  = sstr + 'e_x0' + cyclestrmpi
                    v0str = sstr + 'e_v0' + cyclestrmpi
                    v1str = sstr + 'e_v1' + cyclestrmpi
                    v2str = sstr + 'e_v2' + cyclestrmpi
                    wtstr = sstr + 'e_wt' + cyclestrmpi
                    fileobj_ni = open(nistr, mode='rb')
                    fileobj_x0 = open(xstr, mode='rb')
                    fileobj_v0 = open(v0str, mode='rb')
                    fileobj_v1 = open(v1str, mode='rb')
                    fileobj_v2 = open(v2str, mode='rb')
                    fileobj_wt = open(wtstr, mode='rb')

                    xinput  = np.fromfile(fileobj_x0, dtype=np.double)
                    v0input = np.fromfile(fileobj_v0, dtype=np.double)
                    v1input = np.fromfile(fileobj_v1, dtype=np.double)
                    v2input = np.fromfile(fileobj_v2, dtype=np.double)
                    wtinput = np.fromfile(fileobj_wt, dtype=np.double)
                    niinput = np.fromfile(fileobj_ni, dtype=np.double)

                    nz_res_inds = np.where(wtinput != 0)[0]
                    t_Nc = len(nz_res_inds)

                    cp.cuda.Device(0).use()
                    self.big_data_ar_list = []
                    self.big_tosum_ar_list = []
                    self.big_collct_ar_list = []
                    self.big_curr_xbins_list = []
                    self.big_forE_xbins_list = []
                    self.nni_list = []
                    self.nni_list.append(0)

                    print("RESTARTING with ", t_Nc, "particles on rank ", rank)
                    t_Nmax = 60000000            

                    self.Nc_list[0] = t_Nc
                    self.Na_list[0] = t_Nc
                    self.Nmax_list[0] = t_Nmax
                    self.Nnew_list[0] = 0

                    temp_big_data_ar = cp.zeros((t_Nmax,6))
                    temp_big_tosum_ar = cp.zeros((t_Nmax,4))

                    # Duping particles
                    temp_big_data_ar[0:t_Nc,self.xx] = cp.array(xinput[nz_res_inds])
                    temp_big_data_ar[0:t_Nc,self.vx] = cp.array(v0input[nz_res_inds])
                    temp_big_data_ar[0:t_Nc,self.vy] = cp.array(v1input[nz_res_inds])
                    temp_big_data_ar[0:t_Nc,self.vz] = cp.array(v2input[nz_res_inds])
                    temp_big_tosum_ar[0:t_Nc,self.wt] = cp.array(wtinput[nz_res_inds]) 
                    
                    en_xx = cp.multiply(temp_big_data_ar[0:t_Nc,self.vx], temp_big_data_ar[0:t_Nc,self.vx])
                    en_yy = cp.multiply(temp_big_data_ar[0:t_Nc,self.vy], temp_big_data_ar[0:t_Nc,self.vy])
                    en_zz = cp.multiply(temp_big_data_ar[0:t_Nc,self.vz], temp_big_data_ar[0:t_Nc,self.vz])
                    temp_big_tosum_ar[0:t_Nc,self.en] = 0.5*self.m_e*self.j_ev*cp.add(en_xx, cp.add(en_yy, en_zz))

                    temp_big_collct_ar = cp.zeros((t_Nmax,4))
                    self.big_collct_ar_list.append(temp_big_collct_ar)

                    # set up bins
                    curr_xbins_temp = cp.zeros(t_Nmax).astype(int)
                    forE_xbins_temp = cp.zeros(t_Nmax).astype(int)
                    curr_xbins_temp[0:t_Nc] = ((temp_big_data_ar[0:t_Nc, self.xx] + 0.5*self.dx)/self.dx).astype(int)
                    forE_xbins_temp[0:t_Nc] = (temp_big_data_ar[0:t_Nc, self.xx]/self.dx).astype(int)
                    self.big_data_ar_list.append(temp_big_data_ar)
                    self.big_tosum_ar_list.append(temp_big_tosum_ar)
                    self.big_curr_xbins_list.append(curr_xbins_temp)
                    self.big_forE_xbins_list.append(forE_xbins_temp) 

                    self.ni_ar = niinput

                    self.ns_ar = np.zeros(self.N+1)

        # Initializing to condition from Liu
        elif (IC_flag == 1):

            print("Initializing with prescribed IC data on rank = ", rank)

            temp_x_ar = np.linspace(0, self.L, self.N + 1)
            jab_ne = self.pressure * (1e7 + 1e9*(1-temp_x_ar/self.L)**2 * (temp_x_ar/self.L)**2)
            self.ni_ar[:] = np.copy(jab_ne)
            jab_Te = 0.5*np.ones(self.N+1)

            # For generating electrons
            ng = 0
            cp.cuda.Device(0).use()
            Nc_temp = self.Nc_list[ng]
            
            ## FOR FIXED PPC INIT

            self.Nc_list[ng] = int(np.sum(self.target_ppc_LE_ar)+np.sum(self.target_ppc_HE_ar)+int(np.sum(self.target_ppc_ME_ar)))
            self.Nmax_list[ng] = int(self.Nmax_fac * self.Nc_list[ng])
            Nmax_temp = self.Nmax_list[ng]

            ## Setting up particles and particle bins
            kTm = 1.5 * 1.17255 * 10**11 # 1 eV
         
            self.big_data_ar_list = []
            self.big_tosum_ar_list = []
            self.big_curr_xbins_list = []
            self.big_forE_xbins_list = []
            self.nni_list = []
            
            tot_data_ar = cp.zeros((Nmax_temp, 6))
            tot_tosum_ar = cp.zeros((Nmax_temp, 4))
            
            curr_Nc = 0
           
            for ii in range(self.N+1):
                ## FOR FIXED PPC INIT
                num_cell = int(self.target_ppc_LE_ar[ii]+self.target_ppc_ME_ar[ii]+self.target_ppc_HE_ar[ii])
                # put in the x's
                tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_x_ar[ii]
                if (ii == 0):
                    tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_x_ar[ii] + self.dx/4.0
                elif (ii == self.N):
                    tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_x_ar[ii] - self.dx/4.0
                else:
                    tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_x_ar[ii]

                # put in the v's
                E_cell = jab_Te[ii] * kTm
                cov = [[E_cell, 0, 0], [0, E_cell, 0], [0, 0, E_cell]]
                tot_data_ar[curr_Nc : curr_Nc+num_cell, self.vx:self.vz+1] = cp.random.multivariate_normal([0,0,0], cov, num_cell)
      
                m_e = 9.10938*10**(-31)
                j_ev = 6.241509*(10**18)

                ## FOR FIXED PPC INIT
                temp_ww = jab_ne[ii] / num_cell / num_nodes

                tot_tosum_ar[curr_Nc : curr_Nc+num_cell, self.wt].fill(temp_ww)
                curr_Nc = curr_Nc + num_cell
            
            tot_data_ar[curr_Nc:, self.xx] = self.L * cp.random.rand(Nmax_temp-curr_Nc)

            # Energy
            en_xx = cp.multiply(tot_data_ar[0:curr_Nc,self.vx], tot_data_ar[0:curr_Nc,self.vx])
            en_yy = cp.multiply(tot_data_ar[0:curr_Nc,self.vy], tot_data_ar[0:curr_Nc,self.vy])
            en_zz = cp.multiply(tot_data_ar[0:curr_Nc,self.vz], tot_data_ar[0:curr_Nc,self.vz])
            tot_tosum_ar[0:curr_Nc, self.en] = 0.5*self.m_e*self.j_ev*cp.add(en_xx, cp.add(en_yy, en_zz))

            self.big_collct_ar_list = []
            temp_big_collct_ar = cp.zeros((Nmax_temp, 4))
            self.big_collct_ar_list.append(temp_big_collct_ar)
 
            # Shuffling for efficiency 
            cp.random.seed(1)
            cp.random.shuffle(tot_data_ar[0:curr_Nc,:])
            cp.random.seed(1)
            cp.random.shuffle(tot_tosum_ar[0:curr_Nc,:])
            
            # Putting back into list    
            self.big_data_ar_list.append(tot_data_ar)
            self.big_tosum_ar_list.append(tot_tosum_ar)
                
            # Bins            
            curr_xbins_temp = cp.zeros(Nmax_temp).astype(int)
            forE_xbins_temp = cp.zeros(Nmax_temp).astype(int)
            curr_xbins_temp[0:Nc_temp] = ((self.big_data_ar_list[ng][0:Nc_temp, self.xx] + 0.5*self.dx)/self.dx).astype(int)
            forE_xbins_temp[0:Nc_temp] = (self.big_data_ar_list[ng][0:Nc_temp, self.xx]/self.dx).astype(int)
            self.big_curr_xbins_list.append(curr_xbins_temp)
            self.big_forE_xbins_list.append(forE_xbins_temp) 

            # Updating particle count
            self.nni_list.append(0)
        
        ## For reshuffles (CURRENTLY OFF)
        self.need_to_reshuffle = True

        ## Grid arrays
        self.ne_ar_list = []
        self.ni_src_list = []
        self.ns_src_list = []
        self.nrg_ar_list = []
        self.counter_g0_ar_list = []
        self.counter_g1_ar_list = []
        self.counter_g2_ar_list = []
        self.counter_g3_ar_list = []
        for ng in range(NUM_GPUS):
            self.ne_ar_list.append(np.zeros(self.N+1))
            self.ni_src_list.append(np.zeros(self.N+1))
            self.ns_src_list.append(np.zeros(self.N+1))
            self.nrg_ar_list.append(np.zeros(self.N+1))
            self.counter_g0_ar_list.append(np.zeros(self.N+1))
            self.counter_g1_ar_list.append(np.zeros(self.N+1))
            self.counter_g2_ar_list.append(np.zeros(self.N+1))
            self.counter_g3_ar_list.append(np.zeros(self.N+1))

        self.ne_ar.fill(0)
        self.ni_src.fill(0)
        self.ns_src.fill(0)
        self.nrg_ar.fill(0)
        

        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()

            self.ne_ar_list[ng], self.ni_src_list[ng], self.ns_src_list[ng], self.nrg_ar_list[ng], self.counter_g0_ar_list[ng], self.counter_g1_ar_list[ng], self.counter_g2_ar_list[ng], self.counter_g3_ar_list[ng], self.Nnew_list[ng] = self.post_processing(
                self.data_out_list[ng],self.data_out_np_list[ng],self.data_out_list2[ng],self.data_out_np_list2[ng],self.data_out_list3[ng], 
                self.data_out_np_list3[ng],self.big_data_ar_list[ng],self.big_tosum_ar_list[ng],self.big_collct_ar_list[ng],self.need_to_reshuffle,self.Na_list[ng],
                self.Nc_list[ng],self.nni_list[ng],self.ne_ar_list[ng],self.ni_src_list[ng],self.ns_src_list[ng],self.nrg_ar_list[ng],
                self.counter_g0_ar_list[ng],self.counter_g1_ar_list[ng],self.counter_g2_ar_list[ng],self.counter_g3_ar_list[ng],
                self.big_N,self.N + 1,self.L,self.dx,self.temp_x_list[ng])

            self.ne_ar += self.ne_ar_list[ng]
            self.ni_src += self.ni_src_list[ng]
            self.ns_src += self.ns_src_list[ng]
            self.nrg_ar += self.nrg_ar_list[ng]

        self.ni_ar[:] = self.ni_ar[:] + self.ni_src[:]
        self.Te_ar.fill(0)
        self.Te_ar[:] = np.divide((2./3.) * self.nrg_ar, self.ne_ar)
        self.Te_ar[:] = np.nan_to_num(self.Te_ar[:])
     
        # Initializing EF
        self.Vcarry_ar, self.E_ar = self.EF_kernel(
            self.E_ar,
            self.ne_ar,
            self.ni_ar,
            self.curr_t,
            self.V_ar,
            self.V_rhs,
            self.Vc_diag,
            self.Vc_lower_diag,
            self.V_tempy,
            self.N,
            self.dx,
            self.epsilon,
            self.V0,
            self.freq,
        )

        ## Random #s, atomics
        self.forgpu_R_vec_list = []
        self.curr_count_list = []

        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            forgpu_R_vec_temp = cp.zeros((self.Nmax_list[ng], 7)) 
            forgpu_R_vec_temp[:,:] = cp.random.rand(self.Nmax_list[ng],7)
            self.forgpu_R_vec_list.append(forgpu_R_vec_temp)

            curr_count_temp = cp.zeros(1).astype(int)
            self.curr_count_list.append(curr_count_temp)


        ## For copying to GPU and device arrays for electron kernel
        self.gpu_E_ar_list = []
        self.gpu_ne_ar_list = []
        self.gpu_ni_ar_list = []
        self.gpu_ns_ar_list = []
        self.gpu_Te_ar_list = []
        self.np_data_ar = np.zeros(5*self.N + 4)
        self.cp_data_ar_list = []
        
        self.d_curr_count_list = []
        self.d_currxbins_list = []
        self.d_forExbins_list = []
        self.d_bigRvec_list = []
        self.d_data_ar_list = []
        self.d_tosum_ar_list = []
        self.d_collct_ar_list = []
        self.d_E_ar_list = []
        self.d_ne_ar_list = []
        self.d_ni_ar_list = []
        self.d_ns_ar_list = []
        self.d_Te_ar_list = []
       
        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            gpu_E_ar_temp = cp.zeros(self.N)
            gpu_ne_ar_temp = cp.zeros(self.N + 1)
            gpu_ni_ar_temp = cp.zeros(self.N + 1)
            gpu_ns_ar_temp = cp.zeros(self.N + 1)
            gpu_Te_ar_temp = cp.zeros(self.N + 1)
            cp_data_ar_temp = cp.zeros(5*self.N + 4)
            self.gpu_E_ar_list.append(gpu_E_ar_temp)
            self.gpu_ne_ar_list.append(gpu_ne_ar_temp)
            self.gpu_ni_ar_list.append(gpu_ni_ar_temp)
            self.gpu_ns_ar_list.append(gpu_ns_ar_temp)
            self.gpu_Te_ar_list.append(gpu_Te_ar_temp)
            self.cp_data_ar_list.append(cp_data_ar_temp)

            d_curr_count_temp = cuda.to_device(self.curr_count_list[ng])
            d_currxbins_temp = cuda.to_device(self.big_curr_xbins_list[ng])
            d_forExbins_temp = cuda.to_device(self.big_forE_xbins_list[ng])
            d_bigRvec_temp = cuda.to_device(self.forgpu_R_vec_list[ng])
            d_data_ar_temp = cuda.to_device(self.big_data_ar_list[ng])
            d_tosum_ar_temp = cuda.to_device(self.big_tosum_ar_list[ng])
            d_collct_ar_temp = cuda.to_device(self.big_collct_ar_list[ng])
            d_E_ar_temp = cuda.to_device(self.gpu_E_ar_list[ng])
            d_ne_ar_temp = cuda.to_device(self.gpu_ne_ar_list[ng])
            d_ni_ar_temp = cuda.to_device(self.gpu_ni_ar_list[ng])
            d_ns_ar_temp = cuda.to_device(self.gpu_ns_ar_list[ng])
            d_Te_ar_temp = cuda.to_device(self.gpu_Te_ar_list[ng])
           
            self.d_curr_count_list.append(d_curr_count_temp)
            self.d_currxbins_list.append(d_currxbins_temp)
            self.d_forExbins_list.append(d_forExbins_temp)
            self.d_bigRvec_list.append(d_bigRvec_temp)
            self.d_data_ar_list.append(d_data_ar_temp)
            self.d_tosum_ar_list.append(d_tosum_ar_temp)
            self.d_collct_ar_list.append(d_collct_ar_temp)
            self.d_E_ar_list.append(d_E_ar_temp)  
            self.d_ne_ar_list.append(d_ne_ar_temp)  
            self.d_ni_ar_list.append(d_ni_ar_temp)  
            self.d_ns_ar_list.append(d_ns_ar_temp)  
            self.d_Te_ar_list.append(d_Te_ar_temp)  
        

        ## For recombination and for counting collisions
        self.collflag_list = []
        self.d_collflag_list = []
        self.did_g0_ar_list = []
        self.did_g1_ar_list = []
        self.did_g2_ar_list = []
        self.did_g3_ar_list = []
        self.d_g0_ar_list = []
        self.d_g1_ar_list = []
        self.d_g2_ar_list = []
        self.d_g3_ar_list = []
        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            
            collflag_temp = cp.zeros(self.Nmax_list[ng])
            d_collflag_temp = cuda.to_device(collflag_temp)
            self.collflag_list.append(collflag_temp)
            self.d_collflag_list.append(d_collflag_temp)

            did_g0_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g1_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g2_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g3_ar_temp = cp.zeros(self.Nmax_list[ng])
            self.did_g0_ar_list.append(did_g0_ar_temp)
            self.did_g1_ar_list.append(did_g1_ar_temp)
            self.did_g2_ar_list.append(did_g2_ar_temp)
            self.did_g3_ar_list.append(did_g3_ar_temp)
            
            d_g0_ar_temp = cuda.to_device(self.did_g0_ar_list[ng])
            d_g1_ar_temp = cuda.to_device(self.did_g1_ar_list[ng])
            d_g2_ar_temp = cuda.to_device(self.did_g2_ar_list[ng])
            d_g3_ar_temp = cuda.to_device(self.did_g3_ar_list[ng])
            self.d_g0_ar_list.append(d_g0_ar_temp)
            self.d_g1_ar_list.append(d_g1_ar_temp)
            self.d_g2_ar_list.append(d_g2_ar_temp)
            self.d_g3_ar_list.append(d_g3_ar_temp)


        # Output frequency, and thus total # outputs
        self.output_freq = int(self.steps_per_cycle)
        self.num_outputs = int(num_steps/ self.output_freq)
        self.out_zdne = np.zeros(self.num_outputs)
        self.out_zdTe = np.zeros(self.num_outputs)
       

        ## For data output 
        self.ne_CA = np.zeros(self.N+1)
        self.ni_CA = np.zeros(self.N+1)
        self.ns_CA = np.zeros(self.N+1)
        self.Te_CA = np.zeros(self.N+1)
        self.E_CA = np.zeros(self.N)
        self.V_CA = np.zeros(self.N-1)
                    
        self.g0_coeffs_ar = np.zeros(self.N + 1) 
        self.g2_coeffs_ar = np.zeros(self.N + 1) 

        ## For ion solver
        self.break_flag = 0

        ## Work arrays for bin handling
        self.bin_write_inds_LE_ar = cp.zeros(self.N+1).astype(int)
        self.d_bin_write_inds_LE_ar = cuda.to_device(self.bin_write_inds_LE_ar)
        self.bin_write_inds_ME_ar = cp.zeros(self.N+1).astype(int)
        self.d_bin_write_inds_ME_ar = cuda.to_device(self.bin_write_inds_ME_ar)
        self.bin_write_inds_HE_ar = cp.zeros(self.N+1).astype(int)
        self.d_bin_write_inds_HE_ar = cuda.to_device(self.bin_write_inds_HE_ar)

        self.max_ppc_realistic_LE = int(10.0*np.max(self.target_ppc_LE_ar))
        self.max_ppc_realistic_ME = int(10.0*np.max(self.target_ppc_ME_ar))
        self.max_ppc_realistic_HE = int(10.0*np.max(self.target_ppc_HE_ar))

        self.cell_particles_LE_ar = cp.zeros((self.max_ppc_realistic_LE,self.N+1)).astype(int)
        self.d_cell_particles_LE_ar = cuda.to_device(self.cell_particles_LE_ar)
        self.cell_particles_ME_ar = cp.zeros((self.max_ppc_realistic_ME,self.N+1)).astype(int)
        self.d_cell_particles_ME_ar = cuda.to_device(self.cell_particles_ME_ar)
        self.cell_particles_HE_ar = cp.zeros((self.max_ppc_realistic_HE,self.N+1)).astype(int)
        self.d_cell_particles_HE_ar = cuda.to_device(self.cell_particles_HE_ar)
       
        self.min_wt_allow = 1e-1
        self.need_to_resample = False 
        self.resample_freq = resample_freq

        ## Initial bins for x, E 
        self.big_forE_xbins_list[ng][0:self.Nnew_list[ng]] = (self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx]/self.dx).astype(int)
        self.big_curr_xbins_list[ng][0:self.Nnew_list[ng]] = ((self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx] + 0.5*self.dx)/self.dx).astype(int)

        ## For non-zero gas temperature
        self.Tg = 300. / 11606.
        self.kTm = 1.5 * 1.17255e11 # 1 eV
        self.Eg = self.Tg * self.kTm
        self.big_gas_v_ar = cp.zeros((self.Nmax_list[0],3))
        self.d_gas_v_ar = cuda.to_device(self.big_gas_v_ar)
        self.cov = [[self.Eg, 0, 0], [0, self.Eg, 0], [0, 0, self.Eg]]


    ## MAIN TIMESTEPPING LOOP
    def run(self, num_steps, name, restart_cyclenum):
        cyclestep = 0 
        
        ng = 0
        print("INIT NC = ", self.Nc_list[ng])
        for bigstep in range(num_steps):

            if (rank == 0 and bigstep % 100 == 0):
                print("Step ", bigstep, ". Nc = ", self.Nc_list[0]) 

            ## COPY->GPU
            self._copy_to_GPU()

            for ng in range(NUM_GPUS):
                cp.cuda.Device(ng).use()

                ## PARTICLE RESAMPLING
                if (bigstep % self.resample_freq == 0):
                    self.need_to_resample = False
                    self._check_for_resampling()
                    if (self.need_to_resample == True):
                        self._do_resampling()

                ## Draw non-zero gas temperature samples
                self.big_gas_v_ar[0:self.Nc_list[ng],:] = cp.random.multivariate_normal([0,0,0], self.cov, self.Nc_list[ng])

                ## Electron Coll+Push kernel
                self.Ncoll_offset = self.electron_kernel(
                    self.Nc_list[ng],
                    self.d_data_ar_list[ng],
                    self.d_tosum_ar_list[ng],
                    self.d_collct_ar_list[ng],
                    self.d_E_ar_list[ng],
                    self.forgpu_R_vec_list[ng],
                    self.d_bigRvec_list[ng],
                    self.d_currxbins_list[ng],
                    self.d_forExbins_list[ng],
                    1e6 * self.nn,
                    self.dt_ratio,
                    self.dt_el,
                    self.d_curr_count_list[ng],
                    self.d_g0_ar_list[ng],
                    self.d_g2_ar_list[ng],
                    self.d_gas_v_ar,
                    self.num_blocks,
                    self.threads_per_block,
                )

                ## Count number of new particles
                self.nni_list[ng] = int(self.curr_count_list[ng][0])

                ## C2P Calculations
                self.need_to_reshuffle = True
                self.ne_ar_list[ng], self.ni_src_list[ng], self.ns_src_list[ng], self.nrg_ar_list[ng], self.counter_g0_ar_list[ng], self.counter_g1_ar_list[ng], self.counter_g2_ar_list[ng], self.counter_g3_ar_list[ng], self.Nnew_list[ng] = self.post_processing(
                    self.data_out_list[ng], self.data_out_np_list[ng], 
                    self.data_out_list2[ng], self.data_out_np_list2[ng], 
                    self.data_out_list3[ng], self.data_out_np_list3[ng], 
                    self.big_data_ar_list[ng], self.big_tosum_ar_list[ng], self.big_collct_ar_list[ng],
                    self.need_to_reshuffle,
                    self.Na_list[ng], self.Nc_list[ng], self.nni_list[ng], 
                    self.ne_ar_list[ng], self.ni_src_list[ng], self.ns_src_list[ng], self.nrg_ar_list[ng], 
                    self.counter_g0_ar_list[ng], self.counter_g1_ar_list[ng], self.counter_g2_ar_list[ng], self.counter_g3_ar_list[ng],
                    self.big_N, self.N + 1,
                    self.L, self.dx,
                    self.temp_x_list[ng],
                )
                self.need_to_reshuffle = False

                self.ne_ar.fill(0.) 
                self.ni_src.fill(0.)
                self.nrg_ar.fill(0.)
                self.nrg_ar_mpi.fill(0.)
                self.ne_ar += self.ne_ar_list[ng]
                self.ni_src += self.ni_src_list[ng]
                self.ns_src += self.ns_src_list[ng]
                self.nrg_ar += self.nrg_ar_list[ng]

            ## Update grid values
            self.ni_ar[:] = self.ni_ar[:] + self.ni_src[:]
            self.Te_ar.fill(0)
            self.Te_ar[:] = np.divide((2./3.) * self.nrg_ar, self.ne_ar)
            self.Te_ar[:] = np.nan_to_num(self.Te_ar[:])
            self.g2_coeffs_ar[:] = self.g2_coeffs_ar[:] + self.ni_src[:]

            ## Fills and binning 
            for ng in range(NUM_GPUS): 
                cp.cuda.Device(ng).use()

                self.big_data_ar_list[ng][self.Nnew_list[ng]:self.vx:self.vx+1].fill(0)
                self.big_tosum_ar_list[ng][self.Nnew_list[ng]:,:].fill(0)
                self.big_tosum_ar_list[ng][:,1:3].fill(0)
                
                self.big_forE_xbins_list[ng][0:self.Nnew_list[ng]] = (self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx]/self.dx).astype(int)
                self.big_curr_xbins_list[ng][0:self.Nnew_list[ng]] = ((self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx] + 0.5*self.dx)/self.dx).astype(int)
            
                ## Update # particles
                self.Nc_list[ng] = self.Nnew_list[ng]
            
            # Update time
            self.curr_t += self.dt_big
            cyclestep = cyclestep+1
          
            ## Grid Solves - Fluid + EF
            self.ni_ar, self.ns_ar, self.break_flag = self.heavies_kernel_fluid(self.break_flag,
                self.E_ar, self.ni_ar, self.ni_rhs, self.Ji_ar,
                self.ns_ar, self.ns_rhs, self.Js_ar, self.dx,
                self.dt_big, self.mu_i, self.D_i,self.D_s, self.N
            )
            
            self.Vcarry_ar, self.E_ar = self.EF_kernel(
                self.E_ar, self.ne_ar, self.ni_ar, self.curr_t,
                self.V_ar, self.V_rhs, self.Vc_diag, self.Vc_lower_diag,
                self.V_tempy, self.N, self.dx, self.epsilon, self.V0, self.freq
            )

            # Storing cycle averaged fields
            if ((bigstep+1) % self.steps_per_cycle != 0):
                self.ne_CA[:] = self.ne_CA[:] + self.ne_ar 
                self.ni_CA[:] = self.ni_CA[:] + self.ni_ar 
                self.ns_CA[:] = self.ns_CA[:] + self.ns_ar 
                self.Te_CA[:] = self.Te_CA[:] + self.Te_ar 
                self.E_CA[:] = self.E_CA[:] + self.E_ar 
                self.V_CA[:] = self.V_CA[:] + self.Vcarry_ar 

            # Data output
            if ((bigstep+1) % self.steps_per_cycle == 0):
                cyclestep = 0
                print("Writing data for cycle ", (bigstep+1)/self.steps_per_cycle)
                sys.stdout.flush()
                self.Te_CA[:] = self.Te_CA[:] / self.steps_per_cycle
                self.ne_CA[:] = self.ne_CA[:] / self.steps_per_cycle
                self.ni_CA[:] = self.ni_CA[:] / self.steps_per_cycle
                self.E_CA[:] = self.E_CA[:] / self.steps_per_cycle
                self.V_CA[:] = self.V_CA[:] / self.steps_per_cycle
                
                newname = name
                self.data_write(rank, bigstep, newname, self.output_freq, restart_cyclenum, self.Nc_list[0],
                                self.big_data_ar_list[0], self.big_tosum_ar_list[0], 
                                self.ne_ar, self.ni_ar, self.ns_ar, self.Te_ar, self.E_ar, self.V_ar,
                                self.g0_coeffs_ar, self.g2_coeffs_ar,
                                self.ne_CA, self.ni_CA, self.ni_CA, self.Te_CA, self.E_CA, self.V_CA,
                )
    
                # Reset for next cycle
                self.ne_CA.fill(0)
                self.ni_CA.fill(0)
                self.ns_CA.fill(0)
                self.Te_CA.fill(0)
                self.E_CA.fill(0)
                self.V_CA.fill(0)
                self.g0_coeffs_ar.fill(0)
                self.g2_coeffs_ar.fill(0)


    def  _copy_to_GPU(self):

        # Concatenate on CPU
        self.np_data_ar[0:self.N] = self.E_ar[0:self.N]
        self.np_data_ar[self.N : 2*self.N + 1] = self.ne_ar[0 : self.N + 1]
        self.np_data_ar[2*self.N + 1 : 3*self.N + 2] = self.ni_ar[0:self.N + 1]
        self.np_data_ar[3*self.N + 2: 4*self.N + 3] = self.ns_ar[0:self.N + 1]
        self.np_data_ar[4*self.N + 3: 5*self.N + 4] = self.Te_ar[0:self.N + 1]
       
        # Copy to each GPU
        for ng in range(NUM_GPUS):
            cp.cuda.Device(ng).use()
            
            self.cp_data_ar_list[ng][:] = cp.asarray(self.np_data_ar[:])
            
            self.gpu_E_ar_list[ng][:] = self.cp_data_ar_list[ng][0 : self.N] 
            self.gpu_ne_ar_list[ng][:] = self.cp_data_ar_list[ng][self.N : 2*self.N+1] 
            self.gpu_ni_ar_list[ng][:] = self.cp_data_ar_list[ng][2*self.N+1 : 3*self.N+2] 
            self.gpu_ns_ar_list[ng][:] = self.cp_data_ar_list[ng][3*self.N+2 : 4*self.N+3] 
            self.gpu_Te_ar_list[ng][:] = self.cp_data_ar_list[ng][4*self.N+3 : 5*self.N+4] 

            self.curr_count_list[ng].fill(0)
            self.collflag_list[ng].fill(False)

    ## Set up arrays for the voltage solver
    def _chol_EF(self, N, dx):
        Ac = np.zeros((2,N-1))
        Ac[0,1:N-1] = -1./(dx**2)
        Ac[1,0:N-1] = 2./(dx**2)
        c = cholesky_banded(Ac) 
        Vc_lower_diag = np.zeros(N-1)
        Vc_lower_diag[1:N-1] = np.copy(c[0,1:N-1])
        Vc_diag = np.zeros(N-1)
        Vc_diag[0:N-1] = np.copy(c[1,0:N-1])

        V_tempy = np.zeros(N + 1)
        return (Vc_diag, Vc_lower_diag, V_tempy)

    ## Set up 
    def _init_kernels(self, pyk):
        """
        Initialize the kernels stored as member variables

        :param pyk: whether to use pykokkos versions of kernels
        """

        if (pyk):
            print("USING PYKOKKOS")
        else:
            print("USING NUMBA")

        reduction.init_reduction(pyk)

        try:
            # NUMBA    
            from kernels.bin_particles import bin_particles 
            from kernels.electron_kernel_numba import electron_kernel
            from kernels.EF_kernel_1D import EF_kernel
            from kernels.heavies_kernel_1D import heavies_kernel_fluid
            from kernels.post_processing_1D import post_processing
            from kernels.data_write_1D import data_write        
       
        except Exception as E:
            # NUMBA    
            from .kernels.bin_particles import bin_particles 
            from .kernels.electron_kernel_numba import electron_kernel
            from .kernels.EF_kernel_1D import EF_kernel
            from .kernels.heavies_kernel_1D import heavies_kernel_fluid
            from .kernels.post_processing_1D import post_processing
            from .kernels.data_write_1D import data_write        

        self.EF_kernel = EF_kernel
        self.electron_kernel = electron_kernel
        self.heavies_kernel_fluid = heavies_kernel_fluid
        self.post_processing = post_processing
        self.data_write = data_write
        self.bin_particles = bin_particles


    ## Resampling execution
    def _do_resampling(self):
        ng = 0
        for cc in range(0, self.N+1):

            tppc_LE = int(self.target_ppc_LE_ar[cc])
            tppc_ME = int(self.target_ppc_ME_ar[cc])
            tppc_HE = int(self.target_ppc_HE_ar[cc])
            curr_ppc_LE = int(self.bin_write_inds_LE_ar[cc])
            curr_ppc_ME = int(self.bin_write_inds_ME_ar[cc])
            curr_ppc_HE = int(self.bin_write_inds_HE_ar[cc])

            ## Too few low energies
            if (0 < curr_ppc_LE < 0.5*self.target_ppc_LE_ar[cc]):
                ids_LE = self.cell_particles_LE_ar[0:curr_ppc_LE,cc]
                HW_ids_temp = cp.where(self.big_tosum_ar_list[ng][ids_LE,self.wt] > self.min_wt_allow)[0]
                HW_ids = ids_LE[HW_ids_temp]

                if (len(HW_ids) > 0):
                    P_wts = self.big_tosum_ar_list[ng][HW_ids,self.wt]/cp.sum(self.big_tosum_ar_list[ng][HW_ids,self.wt])
                    keepers = cp.random.choice(HW_ids,2*tppc_LE-curr_ppc_LE,True,p=P_wts)
                    unique_ids, unique_cts = cp.unique(keepers,return_counts=True)
                    n_unique = len(unique_ids)

                    for vv in range(n_unique): 
                        curr_copies = int(unique_cts[vv])
                        self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] = self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] / (curr_copies+1)
                        self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,6))*self.big_data_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,4))*self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.Nc_list[ng] = self.Nc_list[ng] + curr_copies
 
            ## Too many low energies
            if (curr_ppc_LE >= 3.0*self.target_ppc_LE_ar[cc]):
                old_Nc = self.Nc_list[ng]
                curr_ppc_LE = min(curr_ppc_LE, self.max_ppc_realistic_LE)
                temp_tppc_LE = int(0.6*self.target_ppc_LE_ar[cc])

                ids_LE = self.cell_particles_LE_ar[0:curr_ppc_LE,cc]
                original_wt = cp.sum(self.big_tosum_ar_list[ng][ids_LE,self.wt])

                # v's and Te before
                v_cell_x = cp.sum(self.big_data_ar_list[ng][ids_LE,self.vx]*self.big_tosum_ar_list[ng][ids_LE,self.wt])/original_wt
                v_cell_y = cp.sum(self.big_data_ar_list[ng][ids_LE,self.vy]*self.big_tosum_ar_list[ng][ids_LE,self.wt])/original_wt
                v_cell_z = cp.sum(self.big_data_ar_list[ng][ids_LE,self.vz]*self.big_tosum_ar_list[ng][ids_LE,self.wt])/original_wt
                T_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][ids_LE,self.wt]*self.big_tosum_ar_list[ng][ids_LE,self.en] / (original_wt))

                # new wt for particles
                wbar = original_wt / temp_tppc_LE

                # use strata sampling for partial sum array 
                # to select kept particles
                p_use = self.big_tosum_ar_list[ng][ids_LE,self.wt] / original_wt
                w_cumsum = cp.cumsum(p_use)
                Rvals = (1/temp_tppc_LE) * (cp.random.rand(temp_tppc_LE) + cp.arange(temp_tppc_LE))
                keep_ids = cp.searchsorted(w_cumsum,Rvals)

                num_keepids = len(keep_ids)
                self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_data_ar_list[ng][ids_LE[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_tosum_ar_list[ng][ids_LE[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,self.wt] = wbar*cp.ones(num_keepids)

                self.big_data_ar_list[ng][ids_LE,:] = cp.zeros((curr_ppc_LE,6))
                self.big_tosum_ar_list[ng][ids_LE,:] = cp.zeros((curr_ppc_LE,4))

                keep_ids = cp.arange(self.Nc_list[ng],self.Nc_list[ng]+num_keepids) 

                # adjustment
                Te_hat_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][keep_ids,self.wt] * self.big_tosum_ar_list[ng][keep_ids,self.en]) / original_wt
                vhat_cell_x = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vx] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_y = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vy] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_z = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vz] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                Tfac = cp.sqrt(T_cell / Te_hat_cell)

                self.big_data_ar_list[ng][keep_ids,self.vx] = v_cell_x + (self.big_data_ar_list[ng][keep_ids,self.vx] - vhat_cell_x ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vy] = v_cell_y + (self.big_data_ar_list[ng][keep_ids,self.vy] - vhat_cell_y ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vz] = v_cell_z + (self.big_data_ar_list[ng][keep_ids,self.vz] - vhat_cell_z ) * Tfac

                # put in energy
                self.big_tosum_ar_list[ng][keep_ids,self.en] = 0.5*self.m_e*self.j_ev * ( \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vx], self.big_data_ar_list[ng][keep_ids,self.vx]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vy], self.big_data_ar_list[ng][keep_ids,self.vy]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vz], self.big_data_ar_list[ng][keep_ids,self.vz]) )

                self.Nc_list[ng] = self.Nc_list[ng] + num_keepids

            ## Too few medium energies
            if (0 < curr_ppc_ME < 0.5*self.target_ppc_ME_ar[cc]):
                if (curr_ppc_ME == 0):
                    continue
                ids_ME = self.cell_particles_ME_ar[0:curr_ppc_ME,cc]
                HW_ids_temp = cp.where(self.big_tosum_ar_list[ng][ids_ME,self.wt] > self.min_wt_allow)[0]
                HW_ids = ids_ME[HW_ids_temp]

                if (len(HW_ids) > 0):
                    P_wts = self.big_tosum_ar_list[ng][HW_ids,self.wt]/cp.sum(self.big_tosum_ar_list[ng][HW_ids,self.wt])
                    keepers = cp.random.choice(HW_ids,2*tppc_ME-curr_ppc_ME,True,p=P_wts)
                    unique_ids, unique_cts = cp.unique(keepers,return_counts=True)
                    n_unique = len(unique_ids)

                    for vv in range(n_unique): 
                        curr_copies = int(unique_cts[vv])
                        self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] = self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] / (curr_copies+1)
                        self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,6))*self.big_data_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,4))*self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.Nc_list[ng] = self.Nc_list[ng] + curr_copies

            ## Too many medium energies
            if (curr_ppc_ME >= 3.0*self.target_ppc_ME_ar[cc]):
                old_Nc = self.Nc_list[ng]
                curr_ppc_ME = min(curr_ppc_ME, self.max_ppc_realistic_ME)
                temp_tppc_ME = int(0.6*self.target_ppc_ME_ar[cc])

                ids_ME = self.cell_particles_ME_ar[0:curr_ppc_ME,cc]
                original_wt = cp.sum(self.big_tosum_ar_list[ng][ids_ME,self.wt])

                # v's and Te before
                v_cell_x = cp.sum(self.big_data_ar_list[ng][ids_ME,self.vx]*self.big_tosum_ar_list[ng][ids_ME,self.wt])/original_wt
                v_cell_y = cp.sum(self.big_data_ar_list[ng][ids_ME,self.vy]*self.big_tosum_ar_list[ng][ids_ME,self.wt])/original_wt
                v_cell_z = cp.sum(self.big_data_ar_list[ng][ids_ME,self.vz]*self.big_tosum_ar_list[ng][ids_ME,self.wt])/original_wt
                T_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][ids_ME,self.wt]*self.big_tosum_ar_list[ng][ids_ME,self.en] / (original_wt))

                # new wt for particles
                wbar = original_wt / temp_tppc_ME

                # use strata sampling for partial sum array 
                # to select kept particles
                p_use = self.big_tosum_ar_list[ng][ids_ME,self.wt] / original_wt
                w_cumsum = cp.cumsum(p_use)
                Rvals = (1/temp_tppc_ME) * (cp.random.rand(temp_tppc_ME) + cp.arange(temp_tppc_ME))
                keep_ids = cp.searchsorted(w_cumsum,Rvals)

                num_keepids = len(keep_ids)
                self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_data_ar_list[ng][ids_ME[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_tosum_ar_list[ng][ids_ME[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,self.wt] = wbar*cp.ones(num_keepids)

                self.big_data_ar_list[ng][ids_ME,:] = cp.zeros((curr_ppc_ME,6))
                self.big_tosum_ar_list[ng][ids_ME,:] = cp.zeros((curr_ppc_ME,4))

                keep_ids = cp.arange(self.Nc_list[ng],self.Nc_list[ng]+num_keepids) 

                # adjustment
                Te_hat_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][keep_ids,self.wt] * self.big_tosum_ar_list[ng][keep_ids,self.en]) / original_wt
                vhat_cell_x = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vx] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_y = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vy] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_z = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vz] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                Tfac = cp.sqrt(T_cell / Te_hat_cell)

                self.big_data_ar_list[ng][keep_ids,self.vx] = v_cell_x + (self.big_data_ar_list[ng][keep_ids,self.vx] - vhat_cell_x ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vy] = v_cell_y + (self.big_data_ar_list[ng][keep_ids,self.vy] - vhat_cell_y ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vz] = v_cell_z + (self.big_data_ar_list[ng][keep_ids,self.vz] - vhat_cell_z ) * Tfac

                # put in energy
                self.big_tosum_ar_list[ng][keep_ids,self.en] = 0.5*self.m_e*self.j_ev * ( \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vx], self.big_data_ar_list[ng][keep_ids,self.vx]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vy], self.big_data_ar_list[ng][keep_ids,self.vy]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vz], self.big_data_ar_list[ng][keep_ids,self.vz]) )

                self.Nc_list[ng] = self.Nc_list[ng] + num_keepids

            ## Too few high energies
            if (0 < curr_ppc_HE < 0.5*self.target_ppc_HE_ar[cc]):
                if (curr_ppc_HE == 0):
                    continue
                ids_HE = self.cell_particles_HE_ar[0:curr_ppc_HE,cc]
                HW_ids_temp = cp.where(self.big_tosum_ar_list[ng][ids_HE,self.wt] > self.min_wt_allow)[0]
                HW_ids = ids_HE[HW_ids_temp]

                if (len(HW_ids) > 0):
                    P_wts = self.big_tosum_ar_list[ng][HW_ids,self.wt]/cp.sum(self.big_tosum_ar_list[ng][HW_ids,self.wt])
                    keepers = cp.random.choice(HW_ids,2*tppc_HE-curr_ppc_HE,True,p=P_wts)
                    unique_ids, unique_cts = cp.unique(keepers,return_counts=True)
                    n_unique = len(unique_ids)

                    for vv in range(n_unique): 
                        curr_copies = int(unique_cts[vv])
                        self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] = self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],self.wt] / (curr_copies+1)
                        self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,6))*self.big_data_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+curr_copies,:] = cp.ones((curr_copies,4))*self.big_tosum_ar_list[ng][HW_ids[unique_ids[vv]],:]
                        self.Nc_list[ng] = self.Nc_list[ng] + curr_copies

            ## Too many high energies
            if (curr_ppc_HE >= 3.0*self.target_ppc_HE_ar[cc]):
                old_Nc = self.Nc_list[ng]
                curr_ppc_HE = min(curr_ppc_HE, self.max_ppc_realistic_HE)
                temp_tppc_HE = int(0.6*self.target_ppc_HE_ar[cc])

                ids_HE = self.cell_particles_HE_ar[0:curr_ppc_HE,cc]
                original_wt = cp.sum(self.big_tosum_ar_list[ng][ids_HE,self.wt])

                # v's and Te before
                v_cell_x = cp.sum(self.big_data_ar_list[ng][ids_HE,self.vx]*self.big_tosum_ar_list[ng][ids_HE,self.wt])/original_wt
                v_cell_y = cp.sum(self.big_data_ar_list[ng][ids_HE,self.vy]*self.big_tosum_ar_list[ng][ids_HE,self.wt])/original_wt
                v_cell_z = cp.sum(self.big_data_ar_list[ng][ids_HE,self.vz]*self.big_tosum_ar_list[ng][ids_HE,self.wt])/original_wt
                T_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][ids_HE,self.wt]*self.big_tosum_ar_list[ng][ids_HE,self.en] / (original_wt))

                # new wt for particles
                wbar = original_wt / temp_tppc_HE

                # use strata sampling for partial sum array 
                # to select kept particles
                p_use = self.big_tosum_ar_list[ng][ids_HE,self.wt] / original_wt
                w_cumsum = cp.cumsum(p_use)
                Rvals = (1/temp_tppc_HE) * (cp.random.rand(temp_tppc_HE) + cp.arange(temp_tppc_HE))
                keep_ids = cp.searchsorted(w_cumsum,Rvals)

                num_keepids = len(keep_ids)
                self.big_data_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_data_ar_list[ng][ids_HE[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,:] = self.big_tosum_ar_list[ng][ids_HE[keep_ids],:]
                self.big_tosum_ar_list[ng][self.Nc_list[ng]:self.Nc_list[ng]+num_keepids,self.wt] = wbar*cp.ones(num_keepids)

                self.big_data_ar_list[ng][ids_HE,:] = cp.zeros((curr_ppc_HE,6))
                self.big_tosum_ar_list[ng][ids_HE,:] = cp.zeros((curr_ppc_HE,4))

                keep_ids = cp.arange(self.Nc_list[ng],self.Nc_list[ng]+num_keepids) 

                # adjustment
                Te_hat_cell = (2/3) * cp.sum(self.big_tosum_ar_list[ng][keep_ids,self.wt] * self.big_tosum_ar_list[ng][keep_ids,self.en]) / original_wt
                vhat_cell_x = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vx] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_y = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vy] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                vhat_cell_z = cp.sum(self.big_data_ar_list[ng][keep_ids,self.vz] * self.big_tosum_ar_list[ng][keep_ids,self.wt]) / original_wt
                Tfac = cp.sqrt(T_cell / Te_hat_cell)

                self.big_data_ar_list[ng][keep_ids,self.vx] = v_cell_x + (self.big_data_ar_list[ng][keep_ids,self.vx] - vhat_cell_x ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vy] = v_cell_y + (self.big_data_ar_list[ng][keep_ids,self.vy] - vhat_cell_y ) * Tfac
                self.big_data_ar_list[ng][keep_ids,self.vz] = v_cell_z + (self.big_data_ar_list[ng][keep_ids,self.vz] - vhat_cell_z ) * Tfac

                # put in energy
                self.big_tosum_ar_list[ng][keep_ids,self.en] = 0.5*self.m_e*self.j_ev * ( \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vx], self.big_data_ar_list[ng][keep_ids,self.vx]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vy], self.big_data_ar_list[ng][keep_ids,self.vy]) + \
                    cp.multiply(self.big_data_ar_list[ng][keep_ids,self.vz], self.big_data_ar_list[ng][keep_ids,self.vz]) )

                self.Nc_list[ng] = self.Nc_list[ng] + num_keepids


        # Removing 0's from merging
        alive_inds = (cp.where(self.big_tosum_ar_list[ng][0:self.Nc_list[ng],self.wt] != 0))[0]
        Na = len(alive_inds)
        self.big_data_ar_list[ng][0:Na,:] = self.big_data_ar_list[ng][alive_inds[0:Na],:]
        self.big_tosum_ar_list[ng][0:Na,:] = self.big_tosum_ar_list[ng][alive_inds[0:Na],:]
        self.big_forE_xbins_list[ng][0:Na] = (self.big_data_ar_list[ng][0:Na, self.xx]/self.dx).astype(int)
        self.big_curr_xbins_list[ng][0:Na] = ((self.big_data_ar_list[ng][0:Na, self.xx] + 0.5*self.dx)/self.dx).astype(int)
        self.big_data_ar_list[ng][Na:self.Nc_list[ng],:] = cp.zeros((self.Nc_list[ng]-Na,6))
        self.big_tosum_ar_list[ng][Na:self.Nc_list[ng],:] = cp.zeros((self.Nc_list[ng]-Na,4))
        self.big_curr_xbins_list[ng][Na:self.Nc_list[ng]] = cp.zeros(self.Nc_list[ng]-Na)
        self.big_forE_xbins_list[ng][Na:self.Nc_list[ng]] = cp.zeros(self.Nc_list[ng]-Na)
        self.Nc_list[ng] = Na


    ## To check if resampling is needed
    def _check_for_resampling(self):
        ng = 0

        # Do we need to resample?
        self.bin_write_inds_LE_ar.fill(0)
        self.cell_particles_LE_ar.fill(0)
        self.bin_write_inds_ME_ar.fill(0)
        self.cell_particles_ME_ar.fill(0)
        self.bin_write_inds_HE_ar.fill(0)
        self.cell_particles_HE_ar.fill(0)
        self.bin_particles(self.big_tosum_ar_list[ng], self.d_cell_particles_HE_ar, self.d_cell_particles_ME_ar, self.d_cell_particles_LE_ar, 
                           self.d_bin_write_inds_HE_ar, self.d_bin_write_inds_ME_ar, self.d_bin_write_inds_LE_ar, self.d_currxbins_list[ng], 
                           self.Nc_list[ng], self.N, self.max_ppc_realistic_HE, self.max_ppc_realistic_ME, self.max_ppc_realistic_LE, 
                           self.threads_per_block)

    
        ## Countings particles per energy bins
        hmin_ids = cp.where(self.bin_write_inds_HE_ar[:] > 0)[0]
        mmin_ids = cp.where(self.bin_write_inds_ME_ar[:] > 0)[0]

        if (len(hmin_ids) > 0): 
            HEmin = cp.min(self.bin_write_inds_HE_ar[cp.where(self.bin_write_inds_HE_ar[:] > 0)[0]+0])
        else:
            HEmin = 100000

        mmin_ids = cp.where(self.bin_write_inds_ME_ar[:] > 0)[0]
        if (len(mmin_ids) > 0): 
            MEmin = cp.min(self.bin_write_inds_ME_ar[cp.where(self.bin_write_inds_ME_ar[:] > 0)[0]+0])
        else:
            MEmin = 100000

        MEmax = cp.max(self.bin_write_inds_ME_ar)
        HEmax = cp.max(self.bin_write_inds_HE_ar)
        LEmax = cp.max(self.bin_write_inds_LE_ar)
        LEmin = cp.min(self.bin_write_inds_LE_ar[cp.where(self.bin_write_inds_LE_ar[:] > 0)[0]+0])

        ## Check what is needed
        if (LEmin < 0.5*self.ttLE):
            self.need_to_resample = True
        if (MEmin < 0.5*self.ttME):
            self.need_to_resample = True
        if (HEmin < 0.5*self.ttHE):
            self.need_to_resample = True
        if (LEmax >= 3.0*self.ttLE):
            self.need_to_resample = True
        if (MEmax >= 3.0*self.ttME):
            self.need_to_resample = True
        if (HEmax >= 3.0*self.ttHE):
            self.need_to_resample = True
        else:
            self.need_to_resample = False
