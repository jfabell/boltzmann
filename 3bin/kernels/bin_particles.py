import cupy as cp
import numpy as np
import math
import numba
from numba import cuda

@cuda.jit
def define_particle_bins(d_tosum_ar,
                         d_cell_particles_HE_ar, d_cell_particles_ME_ar, d_cell_particles_LE_ar, 
                         d_write_inds_HE_ar, d_write_inds_ME_ar, d_write_inds_LE_ar, 
                         d_curr_xbins, Nc, N, max_ppc_HE, max_ppc_ME, max_ppc_LE):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    j = tx + bx*bw

    if (j < Nc and d_tosum_ar[j,0] > 0):
        cc = d_curr_xbins[j]

        # 0 < E < 10 is low energy
        if (cc >= 0 and cc <= N and d_tosum_ar[j,3] < 10.0):
            write_ind = cuda.atomic.add(d_write_inds_LE_ar, cc, 1)
            if (write_ind <  max_ppc_LE):
                d_cell_particles_LE_ar[write_ind,cc] = j

        # 10 < E < 15.76 is medium energy
        elif (cc >= 0 and cc <= N and d_tosum_ar[j,3] < 15.76):
            write_ind2 = cuda.atomic.add(d_write_inds_ME_ar, cc, 1)
            if (write_ind2 < max_ppc_ME):
                d_cell_particles_ME_ar[write_ind2,cc] = j

        # 15.76 < E is high energy        
        elif (cc >= 0 and cc <= N and d_tosum_ar[j,3] >= 15.76):
            write_ind3 = cuda.atomic.add(d_write_inds_HE_ar, cc, 1)
            if (write_ind3 < max_ppc_HE):
                d_cell_particles_HE_ar[write_ind3,cc] = j

def bin_particles(d_tosum_ar, d_cell_particles_HE_ar, d_cell_particles_ME_ar, d_cell_particles_LE_ar, d_write_inds_HE_ar, d_write_inds_ME_ar, d_write_inds_LE_ar,
                  d_curr_xbins, Nc, N, max_ppc_HE, max_ppc_ME, max_ppc_LE, threads_per_block):
    
    blocks = (Nc + (threads_per_block-1)) // threads_per_block
    
    define_particle_bins[blocks, threads_per_block](d_tosum_ar, d_cell_particles_HE_ar, d_cell_particles_ME_ar, d_cell_particles_LE_ar,
                                                    d_write_inds_HE_ar, d_write_inds_ME_ar, d_write_inds_LE_ar, d_curr_xbins, 
                                                    Nc, N, max_ppc_HE, max_ppc_ME, max_ppc_LE)

