import cupy as cp
import numpy as np
import math
import numba
from numba import cuda

@cuda.jit
def define_particle_bins(d_tosum_ar, d_cell_particles_LE_ar,d_write_inds_LE_ar, 
                         d_curr_xbins, Nc, N, max_ppc_LE):

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    j = tx + bx*bw

    if (j < Nc and d_tosum_ar[j,0] > 0):
        cc = d_curr_xbins[j]

        if (cc >= 0 and cc <= N):
            write_ind = cuda.atomic.add(d_write_inds_LE_ar, cc, 1)
            if (write_ind <  max_ppc_LE):
                d_cell_particles_LE_ar[write_ind,cc] = j

def bin_particles(d_tosum_ar, d_cell_particles_LE_ar, d_write_inds_LE_ar,
                  d_curr_xbins, Nc, N, max_ppc_LE, threads_per_block):
    
    blocks = (Nc + (threads_per_block-1)) // threads_per_block
    
    define_particle_bins[blocks, threads_per_block](d_tosum_ar,  d_cell_particles_LE_ar,
                                                    d_write_inds_LE_ar, d_curr_xbins, 
                                                    Nc, N, max_ppc_LE)

