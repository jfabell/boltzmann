import cupy as cp
import numpy as np

from . import reduction

def back_to_cpu(data_out, data_out_np, data_out2, data_out_np2, 
                ne_ar, ni_src, ns_src, nrg_ar, 
                counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar, 
                Nc, Na, RN, nni):
   
    # Handling first reduction 
    data_out_np[0:RN,:] = cp.asnumpy(data_out[0:RN,:])
    ne_ar[0:RN] = data_out_np[0:RN,0]
    ni_src = data_out_np[0:RN,1]
    ns_src = data_out_np[0:RN,2]
    nrg_ar[0:RN] = data_out_np[0:RN,3]

    # Handling collision counting reduction
    data_out_np2[0:RN,:] = cp.asnumpy(data_out2[0:RN,:])
    counter_g0_ar[0:RN] = data_out_np2[0:RN,0]
    counter_g1_ar[0:RN] = data_out_np2[0:RN,1]
    counter_g2_ar[0:RN] = data_out_np2[0:RN,2]
    counter_g3_ar[0:RN] = data_out_np2[0:RN,3]
    return(ne_ar, ni_src, ns_src, nrg_ar, counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar)

def reshuffle_data_atomics(need_to_reshuffle, big_data_ar, big_tosum_ar, Nc, nni):

    if (need_to_reshuffle):
        alive_inds = (cp.where(big_tosum_ar[0:Nc+nni,0] > 0))[0]
        Na = len(alive_inds)
        big_data_ar[0:Na,:] = big_data_ar[alive_inds[0:Na],:]
        big_tosum_ar[0:Na,:] = big_tosum_ar[alive_inds[0:Na],:]
        big_data_ar[Na:Nc+nni] = cp.zeros((Nc+nni-Na,6))
        big_tosum_ar[Na:Nc+nni] = cp.zeros((Nc+nni-Na,4))
        Nnew = Na
    else:
        Nnew = Nc + nni
    return(Nnew)

def post_processing(data_out, data_out_np, data_out2, data_out_np2, data_out3, data_out_np3, big_data_ar, big_tosum_ar, big_collct_ar, need_to_reshuffle,
                    Na, Nc, nni, ne_ar, ni_src, ns_src, nrg_ar, counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar, big_N, RN, L, dx, temp_x):

    
    temp_x[0:Nc+nni] = (big_data_ar[0:Nc+nni,0] + 0.5*dx) / (L+dx)
    data_out.fill(0) 
    data_out2.fill(0)
    data_out3.fill(0)

    # multiply g2 src, energy by n_e so that we average correctly later
    nz_ids = cp.where(big_tosum_ar[0:Nc+nni,0] > 0.)[0]
    big_tosum_ar[nz_ids,1] = cp.multiply(big_tosum_ar[nz_ids,1], big_tosum_ar[nz_ids,0])
    big_tosum_ar[nz_ids,2] = cp.multiply(big_tosum_ar[nz_ids,2], big_tosum_ar[nz_ids,0])
    big_tosum_ar[nz_ids,3] = cp.multiply(big_tosum_ar[nz_ids,3], big_tosum_ar[nz_ids,0])

    reduction.reduction(4, 0, big_N, RN, data_out, big_tosum_ar[0:Nc+nni,:], temp_x[0:Nc+nni], Nc+nni)
    reduction.reduction(4, 0, big_N, RN, data_out2, big_collct_ar[0:Nc+nni,:], temp_x[0:Nc+nni], Nc+nni)
   
    ne_ar, ni_src, ns_src, nrg_ar, counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar = back_to_cpu(data_out, data_out_np, data_out2, data_out_np2,
                                                      ne_ar, ni_src, ns_src, nrg_ar, counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar,
                                                      Nc, Na, RN, nni)

    big_tosum_ar[nz_ids,3] = cp.divide(big_tosum_ar[nz_ids,3], big_tosum_ar[nz_ids,0])
    
    # Reshuffle
    Nnew = reshuffle_data_atomics(need_to_reshuffle, big_data_ar, big_tosum_ar, Nc, nni)
    #Nnew = Nc+nni    

    return(ne_ar, ni_src, ns_src, nrg_ar, counter_g0_ar, counter_g1_ar, counter_g2_ar, counter_g3_ar, Nnew)


