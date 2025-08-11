import numpy as np

def data_write(rank, bigstep, name, output_freq, restart_cyclenum, Nc, 
               big_data_ar, big_tosum_ar, ne_ar, ni_ar, ns_ar, Te_ar, E_ar, V_ar,
               g0_ar, g2_ar, 
               ne_CA, ni_CA, ns_CA, Te_CA, E_CA, V_CA,
               ):

    write_str = str(int((bigstep+1)/output_freq) + restart_cyclenum)
    cyclestr = '_cycle' + str(write_str) + '.dat'
    cyclestr_rk = '_cycle' + str(write_str) + '_mpi' + str(rank) + '.dat'

    ## Only MPI rank 0 outputs the grid quantities
    if (rank == 0):
        fileobj_ne = open(name + 'ne' + cyclestr, mode='wb')
        fileobj_ni = open(name + 'ni' + cyclestr, mode='wb')
        fileobj_Te = open(name + 'Te' + cyclestr, mode='wb')
        fileobj_E = open(name + 'E' + cyclestr, mode='wb')
        #fileobj_V = open(name + 'V' + cyclestr, mode='wb')
        fileobj_g2 = open(name + 'ionprod_total' + cyclestr, mode='wb')
        
        ne_ar.tofile(fileobj_ne)
        ni_ar.tofile(fileobj_ni)
        Te_ar.tofile(fileobj_Te)
        E_ar.tofile(fileobj_E)
        #V_ar.tofile(fileobj_V)
        g2_ar.tofile(fileobj_g2) 
        
        fileobj_ne.close()
        fileobj_ni.close()
        fileobj_Te.close()
        fileobj_E.close()
        #fileobj_V.close()
        fileobj_g2.close()
        
        ## CYCLE AVERAGES
        write_str = str(int((bigstep+1)/output_freq) + restart_cyclenum)
        cyclestr = '_cycle' + str(write_str) + '.dat'
        
        fileobj_ne_CA = open(name + 'ne_CA' + cyclestr, mode='wb')
        fileobj_ni_CA = open(name + 'ni_CA' + cyclestr, mode='wb')
        fileobj_Te_CA = open(name + 'Te_CA' + cyclestr, mode='wb')
        fileobj_E_CA = open(name + 'E_CA' + cyclestr, mode='wb')
        #fileobj_V_CA = open(name + 'V_CA' + cyclestr, mode='wb')
        
        ne_CA.tofile(fileobj_ne_CA)
        ni_CA.tofile(fileobj_ni_CA)
        Te_CA.tofile(fileobj_Te_CA)
        E_CA.tofile(fileobj_E_CA)
        #V_CA.tofile(fileobj_V_CA)
        
        fileobj_ne_CA.close()
        fileobj_ni_CA.close()
        fileobj_Te_CA.close()
        fileobj_E_CA.close()
        #fileobj_V_CA.close()

    ## All MPI ranks output their particle data
    fileobj_x0 = open(name + 'e_x0' + cyclestr_rk, mode='wb')
    fileobj_v0 = open(name + 'e_v0' + cyclestr_rk, mode='wb')
    fileobj_v1 = open(name + 'e_v1' + cyclestr_rk, mode='wb')
    fileobj_v2 = open(name + 'e_v2' + cyclestr_rk, mode='wb')
    fileobj_wt = open(name + 'e_wt' + cyclestr_rk, mode='wb')
    
    big_data_ar[0:Nc,0].tofile(fileobj_x0)
    big_data_ar[0:Nc,3].tofile(fileobj_v0)
    big_data_ar[0:Nc,4].tofile(fileobj_v1)
    big_data_ar[0:Nc,5].tofile(fileobj_v2)
    big_tosum_ar[0:Nc,0].tofile(fileobj_wt)
    
    fileobj_x0.close()
    fileobj_v0.close()
    fileobj_v1.close()
    fileobj_v2.close()
    fileobj_wt.close()
