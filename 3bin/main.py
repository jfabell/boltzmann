import argparse

import numba

try:
    from boltzmann import BoltzmannDSMC
except:
    from .boltzmann import BoltzmannDSMC


def main(restart_cyclenum, num_steps, name, num_particles, gridpts, bnf, num_nodes, pyk, resample_freq, in_dt, in_spc):
    boltzmann = BoltzmannDSMC(num_steps, name, restart_cyclenum, num_particles, gridpts, bnf, num_nodes, pyk, resample_freq, in_dt, in_spc)
    boltzmann.run(num_steps, name, restart_cyclenum)

def run(restart_cyclenum, num_steps, name, num_particles, gridpts, bnf, num_nodes, pyk, resample_freq, in_dt, in_spc):

    main(restart_cyclenum, num_steps, name, num_particles, gridpts, bnf, num_nodes, pyk, resample_freq, in_dt, in_spc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=400)
    parser.add_argument("-N", "--num_particles", type=int, default=200000)
    parser.add_argument("-g", "--gridpts", type=int, default=256)
    parser.add_argument("-bnf", type=int, default=4)
    parser.add_argument("-n", "--num_nodes", type=int, default=1)
    parser.add_argument("-p", "--pykokkos", action="store_true")
    parser.add_argument("-res", "--restart_cyclenum", type=int, default=0)
    parser.add_argument("-name", "--data_name", type=str, default="")
    parser.add_argument("-rsf", "--resample_freq", type=int, default=1)
    parser.add_argument("-dt", "--in_dt", type=float, default=50000.0)
    parser.add_argument("-spc", "--in_spc", type=int, default=5000)
    args = parser.parse_args()
    run(args.restart_cyclenum, args.steps, args.data_name, 
                  args.num_particles, args.gridpts, args.bnf, args.num_nodes, args.pykokkos, 
                  args.resample_freq, args.in_dt, args.in_spc)
