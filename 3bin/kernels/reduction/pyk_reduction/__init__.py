import math

import cupy as cp
import pykokkos as pk

@pk.workload(
    w = pk.ViewTypeInfo(space=pk.CudaSpace),
    v = pk.ViewTypeInfo(space=pk.CudaSpace),
    x = pk.ViewTypeInfo(space=pk.CudaSpace)
)
class Reduction:
    def __init__(self, r, M, RM, w, v, x, p, ns):
        self.w: pk.View1D[float] = pk.from_cupy(cp.ravel(w, order="A"))
        self.v: pk.View1D[float] = pk.from_cupy(cp.ravel(v, order="A"))
        self.x: pk.View1D[float] = pk.from_cupy(x)

        self.r: int = r
        self.RM: int = RM
        self.M: int = M
        self.p: int = p
        self.scratch_size: int = int(ns / 8)

    @pk.main
    def run(self):
        blockx: int = 64
        blocky: int = self.r
        x_threads: int = blockx * (self.p // blockx + 1)
        y_threads: int = blocky * (self.r // blocky + 1)

        bloks: int = math.ceil(self.M / self.RM)
        ns: int = bloks * self.r * 8

        league_size: int = self.M // bloks + 1
        team_size: int = bloks

        pk.parallel_for(pk.MDRangePolicy([0, 0], [x_threads, y_threads]), self.reda)
        pk.parallel_for(
            pk.TeamPolicy(league_size, team_size).set_scratch_size(0, pk.PerTeam(ns)),
            self.redcheck)

    @pk.workunit
    def reda(self, i: int, j: int):
        if i < self.p and j < self.r:
            b: int = self.x[i] * self.M
            pk.atomic_add(self.w, [self.r * b + j], self.v[self.r * i + j])

    @pk.workunit
    def redcheck(self, team_member: pk.TeamMember):
        i: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
        tid: int = team_member.team_rank()

        if i < self.M:
            bb: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), self.scratch_size)
            for j in range(self.r):
                bb[self.r * tid + j] = self.w[self.r * i + j]

            team_member.team_barrier()

            s: pk.uint32 = (team_member.team_size() / 2)
            while s > 0:
                if tid < s:
                    for j in range(self.r):
                        bb[self.r * tid + j] += bb[self.r * (tid + s) + j]
                
                team_member.team_barrier()
                s >>= 1

            if tid == 0:
                for j in range(self.r):
                    self.w[self.r * team_member.league_rank() + j] = bb[j]

def reduction(r: int, p: int, M: int, RM: int, dw, dv, dx, ptcl: int):
    bloks: int = -(M // -RM)
    ns: int = bloks * r * 8

    r = Reduction(r, M, RM, dw, dv, dx, ptcl, ns)
    pk.execute(pk.Cuda, r)


def pyk_reduction_fast(r: int, p: int, M: int, RM: int, dw, dv, dx, ptcl: int):
    pyk_reduction_fast.workload.w = pk.from_cupy(cp.ravel(dw, order="A"))
    pyk_reduction_fast.workload.v = pk.from_cupy(cp.ravel(dv, order="A"))
    pyk_reduction_fast.workload.x = pk.from_cupy(dx)

    pyk_reduction_fast.workload.r = r
    pyk_reduction_fast.workload.RM = RM
    pyk_reduction_fast.workload.M = M
    pyk_reduction_fast.workload.p = ptcl

    bloks: int = -(M // -RM)
    ns: int = bloks * r * 8

    pyk_reduction_fast.workload.scratch_size = int(ns / 8)
    pk.execute(pk.Cuda, pyk_reduction_fast.workload)

pyk_reduction_fast.workload = Reduction(0, 0, 0, cp.zeros((1,1)), cp.zeros((1,1)), cp.zeros(1), 0, 0)
