from typing import Iterable, List

from opt_einsum import contract

from ..insertion import Operator, OperatorDisplacement, InsertionRow
from ..insertion.gamma import gamma
from ..data import get_elemental_data
from ..filedata.abstract import FileData
from ..backend import get_backend


def twopoint(
    operators: List[Operator],
    elemental: FileData,
    perambulator: FileData,
    timeslices: Iterable[int],
    Lt: int,
    usedNe: int = None,
    perambulator_bw = None,
    is_sum_over_source_t = True,
):
    '''
    Calculate the two-point correlation function C(t) for a given set of operators.

    Parameters:
    - operators: List of Operator objects defining the interpolating fields.
    - elemental: FileData object containing the elemental vectors.
    - perambulator: FileData object containing the perambulator (quark propagator) data.
    - timeslices: Iterable of time slices at which the correlation function is to be evaluated.
    - Lt: Temporal extent of the lattice.
    - usedNe: Number of eigenvectors used in the calculation (None uses all available).
    - perambulator_bw: Optional FileData object for a different backward quark propagator.
    - is_sum_over_source_t: return all t source correlator at axis = 0, default is True.

    If 'perambulator_bw' is not None, it is used for the backward quark propagator,
    allowing for calculations involving different quark flavors or configurations.

    Returns:
    - A NumPy array of shape (Nop, Lt) containing the two-point correlation function
      values for each operator and timeslice.

    Note:
    - The backend for tensor operations is determined by the 'get_backend' function.
    - The function prints the data throughput rate for the perambulator at each timeslice.
    - The returned correlation function values are multiplied by -1 as per convention.
    '''
    backend = get_backend()
    Nop = len(operators)
    Nt = len(timeslices)

    ret = backend.zeros((Nop, Nt, Lt), "<c16") 
    phis = get_elemental_data(operators, elemental, usedNe)
    for it, t_src in enumerate(timeslices):
        tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]
        # tau = backend.roll(perambulator[t, :, :, :, :usedNe, :usedNe], -t, 0)
        if perambulator_bw is None:
            tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        else:
            tmp = perambulator_bw[t_src, :, :, :, :usedNe, :usedNe]
            tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tmp.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi[0].conj(), gamma(8))
            ret[idx, it] = contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t",
                tau_bw,
                phi[0],
                backend.roll(phi[1], -t_src, 1),
                tau,
                gamma_src,
                phi[1][:, t_src].conj(),
            )
        # print(f"t{t}: {perambulator.size_in_byte/perambulator.time_in_sec/1024**2:.5f} MB/s")
    if is_sum_over_source_t:
        return -ret.mean(1)
    else:
        return -ret


def twopoint_matrix(
    operators: List[Operator],
    elemental: FileData,
    perambulator: FileData,
    timeslices: Iterable[int],
    Lt: int,
    usedNe: int = None,
    is_sum_over_source_t = True,
):
    backend = get_backend()
    Nop = len(operators)
    Nt = len(timeslices)

    ret = backend.zeros((Nop, Nop, Nt, Lt), "<c16")
    phis = get_elemental_data(operators, elemental, usedNe)
    for it, t_src in enumerate(timeslices):
        tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]
        # tau = backend.roll(perambulator[t, :, :, :, :usedNe, :usedNe], -t, 0)
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for isrc in range(Nop):
            for isnk in range(Nop):
                phi_src = phis[isrc]
                gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi_src[0].conj(), gamma(8))
                phi_snk = phis[isnk]
                ret[isrc, isnk, it] = contract(
                    "tijab,xjk,xtbc,tklcd,yli,yad->t",
                    tau_bw,
                    phi_snk[0],
                    backend.roll(phi_snk[1], -t_src, 1),
                    tau,
                    gamma_src,
                    phi_src[1][:, t_src].conj(),
                )
        print(f"t{t_src}: {perambulator.size_in_byte/perambulator.time_in_sec/1024**2:.5f} MB/s")
    
    if is_sum_over_source_t:
        return -ret.mean(2)
    else:
        return -ret


def twopoint_isoscalar(
    operators: List[Operator],
    elemental: FileData,
    perambulator: FileData,
    timeslices: Iterable[int],
    Lt: int,
    usedNe: int = None,
    Nf: int = 2,
    is_sum_over_source_t = True,
):
    backend = get_backend()
    Nop = len(operators)
    Nt = len(timeslices)
    if Lt != Nt:
        raise ValueError("Disconnect must compute full timeslices!")

    connected = backend.zeros((Nop, Nt, Lt), "<c16")
    loop_src = backend.zeros((Nop, Lt), "<c16")
    loop_snk = backend.zeros((Nop, Lt), "<c16")
    phis = get_elemental_data(operators, elemental, usedNe)

    # for t_src in timeslices:
    for it, t_src in enumerate(timeslices):
        tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]
        # tau = backend.roll(perambulator[t, :, :, :, :usedNe, :usedNe], -t, 0)
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for idx in range(Nop):
            phi = phis[idx]
            gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi[0].conj(), gamma(8))
            connected[idx, it] = contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t",
                tau_bw,
                phi[0],
                backend.roll(phi[1], -t_src, 1),
                tau,
                gamma_src,
                phi[1][:, t_src].conj(),
            )
            loop_src[idx, t_src] = contract("ijab,yji,yab", tau[0], gamma_src, phi[1][:, t_src].conj())
            loop_snk[idx, t_src] = contract("ijab,xji,xba", tau[0], phi[0], phi[1][:, t_src])
        print(f"t{t_src}: {perambulator.size_in_byte/perambulator.time_in_sec/1024**2:.5f} MB/s")

    disconnected = contract("xi, xj -> xij", loop_src, loop_snk)
    for t_src in timeslices:
        disconnected[:, t_src, :] = backend.roll(disconnected[:, t_src, :], -t_src, axis=1)

    if is_sum_over_source_t:
        return (-connected + Nf * disconnected).mean(1)
    else:
        return (-connected + Nf * disconnected)


def twopoint_isoscalar_matrix(
    operators: List[Operator],
    elemental: FileData,
    perambulator: FileData,
    timeslices: Iterable[int],
    Lt: int,
    usedNe: int = None,
    Nf: int = 2,
    is_sum_over_source_t = True,
):
    backend = get_backend()
    Nop = len(operators)
    Nt = len(timeslices)
    if Lt != Nt:
        raise ValueError("Disconnect must compute full timeslices!")

    connected = backend.zeros((Nop, Nop, Nt, Lt), "<c16")
    loop_src = backend.zeros((Nop, Lt), "<c16")
    loop_snk = backend.zeros((Nop, Lt), "<c16")
    phis = get_elemental_data(operators, elemental, usedNe)
    for it, t_src in enumerate(timeslices):
        tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]
        # tau = backend.roll(perambulator[t, :, :, :, :usedNe, :usedNe], -t, 0)
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for isrc in range(Nop):
            phi_src = phis[isrc]
            gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi_src[0].conj(), gamma(8))
            loop_src[isrc, t_src] = contract("ijab,yji,yab", tau[0], gamma_src, phi_src[1][:, t_src].conj())
            loop_snk[isrc, t_src] = contract("ijab,xji,xba", tau[0], phi_src[0], phi_src[1][:, t_src])
            for isnk in range(Nop):
                phi_snk = phis[isnk]
                connected[isrc, isnk, it] = contract(
                    "tijab,xjk,xtbc,tklcd,yli,yad->t",
                    tau_bw,
                    phi_snk[0],
                    backend.roll(phi_snk[1], -t_src, 1),
                    tau,
                    gamma_src,
                    phi_src[1][:, t_src].conj(),
                )
        print(f"t{t_src}: {perambulator.size_in_byte/perambulator.time_in_sec/1024**2:.5f} MB/s")

    disconnected = contract("xi, yj -> xyij", loop_src, loop_snk)
    for t_src in timeslices:
        disconnected[:, :, t_src, :] = backend.roll(disconnected[:, :, t_src, :], -t_src, axis=2)
    if is_sum_over_source_t:
        return (-connected + Nf * disconnected).mean(2)
    else:
        return (-connected + Nf * disconnected)


def twopoint_matrix_multi_mom(
    insertions: List[InsertionRow],
    mom_list: List,
    elemental: FileData,
    perambulator: FileData,
    timeslices: Iterable[int],
    Lt: int,
    usedNe: int = None,
    insertions_coeff_list: List = None,
    is_sum_over_source_t = True,
    distance_list:List = None,
):
    backend = get_backend()
    Nmom = len(mom_list)
    Nt = len(timeslices)
    Nop = len(insertions)
    op_src_list = []
    op_snk_list = []
    if insertions_coeff_list is None:
        insertions_coeff_list = [1] * len(insertions)
    assert len(insertions) == len(insertions_coeff_list)
    if distance_list is not None:
        assert len(distance_list) == len(insertions_coeff_list)
    for imom in range(Nmom):
        px, py, pz = mom_list[imom]
        for isrc in range(Nop):
            for isnk in range(Nop):
                if distance_list is not None:
                    op_src_list.append(OperatorDisplacement("", [insertions[isrc](px, py, pz)], [insertions_coeff_list[isrc]], distances=[distance_list[isrc]]))
                    op_snk_list.append(OperatorDisplacement("", [insertions[isnk](px, py, pz)], [insertions_coeff_list[isnk]], distances=[distance_list[isnk]]))
                else:
                    op_src_list.append(Operator("", [insertions[isrc](px, py, pz)], [insertions_coeff_list[isrc]]))
                    op_snk_list.append(Operator("", [insertions[isnk](px, py, pz)], [insertions_coeff_list[isnk]]))
    Nterm = Nmom * Nop * Nop

    ret = backend.zeros((Nterm, Nt, Lt), "<c16")
    phis_src = get_elemental_data(op_src_list, elemental, usedNe)
    phis_snk = get_elemental_data(op_snk_list, elemental, usedNe)
    for it, t_src in enumerate(timeslices):
        tau = perambulator[t_src, :, :, :, :usedNe, :usedNe]
        # tau = backend.roll(perambulator[t, :, :, :, :usedNe, :usedNe], -t, 0)
        tau_bw = contract("ii,tjiba,jj->tijab", gamma(15), tau.conj(), gamma(15))
        for item in range(Nterm):
            phi_src = phis_src[item]
            gamma_src = contract("ij,xkj,kl->xil", gamma(8), phi_src[0].conj(), gamma(8))
            phi_snk = phis_snk[item]
            ret[item, it] = contract(
                "tijab,xjk,xtbc,tklcd,yli,yad->t",
                tau_bw,
                phi_snk[0],
                backend.roll(phi_snk[1], -t_src, 1),
                tau,
                gamma_src,
                phi_src[1][:, t_src].conj(),
            )
        print(f"t{t_src}: {perambulator.size_in_byte/perambulator.time_in_sec/1024**2:.5f} MB/s")
    # ret /= Nt
    ret = ret.reshape((Nmom, Nop, Nop, Nt, Lt))
    if is_sum_over_source_t:
        return -ret.mean(3)
    else:
        return -ret
