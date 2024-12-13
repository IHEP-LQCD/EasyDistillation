import functools
from typing import List, Tuple

from opt_einsum import contract

from ..constant import Nc, Nd
from ..backend import get_backend, check_QUDA
from ..preset import GaugeField, Eigenvector


class NoisevectorGenerator:
    def __init__(
        self,
        eigenvector: Eigenvector,
        dilution: Tuple,
        highmode: int = 0,
        full_noisev: bool = False,
    ):
        backend = get_backend()
        self.eigenvector = eigenvector
        self.highmode = highmode
        self.full_noisev = full_noisev
        self.eigv_list = dilution[0]
        self.transfer_matrix=None
        if isinstance(dilution[1], int):
            self.noisev_list = [dilution[1]] * len(dilution[0])
        else:
            self.noisev_list = dilution[1]
        assert len(self.noisev_list) == len(self.eigv_list)
        assert (backend.array(self.noisev_list) <= backend.array(self.eigv_list)).all()

    def load(self, key: str):
        self._eigenvector_data = self.eigenvector.load(key)

    def calc(self, seed=None):
        backend = get_backend()
        backend.random.seed(seed)
        eigenvector = self._eigenvector_data
        Lt, Ne, Lz, Ly, Lx, Nc = eigenvector.shape
        noisev_shape = list(eigenvector.shape)
        if self.full_noisev:
            noisev_shape[1] = sum(self.eigv_list) + self.highmode
        else:
            noisev_shape[1] = sum(self.noisev_list) + self.highmode
        noisev_shape = tuple(noisev_shape)
        noisev = backend.zeros(noisev_shape, dtype="<c16")

        noisev_start = 0
        eigv_start = 0
        for nsrc in range(len(self.noisev_list)):
            noisev_lenth = self.noisev_list[nsrc]
            eigv_lenth = self.eigv_list[nsrc]
            if noisev_lenth == eigv_lenth:
                noisev[:, noisev_start : noisev_start + noisev_lenth] = eigenvector[
                    :, eigv_start : eigv_start + eigv_lenth
                ]
            else:
                if noisev_lenth == 1:
                    random_array = backend.random.uniform(0, 2 * backend.pi, (Lt, eigv_lenth))
                    transfer_matrix = backend.exp(1j * random_array)
                    noisev[:, noisev_start] = contract(
                        "tizyxa,ti->tzyxa",
                        eigenvector[:, eigv_start : eigv_start + eigv_lenth],
                        transfer_matrix,
                    )
                else:
                    if self.full_noisev:
                        noisev_lenth = eigv_lenth
                    transfer_matrix = backend.zeros((Lt, eigv_lenth, noisev_lenth), dtype="<c16")
                    for t in range(Lt):
                        random_array = backend.random.randn(eigv_lenth, eigv_lenth) + 1j * backend.random.randn(
                            eigv_lenth, eigv_lenth
                        )
                        Q, R = backend.linalg.qr(random_array)
                        transfer_matrix[t] = Q[:, :noisev_lenth]
                    for t in range(Lt):
                        noisev[t, noisev_start : noisev_start + noisev_lenth] = contract(
                            "izyxa,ij->jzyxa",
                            eigenvector[t, eigv_start : eigv_start + eigv_lenth],
                            transfer_matrix[t],
                        )
            noisev_start += noisev_lenth
            eigv_start += eigv_lenth

        if self.highmode:
            noisev_eigensys = noisev_start
            while noisev_start < sum(self.noisev_list) + self.highmode:
                random_vector = backend.random.randn(Lt, Lz, Ly, Lx, Nc) + 1j * backend.random.randn(Lt, Lz, Ly, Lx, Nc)
                for t in range(Lt):
                    random_vector[t] -= contract(
                        "zyxa,izyxa,iuvwb->uvwb",
                        random_vector[t],
                        eigenvector[t, :eigv_start].conj(),
                        eigenvector[t, :eigv_start],
                    )
                    random_vector[t] -= contract(
                        "zyxa,izyxa,iuvwb->uvwb",
                        random_vector[t],
                        noisev[t, noisev_eigensys:noisev_start].conj(),
                        noisev[t, noisev_eigensys:noisev_start],
                    )
                if backend.linalg.norm(random_vector) > 1e-10:
                    noisev[:, noisev_start] = random_vector / backend.linalg.norm(random_vector)
                    # eigenvector = backend.concatenate(
                    #     (eigenvector, noisev[:, noisev_start : noisev_start + 1]),
                    #     axis=1,
                    # )
                    noisev_start += 1
                    # eigv_start += 1
        return noisev
