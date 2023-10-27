import numpy as np
import cupy as cp

from opt_einsum import contract
import time

print("Job started!")

# ------------------------------------------------------------------------------

MOM_list = []
Nt = 32
Nx = 64
conf_id = 58150
Px = 0
Py = 0
Pz = 0
Nev = 100
Nev1 = 100
eig_dir = f"/public/home/sunp/distillation/work_2023_09_17/beta6.20_mu-0.2770_ms-0.2400_L32x64_eigenvector_mom0/{conf_id}.eigenvector.npy"
output_dir = f"/public/home/gengyq/Proton_v2/data/{conf_id}/C32P29.{conf_id}.VVV.npy"


def readin_eigvecs(eig_dir, t, Nev, Nev1, Nx):
    f = np.load("%s" % (eig_dir))
    eigvecs = f[t, :, :, :, :, :]
    eigvecs = eigvecs.reshape(Nev, Nx * Nx * Nx, 3)
    eigvecs = eigvecs[0:Nev1, :, :]
    eigvecs_cupy = cp.asarray(eigvecs)
    return eigvecs_cupy


eigvecs_cupy = readin_eigvecs(eig_dir, 0, Nev, Nev1, conf_id, Nx)

print(eigvecs_cupy.shape)


# ------------------------------------------------------------------------------
def phase_calc(Mom):
    phase_factor = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                phase_factor[z * Nx * Nx + y * Nx + x] = np.exp(
                    -np.dot(Mom, Pos) * 2 * np.pi * 1j / Nx
                )
    return cp.asarray(phase_factor)


# ------------------------------------------------------------------------------


st = time.time()
VVV = cp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)

for t in range(0, Nt):
    st1 = time.time()
    eigvecs_cupy = readin_eigvecs(eig_dir, t, Nev, Nev1, conf_id, Nx)
    ed1 = time.time()
    print("Read-in eigenvector done , time used: %.3f s" % (ed1 - st1))
    # eigen_x_sum = np.transpose(eigvecs_cupy)
    # eigen_x_sum=cp.sum(eigvecs_cupy,axis=1)
    # eigen_1=eigen_sum[1]
    print(cp.shape(eigvecs_cupy))

    Mom = np.array([Pz, Py, Px])
    print(Mom)

    st2 = time.time()
    phase_factor_cupy = phase_calc(Mom)
    for xi in range(
        0, Nx
    ):  # I did this becasue the intermediate array is too large for a single GPU to handle
        VVV[t] += (
            contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
            )
            + contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
            )
            + contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
            )
            - contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
            )
            - contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
            )
            - contract(
                "x,ax,bx,cx->abc",
                phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
            )
        )
    ed2 = time.time()
    print("Contraction done , time used: %.3f s" % (ed2 - st2))

np.save(output_dir, VVV)
ed = time.time()
print("****************all complete , time used: %.3f s****************" % (ed - st))
