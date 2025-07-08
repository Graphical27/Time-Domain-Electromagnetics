import pandas as pd
import numpy as np
from simpeg.electromagnetics import time_domain
from simpeg import (
    optimization,
    discretize,
    maps,
    data_misfit,
    regularization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
import matplotlib.pyplot as plt


def make_wide_and_params(mesh, sigma, times, dobs):
    # depths & resistivities of the active (subsurface) cells
    depths = mesh.cell_centers_z[mesh.cell_centers_z < 0.0]
    resistivities = 1.0 / sigma[mesh.cell_centers_z < 0.0]

    n_times = len(times)
    n_layers = len(depths)

    # reshape dobs into (n_times × n_layers)
    dBdt = dobs.reshape(n_times, n_layers)

    # build long DataFrame
    rows = []
    for i, t in enumerate(times):
        for layer in range(n_layers):
            rows.append([t, layer + 1, dBdt[i, layer]])
    df_long = pd.DataFrame(rows, columns=["Time", "Layer", "dBdt"])

    # pivot to wide form: one row per Time, columns Layer_1 … Layer_n
    df_wide = df_long.pivot(index="Time", columns="Layer", values="dBdt")
    df_wide.columns = [f"Layer_{int(c)}" for c in df_wide.columns]
    df_wide.reset_index(inplace=True)
    df_wide.to_csv("synthetic_EM_wide.csv", index=False)

    # build parameter DataFrame
    df_res = pd.DataFrame(
        [resistivities],
        columns=[f"Resistivity_{i+1}" for i in range(n_layers)]
    )
    # thickness of each active cell in vertical direction
    active_inds = np.where(mesh.cell_centers_z < 0.0)[0]
    thickness = mesh.h[2][active_inds]
    df_thk = pd.DataFrame(
        [thickness[:-1]],
        columns=[f"Thickness_{i+1}" for i in range(n_layers - 1)]
    )

    df_params = pd.concat([df_res, df_thk], axis=1)
    df_params.to_csv("layer_params.csv", index=False)

    return df_wide, df_params


def run(plotIt=True):
    # --- build mesh ---
    cs, ncx, ncz, npad = 5.0, 25, 15, 15
    hx = [(cs, ncx), (cs, npad, 1.3)]
    hz = [(cs, npad, -1.3), (cs, ncz), (cs, npad, 1.3)]
    mesh = discretize.CylindricalMesh([hx, 1, hz], "00C")

    # Active cells & vertical-1D mapping
    active = mesh.cell_centers_z < 0.0
    layer = (mesh.cell_centers_z < 0.0) & (mesh.cell_centers_z >= -100.0)
    actMap = maps.InjectActiveCells(mesh, active, np.log(1e-8),
                                    nC=mesh.shape_cells[2])
    mapping = maps.ExpMap(mesh) * maps.SurjectVertical1D(mesh) * actMap

    # true conductivity model: air, half-space, and a thin layer
    sig_air = 1e-8
    sig_half = 2e-3
    sig_layer = 1e-3
    sigma = np.ones(mesh.shape_cells[2]) * sig_air
    sigma[active] = sig_half
    sigma[layer] = sig_layer
    mtrue = np.log(sigma[active])

    # receiver & source definition
    rx_offset = 1e-3
    rx = time_domain.Rx.PointMagneticFluxTimeDerivative(
        np.array([[rx_offset, 0.0, 30]]),
        np.logspace(-5, -3, 31),
        "z"
    )
    src = time_domain.Src.MagDipole([rx], location=np.array([0.0, 0.0, 80]))
    survey = time_domain.Survey([src])

    # time stepping & simulation
    time_steps = [(1e-6, 20), (1e-5, 20), (1e-4, 20)]
    simulation = time_domain.Simulation3DElectricField(
        mesh, sigmaMap=mapping, survey=survey, time_steps=time_steps
    )

    # synthetic data with 5% noise
    data = simulation.make_synthetic_data(mtrue, relative_error=0.05)

    # inversion setup
    dmisfit = data_misfit.L2DataMisfit(simulation=simulation, data=data)
    regMesh = discretize.TensorMesh([mesh.h[2][mapping.maps[-1].active_cells]])
    reg = regularization.WeightedLeastSquares(regMesh, alpha_s=1e-2, alpha_x=1.0)
    opt = optimization.InexactGaussNewton(maxIter=5, LSshorten=0.5)
    invProb = inverse_problem.BaseInvProblem(dmisfit, reg, opt)
    beta = directives.BetaSchedule(coolingFactor=5, coolingRate=2)
    betaest = directives.BetaEstimate_ByEig(beta0_ratio=1.0)
    inv = inversion.BaseInversion(invProb, directiveList=[beta, betaest])
    m0 = np.log(np.ones(mtrue.size) * sig_half)
    simulation.counter = opt.counter = utils.Counter()
    opt.remember("xc")
    mopt = inv.run(m0)

    # optional plotting
    if plotIt:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].loglog(rx.times, -invProb.dpred, "b.-")
        ax[0].loglog(rx.times, -data.dobs, "r.-")
        ax[0].legend(("Noisefree", "Observed"), fontsize=12)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("dB/dt (T/s)")
        ax[0].grid(True, linestyle="--", alpha=0.5)

        depths = mesh.cell_centers_z[active]
        ax[1].semilogx(sigma[active], depths, "k--")
        ax[1].semilogx(np.exp(mopt), depths, "r-")
        ax[1].set_ylim(depths.min(), depths.max())
        ax[1].set_xlabel("Conductivity (S/m)")
        ax[1].set_ylabel("Depth (m)")
        ax[1].legend(("True", "Recovered"), fontsize=12)
        ax[1].grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # build wide‐form data & layer parameters
    df_wide, df_params = make_wide_and_params(
        mesh, sigma, rx.times, data.dobs
    )
    print("→ synthetic_EM_wide.csv and layer_params.csv written.")


if __name__ == "__main__":
    run()
