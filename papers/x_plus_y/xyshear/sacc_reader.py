import sacc
import numpy as np
import pyccl as ccl
import argparse
from xyshear import xyshearLike
import xcell.mappers as xcm
import yaml
import matplotlib.pyplot as plt


def get_tracers_list(sacc_obj, probes_order, probe_aliases, DES_bin_id=None):
    """
    """
    tracers_sacc = sacc_obj.get_tracer_combinations()

    tracers = {p: [] for p in probes_order}

    for p1, p2 in probes_order:
        tr1 = probe_aliases[p1]
        tr2 = probe_aliases[p2]

        if "DES" in tr1+tr2:
            if DES_bin_id is not None:
                if "DES" in tr1:
                    tr_tuple = (f"{tr1}__{DES_bin_id}", tr2)
                else:
                    tr_tuple = (tr1, f"{tr2}__{DES_bin_id}")
                
                if tr_tuple in tracers_sacc:
                    tracers[p1+p2].append(tr_tuple)
                else:
                    tracers[p1+p2].append(tr_tuple[::-1])

            else:
                for bin_id in range(4):
                    if "DES" in tr1:
                        tr_tuple = (f"{tr1}__{bin_id}", tr2)
                    else:
                        tr_tuple = (tr1, f"{tr2}__{bin_id}")

                    if tr_tuple in tracers_sacc:
                        tracers[p1+p2].append(tr_tuple)
                    
                    else:
                        tracers[p1+p2].append(tr_tuple[::-1])

        else:
            tr_tuple = (tr1, tr2)
            if tr_tuple in tracers_sacc:
                tracers[p1+p2].append(tr_tuple)
            else:
                tracers[p1+p2].append(tr_tuple[::-1])

    return tracers


def extract_cls_vec_and_cov(sacc_obj, probes_order, probe_aliases, probe_spins, probe_lims, DES_bin_id=None):
    """
    """
    tracers_list = get_tracers_list(sacc_obj, probes_order, probe_aliases, DES_bin_id=DES_bin_id)
    
    cls_vec = []
    inds = []
    ells_vec = []

    for p1, p2 in probes_order:
        spin = f"cl_{probe_spins[p1]}{probe_spins[p2]}"
        if spin == "cl_e0":
            spin = "cl_0e"
        lmin1, lmax1 = probe_lims[p1]
        lmin2, lmax2 = probe_lims[p2]
        lmin, lmax = max(lmin1, lmin2), min(lmax1, lmax2)


        for t1, t2 in tracers_list[p1+p2]:
            ells, cells, ind = sacc_obj.get_ell_cl(spin, t1, t2, return_ind=True)
            mask = (ells >= lmin) & (ells <= lmax)

            ells, cells, ind = ells[mask], cells[mask], ind[mask]
            cls_vec.append(cells)
            ells_vec.append(ells)
            inds += list(ind)

    cov = sacc_obj.covariance
    cov = cov.keeping_indices(inds).covmat

    return ells_vec, cls_vec, cov


def get_model_from_cobaya(mode, lmin, lmax=2000, DES_bin_id=None):
    """
    """
    path_to_data = "/mnt/zfsusers/alaposta/tSZ_shear_correlations/outputs/tSZxDESy3WL_ROSAT_p15"
    tSZ_name = "tSZ15"

    config = {
        "yshearLike": {
            "external": xyshearLike,
            "path_to_data": path_to_data,
            "DES_bin_id": DES_bin_id,
            "tSZ_name": tSZ_name,
            "lmin": lmin,
            "lmax": lmax,
            "use_baryon_profile": False,
            "mode": mode,
            "rosat_data": "/mnt/zfsusers/alaposta/tSZ_shear_correlations/software_131023/fork/xCell-likelihoods/papers/ROSATx/data/"
        }
    }

    params = {
        "lMc": 14.0,
        "alpha_T": 1.0,
        "gamma": 1.1,
        "eta_b": 0.5
    }

    from cobaya.model import get_model
    model = get_model({
        "likelihood": config,
        "params": params
    })

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sacc_file", type=str)

    args = parser.parse_args()

    s = sacc.Sacc.load_fits(args.sacc_file)

    probes_order = ["sx", "sy", "xy"]
    probe_aliases = {
        "s": "DESY3wl",
        "x": "ROSAT",
        "y": "Planck__tSZ15"
    }
    probe_spins = {
        "s": "e",
        "x": "0",
        "y": "0"
    }

    probe_lims = {
        "s": (0, 4000),
        "x": (0, 4000),
        "y": (0, 4000),
    }

    DES_bin_id = None

    ells, cls_vec, cov = extract_cls_vec_and_cov(
        s, probes_order, 
        probe_aliases, probe_spins, 
        probe_lims, DES_bin_id=DES_bin_id
    )

    path_to_inputs = xcm.__file__.replace("xcell/mappers/__init__.py", "input")

    config_Xray = f"{path_to_inputs}/ROSAT.yml"
    config_tSZ = f"{path_to_inputs}/Planck__tSZ15.yml"

    with open(config_Xray, "r") as f:
        config = yaml.safe_load(f)
        config["nside"] = 1024
        config["coords"] = "C"
        mapper_Xray = xcm.MapperROSATXray(config)

    with open(config_tSZ, "r") as f:
        config = yaml.safe_load(f)
        config["nside"] = 1024
        config["coords"] = "C"
        mapper_tSZ = xcm.MapperP15tSZ(config)

    mask_Xray = mapper_Xray.get_mask()
    mask_tSZ = mapper_tSZ.get_mask()
    mask_Xray[mask_Xray > 0] = 1.

    mask_tot = mask_Xray * mask_tSZ

    fsky = np.mean(mask_tot)

    like_xy = get_model_from_cobaya("xy", lmin=30).likelihood["yshearLike"]
    
    tkk = ccl.halos.halomod_Tk3D_1h(
        like_xy.cosmology,
        like_xy.hm_calc,
        like_xy.profile_Xray,
        prof2=like_xy.profile_tSZ,
        prof12_2pt=like_xy.prof_2pt,
        lk_arr=np.log(like_xy.k_arr),
        a_arr=like_xy.a_arr,
        use_log=True
    )
    tll = ccl.angular_cl_cov_cNG(
        like_xy.cosmology,
        like_xy.tracer_Xray,
        like_xy.tracer_tSZ,
        ell=ells[-1],
        t_of_kk_a=tkk,
        fsky=fsky
    )

    n = len(ells[-1])

    cov_std = cov[-n:, -n:]
    cov_ng = tll

    full_cov_ng = s.covariance.covmat.copy()
    indices = s.indices(tracers=("ROSAT", "Planck__tSZ15"))
    full_cov_ng[np.ix_(indices, indices)] = full_cov_ng[np.ix_(indices, indices)] + tll

    s.covariance.covmat = full_cov_ng 

    print(((cov_std+cov_ng)/cov_std).diagonal())

    plt.figure()
    plt.plot(ells[-1], cov_std.diagonal(), color="navy", lw=3., label="Gaussian covariance")
    plt.plot(ells[-1], cov_std.diagonal()+cov_ng.diagonal(), color="forestgreen", lw=3., label="Gaussian + NG covariance")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\sigma^2_{Xy}$")
    plt.tight_layout()
    plt.savefig("xy_ng_cov.png", dpi=300)

    s.save_fits(args.sacc_file.replace(".fits", "_NG.fits"))