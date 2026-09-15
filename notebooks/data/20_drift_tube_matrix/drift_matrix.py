"""DRIFT-OPT matrix runner: the drift-tube PCB optimisation campaign.
Env: COMBOS="i0:i1" slice of the W7xW7 list; appends drift_matrix.csv."""
import csv, itertools, os, sys, time
import numpy as np
from pathlib import Path
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.build_rz import build_rz_model

W7 = [0.4, 0.5, 0.8, 1.0, 1.2, 1.6, 2.4]          # declared ring widths, mm
E_VMM, L_LADDER, X0, R_OUT_MAX = 0.2, 81.0, 2.0, 20.0
R_IN_SCAN = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
GAMMAS = np.arange(0.05, 0.96, 0.05)
BIAS_GATE = 1e-3
# BASE resolves through the repo: the campaign cut baked
# the certifying machine's absolute path here, which is why this
# bundle only ever ran there. The deck of record ships in examples/.
from ion_gym.io.paths import repo_root as _repo_root
BASE = str(_repo_root() / "examples" / "drift_tube_mason_schamp.json")
CKPT = Path(__file__).parent / "drift_matrix.csv"
FIELDS = ["w_c", "w_s", "p", "r_in", "h", "n_rings", "r_out",
          "gamma_eff", "bias_g05", "bias_g50", "bias_g90", "solve_s"]


def build_spec(w_c, w_s, r_in, h, d=None):
    import json
    p = w_c + w_s
    n = int(round(L_LADDER / p)) + 1
    L = (n - 1) * p
    d = d if d is not None else max(2.0 * p, 5.0)
    v_in = E_VMM * L
    spec = json.load(open(BASE))
    g = spec["geometry"]
    g["width_mm"] = X0 + L + w_c + 2.0
    g["height_mm"] = r_in + d + 0.5
    g["mm_per_gu"] = h
    g["electrodes"] = [
        {"name": f"ring_{i+1:02d}",
         "shapes": [{"type": "rect", "x_mm": X0 + i * p, "y_mm": r_in,
                     "width_mm": w_c, "height_mm": d}],
         "stl": None, "is_grid": False,
         "dc": round(v_in * (1.0 - i / (n - 1)), 6),
         "rf_groups": [], "dc_group": "ladder", "dc_index": i + 1,
         "dc_weight": None, "color": [120, 120, 200], "basis": None,
         "rf_group": None} for i in range(n)]
    g["dc_groups"][0]["v_in"] = v_in
    spec["name"] = f"drift wc{w_c} ws{w_s} rin{r_in}"
    return SimSpec.from_dict(spec), n, L, d


def evaluate(w_c, w_s, r_in, h=None):
    h = h or min(0.25, max(0.1, min(w_c, w_s) / 2.0))
    sp, n, L, d = build_spec(w_c, w_s, r_in, h)
    t0 = time.time()
    m = build_rz_model(sp)
    mm, u0 = m.mm_per_gu, m.u0
    lo, hi = X0 + 3.0 * r_in, X0 + L - 3.0 * r_in
    if hi - lo < 12.0:
        return None                      # window too short at this r_in
    xs = np.arange(lo, hi + mm / 2, mm)
    ix = np.round(xs / mm).astype(int)

    def col(r):
        iu = int(round((r - u0) / mm))
        return np.array([m.EzA[i, iu] for i in ix])
    inv0 = np.mean(1.0 / col(0.0))
    biases = {}
    g_eff = 0.0
    for gam in GAMMAS:
        b = abs(np.mean(1.0 / col(gam * r_in)) / inv0 - 1.0)
        biases[round(gam, 2)] = b
        if b < BIAS_GATE and g_eff == round(gam - 0.05, 2):
            pass
    # largest contiguous-from-axis Gamma under the gate
    for gam in GAMMAS:
        if biases[round(gam, 2)] < BIAS_GATE:
            g_eff = round(gam, 2)
        else:
            break
    return dict(w_c=w_c, w_s=w_s, p=w_c + w_s, r_in=r_in, h=h,
                n_rings=n, r_out=r_in + d, gamma_eff=g_eff,
                bias_g05=biases[0.05], bias_g50=biases[0.5],
                bias_g90=biases[0.9], solve_s=round(time.time() - t0, 1))


def main():
    combos = [c for c in itertools.product(W7, W7)]
    i0, i1 = (int(v) for v in os.environ.get("COMBOS", "0:49").split(":"))
    done = set()
    if CKPT.exists():
        for r in csv.DictReader(open(CKPT)):
            done.add((float(r["w_c"]), float(r["w_s"]), float(r["r_in"])))
    new = not CKPT.exists()
    with open(CKPT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for w_c, w_s in combos[i0:i1]:
            if w_c + w_s + max(2 * (w_c + w_s), 5.0) > R_OUT_MAX + 10.0:
                pass                     # r_out constraint applied per r_in
            for r_in in R_IN_SCAN:
                if (w_c, w_s, r_in) in done:
                    continue
                d = max(2.0 * (w_c + w_s), 5.0)
                if r_in + d > R_OUT_MAX:
                    continue
                row = evaluate(w_c, w_s, r_in)
                if row is None:
                    continue
                w.writerow(row); f.flush()
            print(f"combo w_c={w_c} w_s={w_s} done "
                  f"({time.strftime('%H:%M:%S')})", flush=True)


if __name__ == "__main__":
    main()
