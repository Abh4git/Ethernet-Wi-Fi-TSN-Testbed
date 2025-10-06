#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi AP Placement Tool using ITU‑R P.1238 (site‑general model) + Extension APs

Features
--------
- P.1238 path loss: L = 20*log10(f_MHz) - 28 + N * log10(d) + Lf(n), d >= 1 m
- Optional simple multi‑wall add‑on (sum per‑wall attenuation for line crossing)
- Floor penetration loss via user‑provided table (Lf_table)
- Greedy placement of Main APs to meet target RSSI
- Greedy placement of Extension APs (mesh/satellite) subject to *backhaul viability*
  (donor main -> extension RSSI >= backhaul_min_rssi_dBm)
- Heatmap of max RSSI across all APs (plotted by default)
- Separate radio params and candidate spacing for main vs extension APs

Usage
-----
python3 wifi_coverage_p1238.py --config example_floorplan.json
(Heatmap shows by default; no need for a --plot flag)

JSON config keys (selected)
---------------------------
Geometry & model:
  width_m, height_m, grid_res_m, f_MHz, N, floors_between, Lf_table, walls[]
Radio:
  Pt_dBm, Gt_dBi, Gr_dBi, target_rssi_dBm
Main APs:
  main_ap_count (or legacy ap_count), candidate_step_m, main_seed_positions[]
Extension APs:
  ext_ap_count, ext_candidate_step_m, ext_Pt_dBm, ext_Gt_dBi,
  ext_seed_positions[], backhaul_min_rssi_dBm, backhaul_Gr_dBi

Notes
-----
- For 6 GHz Wi‑Fi, start with 5.2 GHz N values and calibrate.
- Wall losses are simplified; for accuracy, measure or use material tables.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Core propagation calculations
# ------------------------------

def p1238_path_loss_dB(f_MHz: float, d_m: float, N: float, floors_between: int = 0, Lf_table: Optional[Dict[int, float]] = None) -> float:
    d = max(d_m, 1.0)
    L_d0 = 20.0 * math.log10(f_MHz) - 28.0  # f in MHz
    # Floor loss
    Lf = 0.0
    if floors_between > 0:
        if Lf_table and floors_between in Lf_table:
            Lf = Lf_table[floors_between]
        elif Lf_table and 1 in Lf_table:
            # simple linear extension if only 1-floor is given
            step = Lf_table.get(2, Lf_table[1] + 4) - Lf_table[1]
            Lf = Lf_table[1] + (floors_between - 1) * step
        else:
            # fallback gentle growth
            Lf = 15 + 4 * (floors_between - 1)
    return L_d0 + N * math.log10(d) + Lf


@dataclass
class Wall:
    """Axis‑aligned wall segment with attenuation per crossing (dB)."""
    x1: float; y1: float; x2: float; y2: float
    loss_dB: float = 3.0

def segments_intersect(p1, p2, q1, q2) -> bool:
    def orientation(a,b,c):
        val = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
        if abs(val) < 1e-12: return 0
        return 1 if val > 0 else 2
    def onseg(a,b,c):
        return min(a[0],c[0]) - 1e-12 <= b[0] <= max(a[0],c[0]) + 1e-12 and \
               min(a[1],c[1]) - 1e-12 <= b[1] <= max(a[1],c[1]) + 1e-12
    o1=orientation(p1,p2,q1); o2=orientation(p1,p2,q2); o3=orientation(q1,q2,p1); o4=orientation(q1,q2,p2)
    if o1!=o2 and o3!=o4: return True
    if o1==0 and onseg(p1,q1,p2): return True
    if o2==0 and onseg(p1,q2,p2): return True
    if o3==0 and onseg(q1,p1,q2): return True
    if o4==0 and onseg(q1,p2,q2): return True
    return False

def multiwall_loss_dB(tx: Tuple[float,float], rx: Tuple[float,float], walls: List[Wall]) -> float:
    loss = 0.0
    for w in walls:
        if segments_intersect(tx, rx, (w.x1,w.y1), (w.x2,w.y2)):
            loss += w.loss_dB
    return loss

def rssi_dBm(Pt_dBm: float, Gt_dBi: float, Gr_dBi: float, L_path_dB: float, L_walls_dB: float=0.0) -> float:
    return Pt_dBm + Gt_dBi + Gr_dBi - (L_path_dB + L_walls_dB)

# ------------------------------
# Grid model & greedy placement
# ------------------------------

@dataclass
class Scenario:
    width_m: float
    height_m: float
    grid_res_m: float = 1.0
    f_MHz: float = 5200.0
    N: float = 31.0
    Pt_dBm: float = 20.0
    Gt_dBi: float = 3.0
    Gr_dBi: float = 0.0
    floors_between: int = 0
    Lf_table: Optional[Dict[int,float]] = None
    walls: List[Wall] = field(default_factory=list)
    candidate_step_m: float = 3.0
    ap_count: int = 2                    # legacy: main AP count
    target_rssi_dBm: float = -67.0
    seed_positions: Optional[List[Tuple[float,float]]] = None  # legacy seeds (for main)

    # --- Extension AP controls ---
    main_ap_count: Optional[int] = None  # overrides ap_count if set
    main_seed_positions: Optional[List[Tuple[float,float]]] = None
    ext_ap_count: int = 0
    ext_candidate_step_m: Optional[float] = None  # defaults to candidate_step_m
    ext_Pt_dBm: Optional[float] = None           # defaults to Pt_dBm
    ext_Gt_dBi: Optional[float] = None           # defaults to Gt_dBi
    backhaul_min_rssi_dBm: float = -65.0
    backhaul_Gr_dBi: float = 0.0
    ext_seed_positions: Optional[List[Tuple[float,float]]] = None

    def grid_points(self) -> np.ndarray:
        xs = np.arange(0, self.width_m + 1e-9, self.grid_res_m)
        ys = np.arange(0, self.height_m + 1e-9, self.grid_res_m)
        xv, yv = np.meshgrid(xs, ys)
        return np.column_stack([xv.ravel(), yv.ravel()])

    def candidate_points(self, for_ext: bool=False) -> List[Tuple[float,float]]:
        step = self.ext_candidate_step_m if (for_ext and self.ext_candidate_step_m) else self.candidate_step_m
        xs = np.arange(step/2.0, self.width_m, step)
        ys = np.arange(step/2.0, self.height_m, step)
        cand = [(float(x), float(y)) for x in xs for y in ys]
        seeds = []
        if not for_ext and self.seed_positions:
            seeds += self.seed_positions
        if not for_ext and self.main_seed_positions:
            seeds += self.main_seed_positions
        if for_ext and self.ext_seed_positions:
            seeds += self.ext_seed_positions
        for p in seeds:
            if p not in cand:
                cand.append(p)
        return cand

    def _rssi_at_point_from(self, ap_xy: Tuple[float,float], rx_xy: Tuple[float,float],
                            Pt: float, Gt: float, Gr: float) -> float:
        d = math.hypot(ap_xy[0]-rx_xy[0], ap_xy[1]-rx_xy[1])
        Lp = p1238_path_loss_dB(self.f_MHz, max(d,1e-6), self.N, self.floors_between, self.Lf_table)
        Lw = multiwall_loss_dB(ap_xy, rx_xy, self.walls) if self.walls else 0.0
        return rssi_dBm(Pt, Gt, Gr, Lp, Lw)

    def rssi_map_for_AP(self, ap_xy: Tuple[float,float], Pt=None, Gt=None, Gr=None) -> np.ndarray:
        Pt = self.Pt_dBm if Pt is None else Pt
        Gt = self.Gt_dBi if Gt is None else Gt
        Gr = self.Gr_dBi if Gr is None else Gr
        pts = self.grid_points()
        rssis = np.empty(len(pts))
        for i,(x,y) in enumerate(pts):
            rssis[i] = self._rssi_at_point_from(ap_xy, (x,y), Pt, Gt, Gr)
        return rssis.reshape(int(self.height_m/self.grid_res_m)+1, int(self.width_m/self.grid_res_m)+1)

    def _greedy_cover(self, cands: List[Tuple[float,float]], Pt: float, Gt: float, Gr: float,
                      target: float, seeds: Optional[List[Tuple[float,float]]], count: int) -> Tuple[List[Tuple[float,float]], np.ndarray]:
        W = int(self.width_m/self.grid_res_m)+1
        H = int(self.height_m/self.grid_res_m)+1
        covered = np.zeros((H,W), dtype=bool)
        chosen = []
        masks = []
        for ap in cands:
            rssi_map = self.rssi_map_for_AP(ap, Pt, Gt, Gr)
            masks.append(rssi_map >= target)
        if seeds:
            for s in seeds:
                if s in cands:
                    idx = cands.index(s)
                else:
                    idx = None
                chosen.append(s)
                covered |= (masks[idx] if idx is not None else (self.rssi_map_for_AP(s, Pt, Gt, Gr) >= target))
        while len(chosen) < count:
            best_i, best_gain = None, -1
            for i,mask in enumerate(masks):
                if cands[i] in chosen: 
                    continue
                gain = np.count_nonzero(~covered & mask)
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i is None or best_gain <= 0:
                break
            chosen.append(cands[best_i])
            covered |= masks[best_i]
        # Final RSSI map: max over chosen
        if chosen:
            rssi_max = None
            for ap in chosen:
                m = self.rssi_map_for_AP(ap, Pt, Gt, Gr)
                rssi_max = m if rssi_max is None else np.maximum(rssi_max, m)
        else:
            rssi_max = np.full((H,W), -200.0)
        return chosen, rssi_max

    def greedy_place_main(self) -> Tuple[List[Tuple[float,float]], np.ndarray]:
        mcount = self.main_ap_count if self.main_ap_count is not None else self.ap_count
        seeds = self.main_seed_positions or self.seed_positions
        cands = self.candidate_points(for_ext=False)
        return self._greedy_cover(cands, self.Pt_dBm, self.Gt_dBi, self.Gr_dBi,
                                  self.target_rssi_dBm, seeds, mcount)

    def greedy_place_extensions(self, main_aps: List[Tuple[float,float]]) -> Tuple[List[Tuple[float,float]], np.ndarray, Dict[Tuple[float,float], float]]:
        if self.ext_ap_count <= 0:
            W = int(self.width_m/self.grid_res_m)+1
            H = int(self.height_m/self.grid_res_m)+1
            return [], np.full((H,W), -200.0), {}
        Pt = self.ext_Pt_dBm if self.ext_Pt_dBm is not None else self.Pt_dBm
        Gt = self.ext_Gt_dBi if self.ext_Gt_dBi is not None else self.Gt_dBi
        Gr = self.Gr_dBi
        cands = self.candidate_points(for_ext=True)
        seeds = self.ext_seed_positions or []

        # Filter candidates by backhaul viability (best main->candidate link)
        viable = []
        backhaul_rssi = {}
        for c in cands:
            best = -999.0
            for m in main_aps:
                r = self._rssi_at_point_from(m, c, self.Pt_dBm, self.Gt_dBi, self.backhaul_Gr_dBi)
                if r > best:
                    best = r
            if best >= self.backhaul_min_rssi_dBm or c in seeds:
                viable.append(c)
                backhaul_rssi[c] = best

        chosen, rssi_map = self._greedy_cover(viable, Pt, Gt, Gr, self.target_rssi_dBm, seeds, self.ext_ap_count)
        chosen_backhaul = {ap: backhaul_rssi.get(ap, float('-inf')) for ap in chosen}
        return chosen, rssi_map, chosen_backhaul

    def place_all(self) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]], np.ndarray, Dict[Tuple[float,float], float]]:
        main_aps, main_map = self.greedy_place_main()
        ext_aps, ext_map, bh = self.greedy_place_extensions(main_aps)
        combined = np.maximum(main_map, ext_map)
        return main_aps, ext_aps, combined, bh

    def plot_heatmap(self, rssi_map: np.ndarray, main_aps: List[Tuple[float,float]], ext_aps: Optional[List[Tuple[float,float]]]=None):
        plt.figure(figsize=(7,5))
        plt.imshow(rssi_map, origin='lower', extent=[0,self.width_m,0,self.height_m])
        plt.colorbar(label='RSSI (dBm)')
        for (x1,y1,x2,y2,loss) in [(w.x1,w.y1,w.x2,w.y2,w.loss_dB) for w in self.walls]:
            plt.plot([x1,x2],[y1,y2])
        if main_aps:
            xs, ys = zip(*main_aps)
            plt.scatter(xs, ys, marker='^', label='Main APs')
        if ext_aps:
            xs, ys = zip(*ext_aps)
            plt.scatter(xs, ys, marker='s', label='Extension APs')
        plt.title('Max RSSI from chosen APs')
        plt.xlabel('x (m)'); plt.ylabel('y (m)')
        if main_aps or ext_aps:
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

def load_scenario(json_path: str) -> Scenario:
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    walls = [Wall(**w) for w in cfg.get('walls', [])]
    def to_tuples(arr):
        return [tuple(p) for p in arr] if arr else None
    # Normalize Lf_table keys to int if provided as strings
    Lf = cfg.get('Lf_table', None)
    if isinstance(Lf, dict):
        Lf = {int(k): float(v) for k, v in Lf.items()}
    sc = Scenario(
        width_m = cfg['width_m'],
        height_m = cfg['height_m'],
        grid_res_m = cfg.get('grid_res_m', 1.0),
        f_MHz = cfg.get('f_MHz', 5200.0),
        N = cfg.get('N', 31.0),
        Pt_dBm = cfg.get('Pt_dBm', 20.0),
        Gt_dBi = cfg.get('Gt_dBi', 3.0),
        Gr_dBi = cfg.get('Gr_dBi', 0.0),
        floors_between = cfg.get('floors_between', 0),
        Lf_table = Lf,
        walls = walls,
        candidate_step_m = cfg.get('candidate_step_m', 3.0),
        ap_count = cfg.get('ap_count', 2),
        target_rssi_dBm = cfg.get('target_rssi_dBm', -67.0),
        seed_positions = to_tuples(cfg.get('seed_positions')),
        main_ap_count = cfg.get('main_ap_count'),
        main_seed_positions = to_tuples(cfg.get('main_seed_positions')),
        ext_ap_count = cfg.get('ext_ap_count', 0),
        ext_candidate_step_m = cfg.get('ext_candidate_step_m'),
        ext_Pt_dBm = cfg.get('ext_Pt_dBm'),
        ext_Gt_dBi = cfg.get('ext_Gt_dBi'),
        backhaul_min_rssi_dBm = cfg.get('backhaul_min_rssi_dBm', -65.0),
        backhaul_Gr_dBi = cfg.get('backhaul_Gr_dBi', 0.0),
        ext_seed_positions = to_tuples(cfg.get('ext_seed_positions'))
    )
    return sc

def main():
    import argparse
    p = argparse.ArgumentParser(description="WiFi AP placement using ITU‑R P.1238 (with extension APs)")
    p.add_argument("--config", default="example_floorplanv1.json", help="Path to JSON config (see example_floorplan.json)")
    args = p.parse_args()

    sc = load_scenario(args.config)
    main_aps, ext_aps, rssi_map, bh = sc.place_all()
    covered = (rssi_map >= sc.target_rssi_dBm).mean() * 100.0
    print(f"Main APs: {main_aps}")
    if sc.ext_ap_count > 0:
        print(f"Extension APs: {ext_aps}")
        for ap, r in bh.items():
            print(f"  Backhaul RSSI at {ap}: {r:.1f} dBm (>= {sc.backhaul_min_rssi_dBm} dBm required)")
    print(f"Coverage >= {sc.target_rssi_dBm:.1f} dBm: {covered:.2f}% of area")
    # Always plot by default
    sc.plot_heatmap(rssi_map, main_aps, ext_aps)

if __name__ == "__main__":
    main()
