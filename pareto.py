"""Enerji ile motor aşınması arasındaki dengeyi (Pareto eğrisi) çizer.

Yalnızca kalkış cezası (w_start) farklı olan PPO (Proximal Policy Optimization) modelleri aynı test
günlerinde çalıştırılır ve enerji kalkış sayısına karşı çizilir. Grafiğe bakan biri ne kadar
tasarruf için ne kadar motor aşınmasını kabul edeceğine karar verebilir.

    uv run python pareto.py    # grafiği runs/pareto.png'ye çizer
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate import make_policies, run_episode

# Her satırda kalkış cezası w_start ve model dosyası yazılıdır. w=1 olan model aşamalardaki C modelidir (PPO-C).
MODELS = [
    (0.3, Path("runs/pareto_w0.3/final_model.zip")),
    (1.0, Path("runs/yol_C/final_model.zip")),
    (3.0, Path("runs/pareto_w3/final_model.zip")),
    (10.0, Path("runs/pareto_w10/final_model.zip")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)  # ortalaması alınacak test günü sayısı
    ap.add_argument("--seed", type=int, default=12345)  # test günlerinin seed'i, evaluate.py ile aynı
    ap.add_argument("--out", type=Path, default=Path("runs/pareto.png"))
    args = ap.parse_args()

    policies = make_policies([m for _, m in MODELS])
    # her kontrolcünün ortalama enerjisi [kWh], kalkış sayısı ve 6,2 bar altında geçen süresi [dk] adına göre tutulur
    points = {}
    for name, (policy, env) in policies.items():
        stats = [run_episode(env, policy, args.seed + ep, None)[0] for ep in range(args.episodes)]
        points[name] = tuple(np.mean([s[k] for s in stats]) for k in ("energy_kwh", "starts", "margin_min"))
        print(f"{name:<28} enerji {points[name][0]:.1f} kWh, kalkış {points[name][1]:.1f}, "
              f"6,2 altı {points[name][2]:.1f} dk")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    names = list(policies)[1:]  # ilki baseline'dır, kalanlar MODELS sırasını izler
    xs = [points[n][1] for n in names]
    ys = [points[n][0] for n in names]
    ax.plot(xs, ys, "o-", color="tab:orange", label="PPO (farklı kalkış cezaları)")
    for (w, _), x, y in zip(MODELS, xs, ys):
        ax.annotate(f"w_start={w:g}", (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    bx, by = points["Kaskad (baseline)"][1], points["Kaskad (baseline)"][0]
    ax.plot(bx, by, "s", color="tab:blue", ms=9, label="Kaskad (baseline)")
    ax.set_xlabel("Motor kalkışı [adet / 8 saat]")
    ax.set_ylabel("Enerji [kWh / 8 saat]")
    ax.set_title(f"Enerji ve motor kalkışı, {args.episodes} test günü")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Grafik: {args.out}")


if __name__ == "__main__":
    main()
