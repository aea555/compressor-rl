"""Eğitilmiş ajanın ne öğrendiğine bakar ve bunu baseline'ın sabit eşikleriyle karşılaştırır.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Ajan bir sinir ağı olduğu için içinden kural okunamaz, bu yüzden ajanın davranışına bakılır. Baseline ile
ajan aynı günlerde çalıştırılır ve hangi basınçta, hangi talepte ne yaptıkları sayılır. Sonuç
runs/analiz/<model>.png dosyasına 4 parçalı bir grafik olarak çizilir, ekrana da özet yazılır.

    uv run python interpret.py                                                  # PPO-C'yi (runs/yol_C) inceler
    uv run python interpret.py --model runs/bdq_5m/bdq_model.pt --episodes 10   # BDQ'yu inceler
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compressor_env.env import LOAD
from evaluate import make_policies, model_label, run_episode

# baseline K2'yi 6,3 bar altında yüke, 7,1 bar üstünde boşa alır (compressor_env/baseline.py)
BASE_K2_ON, BASE_K2_OFF = 6.3, 7.1
BASE_VSD_SP = 6.9  # baseline'da VSD'nin tutmaya çalıştığı basınç [bar]


def collect(policy, env, episodes: int, seed: int):
    """Bir kontrolcüyü birkaç gün çalıştırıp bütün adımların kaydını bir araya toplar."""
    p, demand, k1, k2, vsd = [], [], [], [], []  # basınç [bar], talep [m³/dk], K1 ve K2 yükte mi, VSD devri (0-1)
    k2_on_p, k2_off_p = [], []  # K2'nin yüke alındığı ve yükten çıktığı andaki basınç [bar]
    for ep in range(episodes):
        _, log = run_episode(env, policy, seed + ep, None)
        prev_k2 = log[0]["state"][1]  # önceki adımda K2'nin durumu
        prev_p = log[0]["pressure"]  # önceki adımdaki basınç [bar], karar bu basınca bakılarak verilir
        for x in log:
            s = x["state"]
            p.append(x["pressure"])
            demand.append(x["demand"])
            k1.append(s[0] == LOAD)
            k2.append(s[1] == LOAD)
            vsd.append(x["vsd_speed"])
            # K2 bu adımda yüke girdiyse ya da yükten çıktıysa karar anındaki basınç kaydedilir
            if s[1] == LOAD and prev_k2 != LOAD:
                k2_on_p.append(prev_p)
            if s[1] != LOAD and prev_k2 == LOAD:
                k2_off_p.append(prev_p)
            prev_k2, prev_p = s[1], x["pressure"]
    arr = lambda a: np.asarray(a, dtype=float)
    return dict(p=arr(p), demand=arr(demand), k1=arr(k1), k2=arr(k2), vsd=arr(vsd),
                k2_on_p=arr(k2_on_p), k2_off_p=arr(k2_off_p))


def binned_mean(x, y, edges):
    """x'i aralıklara bölüp her aralıkta y'nin ortalamasını döndürür. Az veri olan aralıklar NaN kalır."""
    idx = np.digitize(x, edges) - 1  # her noktanın düştüğü aralık
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(out)):
        m = idx == i
        if m.sum() >= 30:  # 30 adımdan (5 dk) az veri varsa ortalamaya güvenilmez
            out[i] = y[m].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("runs/yol_C/final_model.zip"))  # incelenecek ajan
    ap.add_argument("--episodes", type=int, default=20)  # istatistik için çalıştırılacak gün sayısı
    ap.add_argument("--seed", type=int, default=12345)  # test günlerinin seed'i, evaluate.py ile aynı
    ap.add_argument("--out", type=Path, default=Path("runs/analiz"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policies = make_policies([args.model])
    data = {name: collect(pol, env, args.episodes, args.seed) for name, (pol, env) in policies.items()}
    agent = model_label(args.model)
    colors = {"Kaskad (baseline)": "tab:blue", agent: "tab:orange"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    p_edges = np.arange(5.8, 7.61, 0.1)  # basınç aralıkları [bar]
    p_mid = (p_edges[:-1] + p_edges[1:]) / 2
    d_edges = np.arange(8, 31, 2)  # talep aralıkları [m³/dk]
    d_mid = (d_edges[:-1] + d_edges[1:]) / 2

    # 1) basıncın en çok hangi değerlerde durduğu çizilir
    ax = axes[0, 0]
    for name, d in data.items():
        ax.hist(d["p"], bins=np.arange(5.5, 7.8, 0.05), density=True, alpha=0.5,
                color=colors[name], label=f"{name} (ort. {d['p'].mean():.2f})")
    ax.axvline(6.0, color="red", ls="--", lw=1, label="6 bar sınırı")
    ax.axvline(6.2, color="red", ls=":", lw=1, label="6,2 güvenlik payı")
    ax.set_xlabel("Basınç [bar]")
    ax.set_ylabel("Zamanın yoğunluğu")
    ax.set_title("Basınç nerede tutuluyor?")
    ax.legend(fontsize=8)

    # 2) K2'nin hangi basınçta yüke alınıp çıkarıldığı çizilir. Baseline'da bu eşikler 6,3 ve 7,1 bar'da sabittir.
    ax = axes[0, 1]
    bins = np.arange(5.5, 7.8, 0.05)
    for name, d in data.items():
        ax.hist(d["k2_on_p"], bins=bins, alpha=0.5, color=colors[name], histtype="stepfilled",
                label=f"{name}: yüke alma ({len(d['k2_on_p'])})")
        ax.hist(d["k2_off_p"], bins=bins, color=colors[name], histtype="step", lw=1.5, ls="--",
                label=f"{name}: yükten çıkarma ({len(d['k2_off_p'])})")
    ax.axvline(BASE_K2_ON, color="k", lw=0.8)
    ax.axvline(BASE_K2_OFF, color="k", lw=0.8)
    ax.text(BASE_K2_ON, ax.get_ylim()[1] * 0.95, " baseline\n açma eşiği", fontsize=7, va="top")
    ax.text(BASE_K2_OFF, ax.get_ylim()[1] * 0.95, " baseline\n kapama eşiği", fontsize=7, va="top")
    ax.set_xlabel("Karar anındaki basınç [bar]")
    ax.set_ylabel("Olay sayısı")
    ax.set_title("K2 (45 kW) hangi basınçta yüke alınıyor ve çıkarılıyor?")
    ax.legend(fontsize=7)

    # 3) VSD devrinin basınca göre nasıl değiştiği çizilir. Baseline'da devri 6,9 bar'ı tutmaya çalışan bir
    # PI kontrolcü ayarlar.
    ax = axes[1, 0]
    for name, d in data.items():
        ax.plot(p_mid, binned_mean(d["p"], d["vsd"], p_edges), "o-", color=colors[name], label=name)
    ax.axvline(BASE_VSD_SP, color="k", lw=0.8, ls=":", label="baseline VSD hedefi (6,9)")
    ax.set_xlabel("Basınç [bar]")
    ax.set_ylabel("Ortalama VSD devri (0..1)")
    ax.set_title("VSD basınca nasıl tepki veriyor?")
    ax.legend(fontsize=8)

    # 4) talebe göre hangi kompresörlerin yükte olduğu çizilir
    ax = axes[1, 1]
    for name, d in data.items():
        ax.plot(d_mid, binned_mean(d["demand"], d["k1"], d_edges), "o-", color=colors[name],
                label=f"{name}: K1 yükte")
        ax.plot(d_mid, binned_mean(d["demand"], d["k2"], d_edges), "s--", color=colors[name],
                label=f"{name}: K2 yükte")
    ax.set_xlabel("Hava talebi [m³/dk]")
    ax.set_ylabel("Yükte olma oranı")
    ax.set_title("Talebe göre hangi kompresörler çalışıyor?")
    ax.legend(fontsize=7)

    fig.suptitle(f"{agent} ve baseline, {args.episodes} gün")
    fig.tight_layout()
    path = args.out / f"{agent.replace('/', '_')}.png"
    fig.savefig(path, dpi=130)

    # ekrana kısa bir özet yazılır
    for name, d in data.items():
        on, off = d["k2_on_p"], d["k2_off_p"]
        print(f"\n{name}")
        print(f"  ortalama basınç {d['p'].mean():.2f} bar, %5-%95 aralığı "
              f"{np.percentile(d['p'], 5):.2f}–{np.percentile(d['p'], 95):.2f}")
        if len(on):
            print(f"  K2 yüke alma basıncı: medyan {np.median(on):.2f} bar ({len(on)} kez)")
        if len(off):
            print(f"  K2 yükten çıkarma basıncı: medyan {np.median(off):.2f} bar ({len(off)} kez)")
        print(f"  K1 yükte: %{100 * d['k1'].mean():.0f}, K2 yükte: %{100 * d['k2'].mean():.0f}, "
              f"VSD ort. devir {d['vsd'].mean():.2f}")
    print(f"\nGrafik: {path}")


if __name__ == "__main__":
    main()
