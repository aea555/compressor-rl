"""Eğitilmiş ajanların eğitimde görmedikleri durumlarda ne yaptığını test eder.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Modeller yeniden eğitilmez, yalnızca ortamın koşulları değiştirilir. Her senaryoda baseline ve ajanlar
aynı seed'lerle aynı günlerde karşılaştırılır. Sonuçlar runs/dayaniklilik/ altına yazılır. sonuclar.md
her senaryo için kontrolcülerin sonuç tablosunu, <senaryo>.png ise senaryonun ilk gününün grafiğini tutar.

    uv run python robustness.py                                                       # PPO w=3, PPO w=1, BDQ ve CEM'i test eder
    uv run python robustness.py --model runs/yol_C/final_model.zip --episodes 5      # yalnızca PPO-C'yi test eder
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from compressor_env import CompressorEnv
from evaluate import make_policies, plot_episode, run_episode

# Her senaryoda kısa ad, açıklama, değişen ortam ayarları, başlangıç saati ve gri alan yazılıdır.
# Değişen ayarlar EnvConfig'teki alanların üzerine yazılır (compressor_env/env.py).
# Başlangıç saati None ise eğitimdeki gibi her gün rastgele seçilir.
# Gri alan grafikte işaretlenen saat aralığını gösterir, arızalı senaryolarda arızanın sürdüğü saatler boyanır.
SCENARIOS = [
    ("normal", "Normal (eğitimdeki koşullar)", {}, None, None),
    ("k2_ariza", "K2 arızası, gündüz vardiyasında 1 saat devre dışı",
     {"failure": (1, 3.0, 1.0)}, 6.0, (3.0, 4.0)),
    ("k1_ariza", "K1 arızası, gece vardiyasında 1 saat devre dışı",
     {"failure": (0, 2.0, 1.0)}, 22.0, (2.0, 3.0)),
    ("talep_+20", "Talep %20 fazla (yeni üretim hattı)", {"demand_scale": 1.2}, None, None),
    ("talep_-30", "Talep %30 az (hafta sonu ya da düşük üretim)", {"demand_scale": 0.7}, None, None),
    ("kacak_x2", "Kaçaklar iki katı (bakımsız hat)", {"leak_m3min": 4.0}, None, None),
    ("24_saat", "24 saat kesintisiz (eğitimde 8 saat)", {"episode_hours": 24.0}, 0.0, None),
]

# Her satırda episode_stats anahtarı, tablodaki sütun başlığı ve sayı biçimi yazılıdır.
METRICS = [
    ("specific_power", "Özgül güç", "{:.2f}"),
    ("low_p_min", "6 bar altı [dk]", "{:.1f}"),
    ("margin_min", "6,2 bar altı [dk]", "{:.1f}"),
    ("starts", "Kalkış", "{:.1f}"),
    ("excess_starts", "Limit aşımı", "{:.1f}"),
]


def main():
    ap = argparse.ArgumentParser()
    # test edilecek modeller verilmezse PPO w=3, PPO w=1, BDQ ve CEM kullanılır
    ap.add_argument("--model", type=Path, nargs="*",
                    default=[Path("runs/pareto_w3/final_model.zip"), Path("runs/yol_C/final_model.zip"),
                             Path("runs/bdq_5m/bdq_model.pt"), Path("runs/cem_sabit/cem_policy.json")])
    ap.add_argument("--episodes", type=int, default=10)  # her senaryoda test edilecek gün sayısı
    ap.add_argument("--seed", type=int, default=12345)  # test günlerinin seed'i, evaluate.py ile aynı
    ap.add_argument("--out", type=Path, default=Path("runs/dayaniklilik"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # her kontrolcü için komut üreten fonksiyon ve modelin eğitildiği ortam hazırlanır
    policies = make_policies(args.model)
    lines = ["# Dayanıklılık testleri", "", f"Her senaryoda {args.episodes} gün ortalaması. "
             "Modeller yeniden eğitilmedi.", ""]

    for key, desc, overrides, start_hour, shade in SCENARIOS:
        print(f"\n=== {desc}")
        results = {}  # kontrolcü adına göre her bölümün metrikleri
        first_logs = {}  # grafik için ilk günün adım adım kaydı
        for name, (policy, train_env) in policies.items():
            # modelin kendi gözlem ve ödül ayarları korunur, senaryonun değişiklikleri bunların üstüne yazılır
            env = CompressorEnv(replace(train_env.cfg, **overrides))
            results[name] = []
            for ep in range(args.episodes):
                stats, log = run_episode(env, policy, args.seed + ep, start_hour)
                results[name].append(stats)
                if ep == 0:
                    first_logs[name] = log

        # tablonun satırlarında kontrolcüler, sütunlarında metrikler yer alır
        lines += [f"## {desc}", "", "| Kontrolcü | " + " | ".join(m[1] for m in METRICS) + " |",
                  "|---" * (len(METRICS) + 1) + "|"]
        for name, stats_list in results.items():
            cells = [fmt.format(np.mean([s[k] for s in stats_list])).replace(".", ",") for k, _, fmt in METRICS]  # sayılar Türkçedeki gibi virgüllü yazılır
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
            print(f"  {name:<28}" + "  ".join(f"{h}: {c}" for (_, h, _), c in zip(METRICS, cells)))
        lines.append("")

        plot_env = CompressorEnv(replace(policies["Kaskad (baseline)"][1].cfg, **overrides))
        plot_episode(first_logs, plot_env, args.out / f"{key}.png", shade=shade, title=desc)

    (args.out / "sonuclar.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nTablo: {args.out / 'sonuclar.md'}")


if __name__ == "__main__":
    main()
