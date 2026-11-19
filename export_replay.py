"""Her model ve senaryo için baseline ile modeli aynı günde çalıştırıp Unity için JSON'a kaydeder.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Modeller Unity'de çalıştırılmaz. Simülasyon burada çalıştırılır ve sonuç kaydedilir, Unity (unity/KompresorViz)
yalnızca bu kayıtları oynatır. Kayıtlarda adım adım değerlerin yanında iki ek bilgi bulunur. Birincisi
enerjinin ne kadarının boşta çalışmaya, ne kadarının gereğinden yüksek basınca, ne kadarının hava
üretimine gittiğini gösterir. İkincisi karar günlüğüdür. Günlüğe her açma, kapama, yüke ve boşa alma
kararıyla birlikte o anki basınç, talep ve kararın sebebi yazılır.

Kayıtlar unity/KompresorViz/Assets/StreamingAssets/replays/ klasörüne yazılır. Orada index.json ve her
kayıt için bir dosya oluşur.

    uv run python export_replay.py          # bütün modelleri ve senaryoları kaydeder, yaklaşık 10 sn sürer
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from compressor_env import CascadeController, CompressorEnv
from compressor_env.env import LOAD, OFF, P_ATM, UNLOAD
from evaluate import make_policies
from robustness import SCENARIOS

# Arayüzdeki modeller burada verilir. Her satırda anahtar, model dosyası, ekrandaki ad ve kısa açıklama yazılıdır.
MODELS = [
    ("ppo_w3", "runs/pareto_w3/final_model.zip", "PPO, kalkış cezası 3",
     "Hem enerjide hem motor kalkışında baseline'dan iyi. Önerilen model bu."),
    ("ppo_w1", "runs/yol_C/final_model.zip", "PPO, kalkış cezası 1",
     "Yaklaşık %7 enerji tasarrufu sağlıyor, kalkış sayısı baseline'a yakın."),
    ("ppo_w0.3", "runs/pareto_w0.3/final_model.zip", "PPO, kalkış cezası 0,3",
     "En az enerjiyi harcıyor ama motorları çok sık açıp kapatıyor."),
    ("ppo_w10", "runs/pareto_w10/final_model.zip", "PPO, kalkış cezası 10",
     "Motorları korumaya öncelik veriyor. Neredeyse hiç kalkış yok ama tasarruf az."),
    ("bdq", "runs/bdq_5m/bdq_model.pt", "BDQ (5M adım)",
     "Değer tabanlı bir yöntemle eğitildi. Enerjide iyi ama kararsız, kararlarını sık sık geri alıyor."),
    ("q_w3", "runs/k2_w3_s0/q_table.npz", "Q tablo, kalkış cezası 3",
     "Vize için yapılan klasik RL modellerinden biri. Keşif planı düzeltilerek eğitildi. Enerjiyi %7 azaltıyor ama motorları baseline'ın yaklaşık 2,5 katı sık çalıştırıyor."),
    ("q_w6", "runs/k2_w6_s1/q_table.npz", "Q tablo, kalkış cezası 6",
     "Aynı tablonun kalkış cezası 6 ile eğitilmiş hali. Enerjiyi %6 azaltıyor ve saatlik kalkış limitini aşmıyor, kalkış sayısı yine baseline'dan fazla."),
    ("q_tile", "runs/tile_qlearning_s0/q_table.npz", "Q-learning (tile coding)",
     "Vize için yapılan klasik RL modellerinden biri. Benzer durumlar birbirinden öğreniyor ve kalkışlar azaldı. Ama kompresörleri boşta bekletiyor, enerjisi baseline'dan fazla."),
    ("cem", "runs/cem_sabit/cem_policy.json", "CEM",
     "Vize için yapılan klasik RL modellerinden biri. Kaskad kontrolün 9 ayarını ödüle bakarak öğrendi. Hem enerjide hem kalkışta baseline'dan iyi."),
]

# Arayüzdeki senaryolar burada verilir. Her satırda anahtar, ekrandaki ad, robustness.py'deki senaryo ve
# başlangıç saati yazılıdır. Başlangıç saati None ise robustness.py'deki saat kullanılır.
VIZ_SCENARIOS = [
    ("gece", "Gece vardiyası", "normal", 22.0),
    ("gunduz", "Gündüz vardiyası", "normal", 6.0),
    ("k2_ariza", "K2 arızası", "k2_ariza", None),
    ("talep_+20", "Talep +%20", "talep_+20", 14.0),
    ("kacak_x2", "Kaçak ×2", "kacak_x2", 6.0),
]



def num(x: float) -> str:
    """Sayıyı arayüzde gösterilecek şekilde 6,9 gibi virgüllü yazar."""
    return f"{x:g}".replace(".", ",")

def record(policy, env: CompressorEnv, seed: int, start_hour):
    """Bir bölümü çalıştırır ve Unity'de gösterilecek değerleri adım adım kaydeder."""
    cfg = env.cfg
    obs, _ = env.reset(seed=seed, options={"start_hour": start_hour})
    if hasattr(policy, "reset"):
        policy.reset(env)

    # basıncın güce ve tüketime etkisi env.py'deki formüllerle hesaplanır
    press_factor = lambda p: 1.0 + cfg.power_per_bar * (max(p, cfg.p_min - 1.0) - cfg.p_ref)  # güç çarpanı
    cons_factor = lambda p: np.sqrt((p + P_ATM) / (cfg.p_ref + P_ATM))  # tüketim (talep ve kaçak) çarpanı
    p_ref = cfg.p_margin  # yeterli sayılan basınç 6,2 bar'dır ve bunun üstü fazla sayılır

    # her listede adım başına bir değer tutulur
    rec = dict(hour=[], pressure=[], demand=[], power=[], k1=[], k2=[], vsdSpeed=[],
               temp1=[], temp2=[], temp3=[], energy=[], starts=[], lowMin=[],
               eUnload=[], ePressure=[], eProduction=[])
    energy = 0.0  # bölüm başından beri harcanan enerji [kWh]
    e_unload = 0.0  # bunun boşta çalışmaya giden kısmı [kWh]
    e_pressure = 0.0  # 6,2 bar'ın üstündeki basınç yüzünden fazladan harcanan kısmı [kWh]
    starts = 0  # bölüm başından beri motor kalkışı [adet]
    low_min = 0.0  # 6 bar altında geçen süre [dk]
    prev = list(env.state)
    done = False
    while not done:
        p_before = env.p  # güç bu adımın başındaki basınçla hesaplanır
        obs, _, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        s = info["state"]
        starts += sum(1 for i in range(env.n_comp) if prev[i] == OFF and s[i] != OFF)
        prev = list(s)

        # bu adımın gücü [kW] boşta çalışma, yüksek basınç ve üretim olarak üçe ayrılır
        power = info["power_kw"]
        unload_kw = sum(f.rated_kw * f.unload_frac for f, st in zip(cfg.fixed, s) if st == UNLOAD)
        run_kw = power - unload_kw  # hava basan motorların gücü
        # aynı kompresörler 6,2 bar'da çalışsaydı hem güç hem tüketim (kaçak dahil) daha az olurdu
        run_at_ref = run_kw * press_factor(p_ref) / press_factor(p_before)
        run_at_ref *= cons_factor(p_ref) / cons_factor(p_before)
        pressure_kw = run_kw - run_at_ref  # yüksek basıncın ek maliyeti [kW], 6,2 bar altında eksi olabilir
        h = cfg.dt / 3600.0  # adım süresi [saat]
        energy += power * h
        e_unload += unload_kw * h
        e_pressure += pressure_kw * h
        low_min += cfg.dt / 60.0 if info["pressure"] < cfg.p_min else 0.0

        rec["hour"].append(round(info["hour"], 4))
        rec["pressure"].append(round(info["pressure"], 4))
        rec["demand"].append(round(info["demand"], 3))
        rec["power"].append(round(power, 2))
        rec["k1"].append(int(s[0]))  # 0 kapalı, 1 boşta, 2 yükte
        rec["k2"].append(int(s[1]))
        rec["vsdSpeed"].append(round(info["vsd_speed"], 3))  # 0 kapalı, 0,2-1 arası devir oranı
        rec["temp1"].append(round(env.temp[0], 1))  # motor sıcaklıkları [°C]
        rec["temp2"].append(round(env.temp[1], 1))
        rec["temp3"].append(round(env.temp[2], 1))
        rec["energy"].append(round(energy, 2))
        rec["starts"].append(starts)
        rec["lowMin"].append(round(low_min, 2))
        rec["eUnload"].append(round(e_unload, 2))
        rec["ePressure"].append(round(e_pressure, 2))
        rec["eProduction"].append(round(energy - e_unload - e_pressure, 2))
    rec["summary"] = info["episode_stats"]
    return rec


def events(rec, names, is_baseline: bool, bands, vsd_sp: float, k1_flow: float):
    """Durum değişikliklerinden karar günlüğü çıkarır. Her kayıtta adım, olay ve sebep bulunur.

    Baseline'ın kuralları bilindiği için sebep kesin yazılabilir. Ajanın içi görülemediği için sebep
    kararın verildiği duruma bakılarak tahmin edilir.
    """
    out = []
    n = len(rec["pressure"])
    states = [rec["k1"], rec["k2"], [2 if v > 0 else 0 for v in rec["vsdSpeed"]]]
    for i in range(1, n):
        p, d = rec["pressure"][i - 1], rec["demand"][i - 1]  # karar anındaki basınç [bar] ve talep [m³/dk]
        for k in range(3):
            a, b = states[k][i - 1], states[k][i]  # önceki ve yeni durum
            if a == b:
                continue
            vsd = k == 2
            name = names[k]
            if vsd:
                what = f"{name} çalıştırıldı" if b else f"{name} durduruldu"
            else:
                what = {(OFF, LOAD): "çalıştırıldı, yükte", (OFF, UNLOAD): "çalıştırıldı, boşta",
                        (UNLOAD, LOAD): "yüke alındı", (LOAD, UNLOAD): "boşa alındı"}.get((a, b), "durduruldu")
                what = f"{name} {what}"
            # kompresör 30 sn içinde eski durumuna dönerse karar gereksiz sayılır, buna titreme denir
            back = any(states[k][j] == a for j in range(i + 1, min(i + 4, n)))
            if is_baseline:
                if vsd:
                    why = f"basınç {num(vsd_sp)} bar hedefinin altında" if b else "en düşük devirde bile basınç yüksek"
                elif b == LOAD:
                    why = f"basınç {num(bands[k][0])} bar'ın altına indi"
                elif b == UNLOAD:
                    why = f"basınç {num(bands[k][1])} bar'ı geçti"
                else:
                    why = "5 dakikadır boştaydı"
            elif back:
                why = "30 sn içinde geri alındı, karar gereksizdi"
            elif vsd:
                why = "basıncı ince ayarlamak için açtı" if b else "düşük devirde verimsiz olduğu için kapattı"
            elif b == LOAD:
                if k == 1 and states[0][i] != LOAD:
                    why = f"talep düşük ({d:.0f} m³/dk), büyük K1 yerine K2 ve VSD yetiyor"
                elif p > bands[k][0]:
                    why = f"baseline'ın {num(bands[k][0])} bar eşiğini beklemeden açtı"
                else:
                    why = "basınç düşüyordu, kapasite ekledi"
            elif a == LOAD and b == OFF:
                if k == 0 and states[1][i] == LOAD:
                    why = "talep düşük, K1 yerine K2 ile devam ediyor"
                else:
                    why = "boşta bekletmek yerine kapattı"
            elif b == UNLOAD:
                why = "boşa aldı, boşta da güç çeker"
            else:
                why = "boştaki kompresörü kapattı"
            out.append((i, f"{what}, {p:.2f} bar, {d:.1f} m³/dk".replace(".", ","), why))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345)  # hangi günün (talep profili) kullanılacağı
    ap.add_argument("--out", type=Path, default=Path("unity/KompresorViz/Assets/StreamingAssets/replays"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cascade = CascadeController()
    policies = make_policies([Path(m[1]) for m in MODELS])
    pol_list = list(policies.values())  # ilki baseline'dır, sonrakiler MODELS sırasını izler
    index = dict(models=[], scenarios=[], files=[])
    for key, _, label, desc in MODELS:
        short = label.replace("PPO, kalkış cezası ", "PPO w=").replace(" (5M adım)", "").replace("Q tablo, kalkış cezası ", "Q w=")
        short = short.replace("Q-learning (tile coding)", "Q tile").replace(" (politika araması)", "")  # buton yazısı
        index["models"].append(dict(key=key, label=label, short=short, desc=desc))
    for key, label, _, _ in VIZ_SCENARIOS:
        index["scenarios"].append(dict(key=key, label=label))

    for sc_key, sc_label, rob_key, hour in VIZ_SCENARIOS:
        _, rob_desc, overrides, rob_hour, shade = next(s for s in SCENARIOS if s[0] == rob_key)
        start_hour = hour if hour is not None else rob_hour
        base_policy, base_env = pol_list[0]
        base_env = CompressorEnv(replace(base_env.cfg, **overrides))
        base = record(base_policy, base_env, args.seed, start_hour)  # baseline her model için aynıdır
        cfg = base_env.cfg
        names = [f.name for f in cfg.fixed] + [cfg.vsd.name]
        for m, (m_key, _, m_label, _) in enumerate(MODELS):
            policy, train_env = pol_list[m + 1]
            env = CompressorEnv(replace(train_env.cfg, **overrides))
            agent = record(policy, env, args.seed, start_hour)
            runs = []
            for rec, label, is_base in ((base, "Klasik kaskad", True), (agent, m_label, False)):
                ev = events(rec, names, is_base, cascade.bands, cascade.sp, cfg.fixed[0].flow_m3min)
                st = rec["summary"]
                runs.append({k: v for k, v in rec.items() if k != "summary"} | dict(
                    name=label,
                    evStep=[e[0] for e in ev], evText=[e[1] for e in ev], evWhy=[e[2] for e in ev],
                    # özet ekranındaki tablo için bölüm sonu değerleri eklenir
                    sumSpecific=round(st["specific_power"], 3), sumMeanP=round(st["mean_p"], 2),
                    sumSwitches=st["load_switches"], sumExcess=st["excess_starts"],
                ))
            data = dict(
                title=rob_desc if rob_key != "normal" else sc_label,
                model=m_label, modelDesc=MODELS[m][3],
                dt=cfg.dt, pMin=cfg.p_min, pMargin=cfg.p_margin, pMax=cfg.p_max,
                shadeStart=shade[0] if shade else -1.0, shadeEnd=shade[1] if shade else -1.0,
                failComp=overrides["failure"][0] if "failure" in overrides else -1,  # arızalı kompresör, yoksa -1
                compNames=names,
                compKw=[f.rated_kw for f in cfg.fixed] + [cfg.vsd.rated_kw],  # anma gücü [kW]
                compFlow=[f.flow_m3min for f in cfg.fixed] + [cfg.vsd.flow_max_m3min],  # tam yükte debi [m³/dk]
                ambientC=cfg.ambient_c, tripC=cfg.trip_temp_c,  # motor sıcaklık skalası [°C]
                runs=runs,
            )
            fname = f"{m_key}__{sc_key}.json"
            (args.out / fname).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            index["files"].append(dict(model=m_key, scenario=sc_key, file=fname))
            e0, e1 = base["energy"][-1], agent["energy"][-1]
            print(f"{sc_label:<18} {m_label:<26} enerji {e1:6.0f} kWh ({(e1 - e0) / e0 * 100:+5.1f}%), "
                  f"kalkış {agent['starts'][-1]:3d} (baseline {base['starts'][-1]})")

    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Yazıldı: {args.out}")


if __name__ == "__main__":
    main()
