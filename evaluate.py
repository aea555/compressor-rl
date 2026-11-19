"""Baseline ile eğitilmiş ajanları aynı günlerde çalıştırıp karşılaştırır.

Baseline, sabit basınç eşiklerine göre çalışan klasik kaskad kontroldür (compressor_env/baseline.py).
Modelin türü dosya uzantısından anlaşılır. .zip uzantısı PPO (Proximal Policy Optimization, train_ppo.py),
.pt BDQ (bdq.py), .npz tablo ya da tile coding ile Q-learning (klasik_rl.py), .json ise CEM (çapraz entropi
yöntemi, klasik_cem.py) modelini gösterir.

    uv run python evaluate.py                                  # yalnızca baseline'ı çalıştırır
    uv run python evaluate.py --model runs/ppo/best_model.zip  # baseline ile PPO'yu karşılaştırır
    uv run python evaluate.py --model runs/yol_A/final_model.zip runs/yol_B/final_model.zip  # birden fazla modeli karşılaştırır
    uv run python evaluate.py --model runs/yol_C/final_model.zip runs/bdq/bdq_model.pt     # PPO ile BDQ'yu karşılaştırır
    uv run python evaluate.py --plot                           # ilk bölümün grafiğini çizer

Her model, klasöründeki env_config.json'da yazan ayarlarla çalıştırılır. Fizik bütün modellerde
aynı olduğu için enerji, basınç ve kalkış sayısı doğrudan karşılaştırılabilir. Toplam ödül her
modelin kendi ödül ayarlarıyla hesaplandığı için modeller arasında karşılaştırılamaz.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compressor_env import CascadeController, CompressorEnv, EnvConfig

# Her satırda episode_stats anahtarı, tablodaki sütun başlığı ve sayı biçimi yazılıdır.
METRICS = [
    ("specific_power", "Özgül güç [kW/(m³/dk)]", "{:.3f}"),
    ("energy_kwh", "Enerji [kWh]", "{:.1f}"),
    ("mean_p", "Ort. basınç [bar]", "{:.2f}"),
    ("low_p_min", "6 bar altı [dk]", "{:.1f}"),
    ("margin_min", "6,2 bar altı [dk]", "{:.1f}"),
    ("starts", "Motor kalkışı", "{:.1f}"),
    ("load_switches", "Yük/boş geçişi", "{:.1f}"),
    ("excess_starts", "Limit aşan kalkış", "{:.1f}"),
    ("trips", "Termik trip", "{:.1f}"),
]


def run_episode(env: CompressorEnv, policy, seed: int, start_hour: float | None):
    """Bir vardiyayı baştan sona çalıştırır. Özet metrikleri ve adım adım kaydı döndürür.

    policy gözlemi alıp komut döndüren bir fonksiyondur ve baseline ya da eğitilmiş bir ajan olabilir.
    Aynı seed verilince talep de aynı olduğu için kontrolcüler aynı gün üzerinde karşılaştırılır.
    """
    options = {} if start_hour is None else {"start_hour": start_hour}
    obs, _ = env.reset(seed=seed, options=options)
    if hasattr(policy, "reset"):
        policy.reset(env)
    log = []  # her adımın info sözlüğü, grafik için
    total_r = 0.0  # bölüm boyunca toplanan ödül
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total_r += r
        log.append(info)
        done = term or trunc
    stats = info["episode_stats"] | {"return": total_r}
    return stats, log


def load_env_config(model_path: Path) -> EnvConfig:
    """Modelin eğitildiği ortam ayarlarını okur.

    train_ppo.py varsayılandan farklı ayarları env_config.json'a yazar. Ara kopyalar checkpoints/
    alt klasöründe durduğu için bir üst klasöre de bakılır. Eski modellerde bu dosya bulunmadığından
    varsayılan ayarlar kullanılır.
    """
    for folder in (model_path.parent, model_path.parent.parent):
        f = folder / "env_config.json"
        if f.exists():
            return replace(EnvConfig(), **json.loads(f.read_text()))
    return EnvConfig()


# Raporda kullanılan modellerin tablolarda ve grafiklerde görünecek adları burada verilir. Anahtar olarak
# eğitim klasörünün adı kullanılır.
LABELS = {
    "pareto_w3": "PPO w=3",
    "yol_C": "PPO w=1",
    "pareto_w0.3": "PPO w=0,3",
    "pareto_w10": "PPO w=10",
    "bdq_5m": "BDQ",
    "klasik_q_kompakt": "Q tablo",
    "tile_qlearning_s0": "Q tile",
    "cem_sabit": "CEM",
}


def model_label(model_path: Path) -> str:
    """Tabloda görünecek 'PPO w=3' ya da 'yol_A/ppo_500000_steps' gibi kısa ismi döndürür.

    Ara kopyalar ve LABELS'ta bulunmayan modeller klasör ve dosya adıyla yazılır.
    """
    if model_path.parent.name == "checkpoints":
        return f"{model_path.parent.parent.name}/{model_path.stem}"
    return LABELS.get(model_path.parent.name, f"{model_path.parent.name}/{model_path.stem}")


def make_policies(model_paths: list[Path]):
    """Karşılaştırılacak kontrolcüleri hazırlar. Her isim için komut üreten fonksiyonu ve ortamı döndürür."""
    base_env = CompressorEnv()
    cascade = CascadeController()

    # Baseline karar verirken ortamın kendisini okur. Dayanıklılık testlerinde her senaryonun ortamı
    # farklı olduğu için baseline hangi ortamda çalıştığını reset sırasında öğrenir.
    def cascade_policy(obs):
        return cascade.act(cascade_policy.env)

    def cascade_reset(env):
        cascade_policy.env = env
        cascade.reset(env)

    cascade_policy.reset = cascade_reset
    policies = {"Kaskad (baseline)": (cascade_policy, base_env)}

    for path in model_paths:
        env = CompressorEnv(load_env_config(path))
        if path.suffix == ".json":  # CEM modelini klasik_cem.py kaydeder
            from klasik_cem import CemPolicy

            policy = CemPolicy.load(path)
        elif path.suffix == ".npz":  # tablo ya da tile coding ajanını klasik_rl.py kaydeder
            from klasik_rl import load_agent

            policy = load_agent(path)  # ajan doğrudan fonksiyon gibi çağrılabilir
        elif path.suffix == ".pt":  # BDQ modelini bdq.py kaydeder
            from bdq import BDQAgent

            agent = BDQAgent(env.observation_space.shape[0], [int(n) for n in env.action_space.nvec])
            agent.load(path)
            # testte keşif yapılmaz, epsilon=0 ile hep en iyi bilinen komut seçilir
            policy = lambda obs, agent=agent: agent.act(obs, epsilon=0.0)
        else:  # PPO modelini Stable-Baselines3 .zip olarak kaydeder
            from stable_baselines3 import PPO

            model = PPO.load(path, device="cpu")
            # deterministic=True verildiğinde rastgele seçim yapılmaz ve en olası komut alınır
            policy = lambda obs, model=model: model.predict(obs, deterministic=True)[0]
        # agent=agent ve model=model yazılmazsa bütün lambda'lar döngüdeki son modeli kullanır
        policies[model_label(path)] = (policy, env)
    return policies


def plot_episode(logs: dict, env: CompressorEnv, path: Path, shade=None, title=None):
    """Bir bölümün grafiğini çizer. Grafikte basınç, talep ile üretim, güç ve her kontrolcünün kompresör
    durumları yer alır.

    logs her kontrolcü adı için adım adım info listesini tutar. shade başlangıç ve bitiş saati olarak
    verilirse o aralık gri boyanır, arıza süresi bu yolla gösterilir. title grafiğin başlığı olur.
    """
    cfg = env.cfg
    names = [f.name for f in cfg.fixed] + [cfg.vsd.name]
    fig, axes = plt.subplots(3 + len(logs), 1, figsize=(12, 3 + 2.2 * (3 + len(logs))), sharex=True)
    first = next(iter(logs.values()))
    hours = np.arange(len(first)) * cfg.dt / 3600.0

    ax = axes[0]
    for label, log in logs.items():
        ax.plot(hours, [x["pressure"] for x in log], label=label, lw=0.9)
    ax.axhline(cfg.p_min, color="red", ls="--", lw=0.8, label="6 bar sınırı")
    ax.set_ylabel("Basınç [bar]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(hours, [x["demand"] for x in first], color="black", lw=0.9, label="Talep")
    for label, log in logs.items():
        ax.plot(hours, [x["supply"] for x in log], lw=0.7, alpha=0.8, label=f"Üretim ({label})")
    ax.set_ylabel("Debi [m³/dk]")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    for label, log in logs.items():
        ax.plot(hours, [x["power_kw"] for x in log], lw=0.8, label=label)
    ax.set_ylabel("Güç [kW]")
    ax.legend(loc="upper right", fontsize=8)

    for ax, (label, log) in zip(axes[3:], logs.items()):
        states = np.array([x["state"][: env.n_fixed] for x in log]).T
        for k in range(env.n_fixed):
            ax.plot(hours, states[k] + 3 * k, lw=0.9, label=names[k])
        ax.plot(hours, np.array([x["vsd_speed"] for x in log]) * 2 + 3 * env.n_fixed, lw=0.9, label=names[-1])
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("Bölüm süresi [saat]")
    if shade is not None:
        for ax in axes:
            ax.axvspan(*shade, color="gray", alpha=0.15)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Grafik: {path}")


def main():
    ap = argparse.ArgumentParser()
    # birden fazla eğitilmiş model verilebilir. Hiç model verilmezse yalnızca baseline çalışır.
    ap.add_argument("--model", type=Path, nargs="*", default=[])
    ap.add_argument("--episodes", type=int, default=10)  # ortalaması alınacak gün sayısı
    ap.add_argument("--seed", type=int, default=12345)  # test günlerinin seed'i
    ap.add_argument("--start-hour", type=float, default=None, help="Sabit başlangıç saati, örneğin 6")
    ap.add_argument("--plot", action="store_true")  # verilirse ilk bölümün grafiği çizilir
    ap.add_argument("--plot-path", type=Path, default=Path("episode.png"))  # grafik dosyası
    args = ap.parse_args()

    policies = make_policies(args.model)
    results = {name: [] for name in policies}  # her kontrolcünün bölüm metrikleri
    first_logs = {}  # grafik için ilk bölümün adım adım kaydı
    for ep in range(args.episodes):
        for name, (policy, env) in policies.items():
            stats, log = run_episode(env, policy, args.seed + ep, args.start_hour)
            results[name].append(stats)
            if ep == 0:
                first_logs[name] = log

    env = policies["Kaskad (baseline)"][1]  # fizik herkes için aynı olduğu için başlık ve grafikte baseline ortamı kullanılır
    col_w = max([20] + [len(n) + 2 for n in policies])  # sütun genişliği en uzun isme göre ayarlanır
    print(f"\n{args.episodes} bölümün ortalaması, her bölüm {env.cfg.episode_hours:.0f} saat")
    print(f"{'Metrik':<30}" + "".join(f"{n:>{col_w}}" for n in policies))
    for key, label, fmt in METRICS + [("return", "Toplam ödül (kendi ayarı)", "{:.1f}")]:
        row = f"{label:<30}"
        for name in policies:
            row += f"{fmt.format(np.mean([s[key] for s in results[name]])):>{col_w}}"
        print(row)

    if args.plot:
        plot_episode(first_logs, env, args.plot_path)


if __name__ == "__main__":
    main()
