"""Bu dosyada kompresörleri yöneten kuralların sayıları CEM ile aranır.

CEM (Cross-Entropy Method, çapraz entropi yöntemi) rastgele adaylar deneyip en iyilerine doğru daralan bir
arama yöntemidir.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Q-learning her durumda her hareketin ileride ne kadar ödül getireceğini tahmin etmeye çalışır. Hareketler
arasındaki farklar bu tahminlerdeki gürültüden küçük kaldığı için Q-learning ile iyi bir sonuç alınamadı.
CEM'de böyle bir tahmin yapılmaz. Kuralların sayıları rastgele değiştirilerek adaylar üretilir, her aday
birkaç gün çalıştırılır ve en çok ödül toplayanlar seçilir. Bütün adaylar aynı günlerde denendiği için
talepteki rastgelelik karşılaştırmayı bozmaz.

Kurallar baseline'daki kaskad kontrolle aynı yapıdadır. K1 ve K2, basınç bir eşiğin altına inince yüke,
başka bir eşiğin üstüne çıkınca boşa alınır. Bir süre boşta kalan kompresör kapatılır. VSD'nin devrini bir
PI kontrolcü ayarlar. PI kontrolcü hedef basınçla ölçülen basınç arasındaki farka ve bu farkın zamanla
birikmesine bakar. Ajan bu eşikleri, süreleri ve kazançları (PARAMS) öğrenir. Sayıların çoğu günün
saatine ve talebe göre değişebilir ve aşağıdaki formülle hesaplanır.

    sayı = a + b * sin(saat) + c * cos(saat) + d * talep

Ajan her sayı için a, b, c ve d'yi öğrenir. Bu sayede gece ve gündüz farklı hedef basınç kullanılabilir.
--sabit verilirse b, c ve d sıfırda kalır ve yalnızca a öğrenilir.

Öğrenilen bütün sayılar bir listede (theta) tutulur. Adaylar bir normal dağılımdan çekilir. Dağılımın
ortalaması (mu) o ana kadarki en iyi tahmini gösterir. Standart sapması (sigma) ise adayların bu tahminden
ne kadar uzağa düşebileceğini belirler. Her turda --pop kadar aday çekilir ve her biri aynı --days
günde çalıştırılır. Ortalama ödülü en yüksek olan %20'lik kısım seçilir. Yeni mu seçilenlerin ortalaması,
yeni sigma ise standart sapması olur. Seçilenler birbirine yaklaştıkça sigma küçülür ve arama daralır.
Eğitim günleri test günlerinden farklıdır, evaluate.py 12345 seed'inden başlar.

    uv run python klasik_cem.py --out runs/cem                                  # eğitir
    uv run python evaluate.py --episodes 20 --model runs/cem/cem_policy.json   # test eder
"""

import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from compressor_env.env import LOAD, OFF, UNLOAD

# Sütunlarda isim, baseline'daki değer, alt ve üst sınır yazılıdır. Son sütunu True olan sayılar saate ve
# talebe göre değişebilir.
PARAMS = [
    ("sp", 6.9, 6.3, 7.3, True),  # VSD'nin tutmaya çalıştığı basınç [bar]
    ("k1_on", 6.5, 6.15, 7.0, True),  # basınç bunun altına inince K1 yüke alınır [bar]
    ("k1_off", 7.3, 6.4, 7.8, True),  # basınç bunun üstüne çıkınca K1 boşa alınır [bar]
    ("k2_on", 6.3, 6.15, 7.0, True),  # K2 için de aynı kural uygulanır [bar]
    ("k2_off", 7.1, 6.3, 7.8, True),
    ("idle1", 5.0, 0.0, 15.0, True),  # K1 bu kadar boşta kalınca kapatılır [dk]
    ("idle2", 5.0, 0.0, 15.0, True),  # K2 için de aynı kural uygulanır [dk]
    ("kp", 0.6, 0.1, 2.0, False),  # PI'ın anlık farka verdiği ağırlık [1/bar]
    ("ki", 4.0, 0.5, 15.0, False),  # PI'ın biriken farka verdiği ağırlık [0,001/(bar*s)]
]
N_FEAT = 4  # değişebilen sayılarda a, b, c ve d olmak üzere 4 katsayı bulunur
Q_MID, Q_SPAN = 17.0, 8.0  # 9-25 m³/dk arasındaki talep -1 ile 1 arasına ölçeklenir
DEMAND_TAU = 300.0  # talebin yumuşatılma süresi [s], talep anlık dalgalanma eşikleri oynatmasın diye yumuşatılır


def n_theta() -> int:
    """Öğrenilen sayıların toplam adedini döndürür."""
    return sum(N_FEAT if adaptive else 1 for *_, adaptive in PARAMS)


def base_theta() -> np.ndarray:
    """Baseline gibi davranan başlangıç değerlerini döndürür.

    a katsayıları baseline'daki değerleri alır, b, c ve d ise 0 olur.
    """
    out = []
    for _, base, _, _, adaptive in PARAMS:
        out += [base] + [0.0] * (N_FEAT - 1) if adaptive else [base]
    return np.array(out)


def init_sigma() -> np.ndarray:
    """Başlangıçtaki sigma'yı döndürür.

    a için izin verilen aralığın altıda biri, b, c ve d için on ikide biri alınır.
    """
    out = []
    for _, _, lo, hi, adaptive in PARAMS:
        out += [(hi - lo) / 6] + [(hi - lo) / 12] * (N_FEAT - 1) if adaptive else [(hi - lo) / 6]
    return np.array(out)


class CemPolicy:
    """Gözleme bakıp kompresörlere komut verir."""

    def __init__(self, theta: np.ndarray):
        self.theta = np.asarray(theta, dtype=float)  # öğrenilen bütün sayılar

    def params(self, hour: float, demand: float) -> dict:
        """Verilen saat ve talep için eşikleri, süreleri ve kazançları hesaplar. Sınırı aşanları sınıra çeker."""
        h = hour * 2 * np.pi / 24.0
        feat = np.array([1.0, np.sin(h), np.cos(h), np.clip((demand - Q_MID) / Q_SPAN, -1.0, 1.0)])
        out, i = {}, 0
        for name, _, lo, hi, adaptive in PARAMS:
            n = N_FEAT if adaptive else 1
            out[name] = float(np.clip(self.theta[i:i + n] @ feat[:n], lo, hi))
            i += n
        # boşa alma eşiği yüke alma eşiğinin en az 0,1 bar üstünde tutulur, aksi halde kompresör sürekli gidip gelir
        out["k1_off"] = max(out["k1_off"], out["k1_on"] + 0.1)
        out["k2_off"] = max(out["k2_off"], out["k2_on"] + 0.1)
        return out

    def reset(self, env):
        """Her bölümün başında çağrılır. Ortamdan yalnızca adım süresini ve VSD kademelerini alır."""
        cfg = env.cfg
        self.dt = cfg.dt  # adım süresi [s]
        self.min_speed = cfg.vsd.min_speed  # VSD'nin en düşük devri (0-1)
        self.levels = np.array([env.vsd_speed(l) for l in range(1, cfg.vsd.n_levels + 1)])  # kademelerin devri (0-1)
        self.idle = [0.0, 0.0]  # K1 ve K2'nin ne zamandır boşta olduğu [s]
        self.speed_cmd = None  # PI'ın istediği VSD devri (0-1), ilk adımda gözlemden alınır
        self.e_prev = None  # önceki adımdaki basınç farkı [bar]
        self.vsd_on = None  # VSD açık mı
        self.demand = None  # yumuşatılmış talep [m³/dk]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        p = float(obs[0]) + 7.0  # basınç [bar]
        hour = (np.arctan2(obs[3], obs[4]) * 24.0 / (2 * np.pi)) % 24.0  # günün saati
        q = float(obs[2]) * 30.0  # talep [m³/dk]
        states = [int(np.argmax(obs[5:8])), int(np.argmax(obs[11:14]))]  # K1 ve K2, 0 kapalı, 1 boşta, 2 yükte
        if self.speed_cmd is None:  # bölümün ilk adımında başlangıç değerleri gözlemden alınır
            self.vsd_on = obs[17] > 0.5
            self.speed_cmd = float(obs[18])
            self.demand = q
        self.demand += (q - self.demand) * min(self.dt / DEMAND_TAU, 1.0)
        th = self.params(hour, self.demand)

        action = []
        for i, (p_on, p_off, idle_min) in enumerate(((th["k1_on"], th["k1_off"], th["idle1"]),
                                                    (th["k2_on"], th["k2_off"], th["idle2"]))):
            s = states[i]
            if p < p_on:
                cmd = LOAD  # basınç düşük olduğu için yüke alınır
            elif p > p_off and s != OFF:
                cmd = UNLOAD  # basınç yüksek olduğu için boşa alınır
            else:
                cmd = s  # basınç iki eşiğin arasında olduğu için durum değişmez
            if cmd == UNLOAD:
                self.idle[i] += self.dt
                if self.idle[i] >= idle_min * 60.0:
                    cmd = OFF  # yeterince boşta kalan kompresör kapatılır
            else:
                self.idle[i] = 0.0
            action.append(cmd)

        # VSD devrini baseline'daki gibi PI kontrolcü ayarlar
        e = th["sp"] - p
        if self.e_prev is None:
            self.e_prev = e
        self.speed_cmd += th["kp"] * (e - self.e_prev) + th["ki"] / 1000.0 * e * self.dt
        self.speed_cmd = float(np.clip(self.speed_cmd, 0.0, 1.0))
        self.e_prev = e
        if self.vsd_on and self.speed_cmd < self.min_speed * 0.5 and p > th["sp"] + 0.25:
            self.vsd_on = False
        elif not self.vsd_on and p < th["sp"]:
            self.vsd_on = True
            self.speed_cmd = max(self.speed_cmd, self.min_speed)
        level = int(np.argmin(np.abs(self.levels - self.speed_cmd))) + 1 if self.vsd_on else 0
        action.append(level)
        return np.asarray(action, dtype=np.int64)

    def describe(self, hours=(2, 10, 18), demands=(8, 17, 25)) -> str:
        """Öğrenilen sayıları birkaç saat ve talep değeri için tablo halinde yazar."""
        lines = [f"{'saat':>5} {'talep':>6} " + " ".join(f"{n:>7}" for n, *_ in PARAMS)]
        for h in hours:
            for d in demands:
                th = self.params(h, d)
                lines.append(f"{h:>5} {d:>6} " + " ".join(f"{th[n]:>7.2f}" for n, *_ in PARAMS))
        return "\n".join(lines)

    def save(self, path: Path):
        path.write_text(json.dumps({"theta": self.theta.tolist()}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CemPolicy":
        return cls(np.array(json.loads(Path(path).read_text())["theta"]))


_CFG = None  # paralel süreçlerin kullandığı ortam ayarları, bu ayarları ana süreç başlatırken verir


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _episode(job) -> float:
    """Bir adayı bir gün boyunca çalıştırıp toplam ödülü döndürür."""
    from compressor_env import CompressorEnv

    theta, seed = job
    env = CompressorEnv(_CFG)
    policy = CemPolicy(theta)
    obs, _ = env.reset(seed=int(seed))
    policy.reset(env)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += r
        done = term or trunc
    return total


def train(args):
    from compressor_env import EnvConfig

    args.out.mkdir(parents=True, exist_ok=True)
    overrides = {"w_start": args.w_start}  # evaluate.py modeli bu ayarla çalıştırır
    (args.out / "env_config.json").write_text(json.dumps(overrides, indent=2))
    (args.out / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    cfg = replace(EnvConfig(), **overrides)

    rng = np.random.default_rng(args.seed)
    mu, sigma = base_theta(), init_sigma()
    if args.sabit:  # --sabit verilince b, c ve d sıfırda tutulur
        fixed = np.ones_like(mu, dtype=bool)
        i = 0
        for *_, adaptive in PARAMS:
            fixed[i] = False
            i += N_FEAT if adaptive else 1
        sigma[fixed] = 0.0
    sigma_min = sigma / 50  # sigma bunun altına inmez, aksi halde arama erken durur
    n_elite = max(2, int(args.pop * args.elite))
    log = []
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for it in range(args.iters):
            seeds = rng.integers(1_000_000, 2_000_000, args.days)  # bu turun eğitim günleri
            cand = mu + sigma * rng.standard_normal((args.pop, len(mu)))
            cand[0] = mu  # şu anki mu da aday olarak denenir
            jobs = [(c, s) for c in cand for s in seeds]
            ret = np.array(pool.map(_episode, jobs, chunksize=8)).reshape(args.pop, args.days).mean(axis=1)
            elite = cand[np.argsort(-ret)[:n_elite]]
            mu = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), sigma_min)
            log.append(dict(iter=it, best=float(ret.max()), mean=float(ret.mean()), mu_return=float(ret[0])))
            print(f"tur {it:>3} | en iyi {ret.max():8.1f} | ortalama {ret.mean():8.1f} | önceki mu {ret[0]:8.1f} | "
                  f"{time.time() - t0:5.0f} sn", flush=True)

    policy = CemPolicy(mu)
    policy.save(args.out / "cem_policy.json")
    (args.out / "egitim.json").write_text(json.dumps(log))
    print(f"Kaydedildi: {args.out / 'cem_policy.json'}\n\nÖğrenilen sayılar (saat, talep [m³/dk]):\n{policy.describe()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)  # her turda denenen aday sayısı
    ap.add_argument("--days", type=int, default=16)  # her adayın kaç günde denendiği
    ap.add_argument("--iters", type=int, default=30)  # tur sayısı
    ap.add_argument("--elite", type=float, default=0.2)  # her turda adayların ne kadarının seçildiği
    ap.add_argument("--w-start", type=float, default=3.0)  # motor kalkışı cezası, PPO w=3 ile aynı
    ap.add_argument("--sabit", action="store_true")  # verilirse sayılar saate ve talebe göre değişmez
    ap.add_argument("--workers", type=int, default=10)  # paralel süreç sayısı, çekirdek sayısı kadar seçilir
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/cem"))
    train(ap.parse_args())
