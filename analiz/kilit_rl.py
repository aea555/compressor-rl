"""Bu script analizde kullanılan bir denemedir.

Kompresöre özel bekleme kilidi, etkisiz hareket maskesi, iki parçalı epsilon ve sık basınç aralıklarıyla
tablo Q-learning eğitir. Sonuçlar ANALIZ_TABLO.md dosyasında yer alır.

    uv run python analiz/kilit_rl.py --name runs/analiz_I_s0 --p ince --eps-adim 3000000
"""
import sys, json, argparse, time
from pathlib import Path
from dataclasses import replace
import numpy as np
P = Path(__file__).resolve().parent.parent  # proje klasörü
sys.path.insert(0, str(P))
from compressor_env import CompressorEnv, EnvConfig
from klasik_rl import P_EDGES, LAYOUTS, current, pressure, to_command, Q_CAPACITY, MAX_STARTS

HOLD = 3
import os
HOLD = int(os.environ.get('HOLD', 3))
P_SETS = {"std": P_EDGES, "ince": [5.8, 6.0, 6.1, 6.2, 6.3, 6.4, 6.6, 6.8, 7.0, 7.3]}
DP_EDGES = {"kompakt": [-0.02, 0.02], "ince": [-0.03, -0.01, 0.01, 0.03]}

class Agent:
    def __init__(self, a):
        self.kilit = a.kilit          # kilit türü "yok", "B" (yalnızca kalkış ve duruşta) ya da "A" (her değişiklikte) olur
        self.lock_steps = round(a.kilit_dk * 6)
        self.bits = a.bit             # True ise kilit bilgisi duruma eklenir
        self.dp_edges = DP_EDGES[a.dp]
        self.p_edges = P_SETS[a.p]
        self.q_edges = LAYOUTS["kompakt"]["q"]
        self.gamma = a.gamma
        self.urgent = a.acil
        self.dims = (len(self.p_edges) + 1, len(self.dp_edges) + 1, 4, 3, 3, 6, 2) + ((2, 2) if self.bits else ())
        n = int(np.prod(self.dims))
        self.q = np.zeros((n, 9)); self.visits = np.zeros((n, 9), dtype=np.int64)
        self.qq = [np.zeros((n, 9)), np.zeros((n, 9))]; self.vv = [np.zeros((n, 9), dtype=np.int64), np.zeros((n, 9), dtype=np.int64)]
        self.reset()

    def reset(self, env=None):
        self.t = 0                    # bölümdeki adım
        self.lock_until = [0, 0]      # K1 ve K2 kilidinin bittiği adım
        self.cmd = None; self.held = 0

    def locked(self, k):
        return self.t < self.lock_until[k]

    def state(self, obs):
        dp = obs[1] * 0.2; q = obs[2] * Q_CAPACITY
        starts = round(max(obs[9], obs[15], obs[20]) * MAX_STARTS)
        k1, k2, vsd = current(obs)
        parts = [int(np.searchsorted(self.p_edges, pressure(obs))), int(np.searchsorted(self.dp_edges, dp)),
                 int(np.searchsorted(self.q_edges, q)), k1, k2, vsd, int(starts >= 5)]
        if self.bits:
            parts += [int(self.locked(0)), int(self.locked(1))]
        return int(np.ravel_multi_index(parts, self.dims))

    def avail(self, obs):
        """Seçilebilir hareketleri döndürür. Etkisiz hareketler ve kilitli kompresörün yasak hareketleri kapalı tutulur."""
        k1, k2, vsd = current(obs)
        m = np.ones(9, dtype=bool)
        m[1] = vsd < 5; m[2] = vsd > 0
        for k, cur in enumerate((k1, k2)):
            for what, target in enumerate((2, 1, 0)):
                a = 3 + 3 * k + what
                if target == cur:
                    m[a] = False
                elif self.kilit != "yok" and self.locked(k):
                    if self.kilit == "A" or (cur == 0) != (target == 0):
                        m[a] = False
        return m

    def act(self, s, m, eps):
        if np.random.rand() < eps:
            return int(np.random.choice(np.flatnonzero(m)))
        ok = m & (self.visits[s] > 0)
        if not ok.any():
            return 0
        row = np.where(ok, self.q[s], -np.inf)
        return int(np.random.choice(np.flatnonzero(row == row.max())))

    def apply(self, a, obs):
        """Hareketi komuta çevirir ve gerekiyorsa kilidi başlatır."""
        cur = current(obs); cmd = to_command(a, obs)
        for k in (0, 1):
            if cmd[k] != cur[k] and (self.kilit_all or (cur[k] == 0) != (cmd[k] == 0)):
                self.lock_until[k] = self.t + self.lock_steps
        return cmd

    @property
    def kilit_all(self):
        return self.kilit == "A"

    # testte bu fonksiyon çağrılır
    def __call__(self, obs):
        urgent = self.urgent and pressure(obs) < 6.2
        if self.cmd is None or self.held >= HOLD or urgent:
            a = self.act(self.state(obs), self.avail(obs), 0.0)
            self.cmd = self.apply(a, obs); self.held = 0
        self.held += 1; self.t += 1
        return self.cmd


def train(a):
    np.random.seed(a.seed)
    env = CompressorEnv(replace(EnvConfig(), w_start=1.0))
    ag = Agent(a)
    obs, _ = env.reset(seed=a.seed)
    s = ag.state(obs); m = ag.avail(obs); act = ag.act(s, m, 1.0)
    step = 0; log = []; t0 = time.time()
    while step < a.timesteps:
        frac = min(step / a.decay_steps, 1.0)
        if a.eps_adim and step < a.eps_adim:      # ilk kısımda epsilon 1'den 0,1'e hızlı iner
            eps = 1.0 - 0.9 * step / a.eps_adim
        elif a.eps_adim:                           # ikinci kısımda epsilon 0,1'den eps_end'e yavaş iner
            eps = 0.1 + min((step - a.eps_adim) / (a.decay_steps - a.eps_adim), 1.0) * (a.eps_end - 0.1)
        else:
            eps = 1.0 + frac * (a.eps_end - 1.0)
        alpha = a.alpha + frac * (a.alpha_end - a.alpha)
        cmd = ag.apply(act, obs)
        r_sum, disc, k = 0.0, 1.0, 0
        while True:
            obs, r, _, done, info = env.step(cmd)
            step += 1; k += 1; ag.t += 1
            r_sum += disc * r; disc *= a.gamma
            if done or k >= HOLD or (ag.urgent and pressure(obs) < 6.2):
                break
        s2 = ag.state(obs); m2 = ag.avail(obs); a2 = ag.act(s2, m2, eps)
        if a.double:
            X = np.random.randint(2); Y = 1 - X
            qX, qY, vX, vY = ag.qq[X], ag.qq[Y], ag.vv[X], ag.vv[Y]
            ok = m2 & (vX[s2] > 0) & (vY[s2] > 0)
            ok1 = m2 & (ag.visits[s2] > 0)
            if ok.any():
                best = np.flatnonzero(ok)[np.argmax(qX[s2][ok])]; nxt = qY[s2, best]
            elif ok1.any():
                nxt = ag.q[s2][ok1].max()
            else:
                nxt = r_sum / (1.0 - disc)
            n = vX[s, act]
            if a.alpha_n:
                alpha = max((n + 1.0) ** -a.alpha_n, a.alpha_taban)
            st = 1.0 if n == 0 else alpha
            qX[s, act] += st * (r_sum + disc * nxt - qX[s, act]); vX[s, act] += 1; ag.visits[s, act] += 1
            ag.q[s, act] = qX[s, act] if vY[s, act] == 0 else 0.5 * (qX[s, act] + qY[s, act])
        else:
            ok = m2 & (ag.visits[s2] > 0)
            nxt = ag.q[s2][ok].max() if ok.any() else r_sum / (1.0 - disc)
            n = ag.visits[s, act]
            if a.alpha_n:
                alpha = max((n + 1.0) ** -a.alpha_n, a.alpha_taban)
            st = 1.0 if n == 0 else alpha
            ag.q[s, act] += st * (r_sum + disc * nxt - ag.q[s, act]); ag.visits[s, act] += 1
        s, m, act = s2, m2, a2
        if done:
            es = info["episode_stats"]; log.append((step, es["energy_kwh"], es["low_p_min"], es["starts"]))
            if len(log) % 1000 == 0:
                L = np.array(log[-1000:])
                print(f"{a.name} adım {step} enerji {L[:,1].mean():.0f} 6alt {L[:,2].mean():.1f} kalkış {L[:,3].mean():.1f} eps {eps:.2f} {time.time()-t0:.0f}s", flush=True)
            obs, _ = env.reset(); ag.reset()
            s = ag.state(obs); m = ag.avail(obs); act = ag.act(s, m, eps)
    return ag, env


def evaluate(ag, n=20, first=12345):
    from evaluate import run_episode
    env = CompressorEnv(replace(EnvConfig(), w_start=1.0))
    rows = []
    for ep in range(n):
        st, _ = run_episode(env, ag, first + ep, None)
        rows.append([st["energy_kwh"], st["low_p_min"], st["margin_min"], st["starts"], st["load_switches"], st["excess_starts"], st["mean_p"], st["return"]])
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kilit", default="yok"); ap.add_argument("--kilit-dk", type=float, default=5.0)
    ap.add_argument("--bit", action="store_true"); ap.add_argument("--dp", default="kompakt")
    ap.add_argument("--acil", action="store_true"); ap.add_argument("--ortalama", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--timesteps", type=int, default=30_000_000); ap.add_argument("--decay-steps", type=int, default=24_000_000)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.2); ap.add_argument("--alpha-end", type=float, default=0.01)
    ap.add_argument("--p", default="std"); ap.add_argument("--eps-adim", type=int, default=0)
    ap.add_argument("--double", action="store_true"); ap.add_argument("--alpha-n", type=float, default=0.0); ap.add_argument("--alpha-taban", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ag, env = train(a)
    np.savez_compressed(f"{a.name}.npz", q=ag.q, visits=ag.visits, args=json.dumps(vars(a)))
    R = evaluate(ag); m = R.mean(0)
    seen = (ag.visits.sum(1) > 0).mean() * 100
    print(f"SONUC {a.name}: E {m[0]:.1f} low {m[1]:.2f} marg {m[2]:.1f} starts {m[3]:.1f} sw {m[4]:.1f} excess {m[5]:.1f} p {m[6]:.2f} R {m[7]:.1f} | durum %{seen:.0f} goruldu ({ag.q.shape[0]})", flush=True)
    np.save(f"{a.name}_test.npy", R)
