"""Bu dosyada BDQ (Branching Dueling Q-Network) ajanı PyTorch ile sıfırdan yazılmıştır.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Ajan her komutun Q değerini tahmin etmeyi öğrenir ve değeri en yüksek komutu seçer. Q değeri, o komut
verilirse bundan sonra toplam ne kadar ödül alınacağını gösterir. Ağda her kompresörün kendi dalı bulunur.
K1 ve K2 dalları 3, VSD dalı 6 değer üretir ve her dal kendi kompresörü için komut seçer. Q değeri iki
parçadan oluşur. V durumun genel olarak ne kadar iyi olduğunu, A ise bir komutun ortalamaya göre ne
kadar iyi olduğunu gösterir. Yaşanan geçişler bir hafızada (replay buffer) tutulur ve eğitimde tekrar
tekrar kullanılır.

Yöntem Tavakoli ve ark.'nın "Action Branching Architectures for Deep RL" (AAAI 2018) makalesinden alınmıştır.

PPO-C, PPO (Proximal Policy Optimization) ile eğitilen üçüncü aşamanın modelidir (runs/yol_C, train_ppo.py).
BDQ bu modelle aynı ortam ayarlarıyla eğitilmiştir.

    uv run python check_bdq.py all                                    # adımları test eder
    uv run python bdq.py --w-start 1.0 --extra-obs --out runs/bdq     # PPO-C ayarlarıyla eğitir
    uv run python bdq.py --w-start 1.0 --extra-obs --reward-scale 0.1 --timesteps 5000000 --out runs/bdq_5m
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Adım 1'de replay buffer (deneyim hafızası) tanımlanır.
# =============================================================================
class ReplayBuffer:
    """Ajanın yaşadığı geçişleri (s, a, r, s', done) saklar.

    Hafıza dolunca başa dönülür ve en eski kaydın üzerine yazılır.
    """

    def __init__(self, capacity: int, obs_dim: int, n_branches: int):
        self.capacity = capacity  # saklanabilecek en fazla geçiş sayısı
        self.ptr = 0  # sonraki kaydın yazılacağı sıra
        self.size = 0  # hafızadaki geçiş sayısı, en fazla capacity

        # geçişlerin parçaları için diziler baştan ayrılır
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s, geçişten önceki durum
        self.actions = np.zeros((capacity, n_branches), dtype=np.int64)  # a, verilen komut, [2, 0, 3] gibi
        self.rewards = np.zeros(capacity, dtype=np.float32)  # r, alınan ödül
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s', geçişten sonraki durum
        self.dones = np.zeros(capacity, dtype=np.float32)  # done, bölüm gerçekten bittiyse 1, yoksa 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.size+=1
        self.ptr+=1
        self.size = min(self.size, self.capacity)
        self.ptr%=self.capacity
        
    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        random_page_nums = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[random_page_nums], device=device),
            "actions": torch.as_tensor(self.actions[random_page_nums], device=device),
            "rewards": torch.as_tensor(self.rewards[random_page_nums], device=device),
            "next_obs": torch.as_tensor(self.next_obs[random_page_nums], device=device),
            "dones": torch.as_tensor(self.dones[random_page_nums], device=device),
        }

    def __len__(self):
        return self.size


# =============================================================================
# Adım 2'de sinir ağı tanımlanır. Ağ ortak bir gövde, bir değer başı ve her kompresör için bir daldan oluşur.
# =============================================================================
class BranchingQNetwork(nn.Module):
    """Ağın yapısı aşağıdaki gibidir.

                          ┌─> value_head ───────────────> V(s)          (1 sayı)
        obs ─> trunk ─────┼─> branch_heads[0] (K1) ────> A_0(s, a)     (3 sayı)
                          ├─> branch_heads[1] (K2) ────> A_1(s, a)     (3 sayı)
                          └─> branch_heads[2] (VSD) ───> A_2(s, a)     (6 sayı)

        Her dalın Q değeri aşağıdaki gibi hesaplanır.

            Q_d(s, a) = V(s) + A_d(s, a) - ortalama_a( A_d(s, a) )
    """

    def __init__(self, obs_dim: int, branch_sizes: list[int], hidden: int = 256, branch_hidden: int = 128):
        super().__init__()
        # bütün dallar ortak gövdeyi paylaşır
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # değer başı durumun genel olarak ne kadar iyi olduğunu tek bir sayıyla verir
        self.value_head = nn.Sequential(nn.Linear(hidden, branch_hidden), nn.ReLU(), nn.Linear(branch_hidden, 1))
        # her kompresörün avantaj dalı komutların ortalamaya göre ne kadar iyi olduğunu verir
        self.branch_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, branch_hidden), nn.ReLU(), nn.Linear(branch_hidden, n))
            for n in branch_sizes
        )

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Bir grup durum için her dalın Q değerlerini [(B, 3), (B, 3), (B, 6)] boyutlarında döndürür.

        obs (B, obs_dim) boyutunda gelir ve B aynı anda işlenen durum sayısını gösterir.

        A'nın ortalaması çıkarılmasaydı V'ye 10 ekleyip A'dan 10 çıkarmak aynı Q'yu verirdi. O zaman ağ
        durumun değerini V'ye mi A'ya mı koyacağını bilemezdi. Ortalama çıkarılınca A'nın ortalaması hep 0
        olur ve durumun genel değeri yalnızca V'de kalır.
        """
        h = self.trunk(obs)  # ortak gövdeye (B, obs_dim) boyutunda girdi verilir, (B, 256) boyutunda çıktı alınır
        v = self.value_head(h)  # durumun genel değeri, (B, 1)
        q_list = []  # dalların Q değerleri
        for head in self.branch_heads:  # sırayla K1, K2 ve VSD dalları
            a = head(h)  # bu dalın avantajları, (B, n), n o kompresörün seçenek sayısı
            # V ile A birleştirilir. mean(dim=1) her satırın ortalamasını alır, keepdim sayesinde boyut (B, 1) kalır.
            q_list.append(v + a - a.mean(dim=1, keepdim=True))
        return q_list


# =============================================================================
# Adım 3 ve 4'te komut seçimi ve öğrenme uygulanır.
# =============================================================================
class BDQAgent:
    def __init__(
        self,
        obs_dim: int,
        branch_sizes: list[int],  # her kompresörün seçenek sayısı, [3, 3, 6] gibi
        gamma: float = 0.995,  # gelecekteki ödüllerin ağırlığı, PPO ile aynı
        lr: float = 1e-4,  # öğrenme hızı
        tau: float = 0.005,  # hedef ağın her adımda online ağa ne kadar yaklaştığı
        device: str = "cpu",
    ):
        self.branch_sizes = branch_sizes
        self.gamma = gamma
        self.tau = tau
        self.device = device
        # online ağ komut seçer ve eğitilir
        self.q_net = BranchingQNetwork(obs_dim, branch_sizes).to(device)
        # Hedef ağ online ağı yavaşça takip eden bir kopyadır. Hedef değer çok oynamasın diye kullanılır.
        self.target_net = BranchingQNetwork(obs_dim, branch_sizes).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        """Tek bir durum için epsilon-greedy ile komut seçer.

        obs (obs_dim,) boyutunda bir numpy dizisi olarak gelir. epsilon rastgele komut seçme olasılığıdır.
        epsilon 1 ise komut hep rastgele, 0 ise hep en iyi bilinen komut seçilir. Fonksiyon np.array([2, 0, 3])
        gibi her dal için bir sayı döndürür.

        Zar her dal için ayrı atıldığı için K1 rastgele seçilirken K2 en iyi komutu alabilir.
        """
        # ağ birden çok durum beklediği için tek durum (obs_dim,) boyutundan (1, obs_dim) boyutuna getirilir
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_list = self.q_net(obs_t)  # [(1, 3), (1, 3), (1, 6)]
        action = []
        for q, n in zip(q_list, self.branch_sizes):  # her dal ve o dalın seçenek sayısı
            if np.random.rand() < epsilon:
                action.append(np.random.randint(n))  # keşif için 0 ile n-1 arasından rastgele bir komut seçilir
            else:
                action.append(int(q.argmax(dim=1)))  # Q değeri en yüksek komut seçilir
        return np.array(action)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Bir grup geçiş için kaybı hesaplar.

        Bütün dallar için tek bir ortak hedef kullanılır ve aşağıdaki gibi hesaplanır.

            y = r + gamma * (1 - done) * ortalama_d( Q_hedef_d(s', a*_d) )
            a*_d = argmax_a Q_online_d(s', a)

        Double DQN'deki gibi komutu online ağ seçer, değerini hedef ağ verir. Her dalın kaybı
        smooth_l1( Q_online_d(s, a_d), y ) ile hesaplanır, sonra dalların ortalaması alınır.

        Bu ortamda bölüm yalnızca süre dolunca biter (truncated). Bu gerçek bir son olmadığı için done=0
        sayılır ve gelecek hesaba katılmaya devam eder. Bu ayrım train() içinde yapılır.
        """
        obs, actions, rewards = batch["obs"], batch["actions"], batch["rewards"]
        next_obs, dones = batch["next_obs"], batch["dones"]

        # 1) Hedef y hesaplanır. Hesap no_grad içinde yapıldığı için hedefe gradyan gitmez.
        with torch.no_grad():
            q_next_online = self.q_net(next_obs)  # s' için online ağın Q değerleri, komut seçmek için
            q_next_target = self.target_net(next_obs)  # s' için hedef ağın Q değerleri, seçilen komutu değerlendirmek için
            next_values = []  # her dal için s'deki en iyi komutun değeri, (B,)
            for q_on, q_tg in zip(q_next_online, q_next_target):
                best_a = q_on.argmax(dim=1, keepdim=True)  # online ağa göre en iyi komut, (B, 1)
                next_values.append(q_tg.gather(1, best_a).squeeze(1))  # bu komutun hedef ağdaki değeri, (B,)
            # ortak hedef için dalların ortalaması alınır
            next_value = torch.stack(next_values, dim=0).mean(dim=0)  # (3, B) boyutu (B,) boyutuna iner
            y = rewards + self.gamma * (1.0 - dones) * next_value

        # 2) Ağın gerçekten verilen komutlar için şu anki tahmini alınır
        q_list = self.q_net(obs)
        losses = []
        for d, q in enumerate(q_list):
            # actions[:, d] her geçişte d. kompresöre verilen komuttur, (B,)
            q_taken = q.gather(1, actions[:, d].unsqueeze(1)).squeeze(1)  # o komutların Q değeri, (B,)
            # 3) Tahmin ile hedef arasındaki fark kayba çevrilir. Huber kaybı (smooth_l1) büyük hatalarda
            # daha yumuşak davranır.
            losses.append(F.smooth_l1_loss(q_taken, y))
        return torch.stack(losses).mean()  # dalların kayıplarının ortalaması alınır

    def update(self, batch: dict) -> float:
        """Bir gradyan adımı atar ve hedef ağı biraz online ağa yaklaştırır. Bu kısım hazır verilmiştir."""
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)  # çok büyük gradyanlar eğitimi bozmasın diye kırpılır
        self.optimizer.step()
        # hedef ağ (1 - tau) * hedef ağ + tau * online ağ olarak güncellenir
        with torch.no_grad():
            for p_t, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                p_t.mul_(1 - self.tau).add_(self.tau * p)
        return loss.item()

    def save(self, path):
        torch.save({"q_net": self.q_net.state_dict(), "branch_sizes": self.branch_sizes}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])


# =============================================================================
# Adım 5'te eğitim döngüsü yer alır. Bu kısım hazır verilmiştir.
# =============================================================================
def train(args):
    from compressor_env import CompressorEnv, EnvConfig

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # varsayılandan farklı ortam ayarları PPO'daki gibi env_config.json'a yazılır
    overrides = {}
    if args.w_start is not None:
        overrides["w_start"] = args.w_start
    if args.extra_obs:
        overrides["extra_obs"] = True
    (args.out / "env_config.json").write_text(json.dumps(overrides, indent=2))
    # hangi klasörün hangi ayarla eğitildiği sonradan anlaşılsın diye bütün eğitim ayarları da kaydedilir
    (args.out / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    env = CompressorEnv(replace(EnvConfig(), **overrides))
    obs_dim = env.observation_space.shape[0]  # 22, extra_obs açıksa 26
    branch_sizes = [int(n) for n in env.action_space.nvec]  # [3, 3, 6]
    agent = BDQAgent(obs_dim, branch_sizes)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, len(branch_sizes))

    obs, _ = env.reset(seed=args.seed)
    ep_return, ep_count, losses = 0.0, 0, []
    for step in range(1, args.timesteps + 1):
        # Epsilon başta 1 olduğu için seçimler tamamen rastgeledir. Epsilon eps_decay_steps boyunca doğrusal
        # olarak eps_end değerine iner.
        epsilon = max(args.eps_end, 1.0 - step / args.eps_decay_steps * (1.0 - args.eps_end))
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # done olarak yalnızca terminated sayılır, sürenin dolması (truncated) sayılmaz.
        # Hafızaya ölçeklenmiş ödül yazılır ve ajan bu ödülden öğrenir. Ekrana yazılan bölüm ödülü
        # ölçeklenmeden toplandığı için PPO ile aynı birimde kalır.
        buffer.add(obs, action, reward * args.reward_scale, next_obs, float(terminated))
        obs = next_obs
        ep_return += reward

        if terminated or truncated:
            ep_count += 1
            s = info["episode_stats"]
            print(
                f"adım {step:>8} | bölüm {ep_count:>4} | ödül {ep_return:>9.1f} | "
                f"özgül güç {s['specific_power']:.2f} | 6 bar altı {s['low_p_min']:.1f} dk | "
                f"eps {epsilon:.2f} | loss {np.mean(losses) if losses else 0:.4f}",
                flush=True,
            )
            obs, _ = env.reset()
            ep_return, losses = 0.0, []

        # hafızada yeterince deneyim birikince her train_freq adımda bir öğrenme adımı atılır
        if len(buffer) >= args.learning_starts and step % args.train_freq == 0:
            losses.append(agent.update(buffer.sample(args.batch_size)))

        # ara modeller PPO'daki gibi ayrı dosyalara kaydedilir
        if step % args.checkpoint_every == 0:
            agent.save(args.out / "checkpoints" / f"bdq_{step}_steps.pt")

    agent.save(args.out / "bdq_model.pt")
    print(f"Kaydedildi: {args.out / 'bdq_model.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)  # toplam eğitim adımı
    ap.add_argument("--buffer-size", type=int, default=200_000)  # hafıza kapasitesi yaklaşık 70 bölüme yeter
    ap.add_argument("--batch-size", type=int, default=256)  # her öğrenme adımında kullanılan geçiş sayısı
    ap.add_argument("--learning-starts", type=int, default=10_000)  # öğrenmeye başlamadan önce toplanacak geçiş sayısı
    ap.add_argument("--train-freq", type=int, default=4)  # kaç ortam adımında bir öğrenme adımı atılacağı
    ap.add_argument("--eps-decay-steps", type=int, default=300_000)  # epsilon'un azaldığı adım sayısı
    ap.add_argument("--eps-end", type=float, default=0.05)  # epsilon'un en düşük değeri
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/bdq"))
    ap.add_argument("--checkpoint-every", type=int, default=250_000)  # kaç adımda bir ara model kaydedileceği
    ap.add_argument("--w-start", type=float, default=None)  # motor kalkışı cezası, PPO-C'de 1.0
    ap.add_argument("--extra-obs", action="store_true")  # gözleme ek bilgiler eklenir, PPO-C'de bu seçenek açıktır
    # Ajan ortamın ödülünü bu sayıyla çarparak öğrenir. Ödüller normalde 0 ile -18 arasındadır ve 0.1 ile
    # çarpılınca Q değerleri daha küçük bir aralıkta kalır. Bu ayar PPO-C'de ödülleri ölçekleyen
    # VecNormalize'ın basit bir karşılığıdır.
    ap.add_argument("--reward-scale", type=float, default=1.0)
    train(ap.parse_args())
