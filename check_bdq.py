"""bdq.py'deki adımları tek tek test eder. BDQ yöntemi bdq.py'nin başında anlatılır.

    uv run python check_bdq.py 1     # adım 1'i (replay buffer) test eder
    uv run python check_bdq.py 2     # adım 2'yi (sinir ağı) test eder
    uv run python check_bdq.py 3     # adım 3'ü (komut seçme) test eder
    uv run python check_bdq.py 4     # adım 4'ü (kayıp hesabı) test eder
    uv run python check_bdq.py all   # bütün adımları sırayla test eder
"""

import sys

import numpy as np
import torch

from bdq import BDQAgent, BranchingQNetwork, ReplayBuffer

OBS_DIM, BRANCHES = 22, [3, 3, 6]


def ok(msg):
    print(f"  ✓ {msg}")


def check_1():
    print("Adım 1: ReplayBuffer")
    buf = ReplayBuffer(capacity=5, obs_dim=OBS_DIM, n_branches=3)
    for i in range(3):
        buf.add(np.full(OBS_DIM, i, np.float32), np.array([i % 3, 0, i]), float(i), np.full(OBS_DIM, i + 1, np.float32), 0.0)
    assert len(buf) == 3, f"3 ekleme sonrası size 3 olmalı, {len(buf)} bulundu"
    assert buf.rewards[2] == 2.0 and buf.actions[2, 2] == 2, "Kayıtlar doğru sıraya yazılmamış"
    ok("ekleme ve size doğru")

    for i in range(3, 7):
        buf.add(np.full(OBS_DIM, i, np.float32), np.array([0, 0, 0]), float(i), np.zeros(OBS_DIM, np.float32), 0.0)
    assert len(buf) == 5, f"Kapasite 5 iken size 5'i geçmemeli, {len(buf)} bulundu"
    assert buf.rewards[0] == 5.0 and buf.rewards[1] == 6.0, "Hafıza dolunca ptr başa dönmeli ve en eski kaydın üzerine yazmalı"
    ok("dolunca başa dönme doğru")

    batch = buf.sample(4)
    expected = {"obs": (4, OBS_DIM), "actions": (4, 3), "rewards": (4,), "next_obs": (4, OBS_DIM), "dones": (4,)}
    for k, shape in expected.items():
        assert k in batch, f"sample() sonucunda '{k}' anahtarı yok"
        assert isinstance(batch[k], torch.Tensor), f"'{k}' bir torch.Tensor olmalı"
        assert tuple(batch[k].shape) == shape, f"'{k}' boyutu {shape} olmalı, {tuple(batch[k].shape)} bulundu"
    assert batch["actions"].dtype == torch.int64, "actions int64 olmalı, gather bunu istiyor"
    ok("sample() boyutları ve tipleri doğru")

    small = ReplayBuffer(capacity=100, obs_dim=OBS_DIM, n_branches=3)
    small.add(np.ones(OBS_DIM, np.float32), np.array([1, 1, 1]), 7.0, np.ones(OBS_DIM, np.float32), 0.0)
    assert torch.all(small.sample(8)["rewards"] == 7.0), "Örnekler sadece dolu kısımdan, yani self.size içinden seçilmeli"
    ok("sadece dolu kısımdan örnekleniyor")


def check_2():
    print("Adım 2: BranchingQNetwork")
    torch.manual_seed(0)
    net = BranchingQNetwork(OBS_DIM, BRANCHES)
    x = torch.randn(8, OBS_DIM)
    qs = net(x)
    assert isinstance(qs, list) and len(qs) == 3, "forward() 3 elemanlı bir liste döndürmeli"
    for q, n in zip(qs, BRANCHES):
        assert tuple(q.shape) == (8, n), f"Dal boyutu (8, {n}) olmalı, {tuple(q.shape)} bulundu"
    ok("çıktı boyutları doğru")

    v = net.value_head(net.trunk(x))
    for d, q in enumerate(qs):
        assert torch.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5), (
            f"Dal {d}: Q'ların ortalaması V'ye eşit olmalı. A'nın ortalamasını çıkardınız mı?"
        )
    ok("V ve A birleşimi doğru, Q'ların ortalaması V'ye eşit")


def check_3():
    print("Adım 3: act()")
    torch.manual_seed(0)
    agent = BDQAgent(OBS_DIM, BRANCHES)
    obs = np.random.randn(OBS_DIM).astype(np.float32)

    a = agent.act(obs, epsilon=0.0)
    assert isinstance(a, np.ndarray) and a.shape == (3,), "act() (3,) boyutunda bir numpy dizisi döndürmeli"
    with torch.no_grad():
        qs = agent.q_net(torch.as_tensor(obs).unsqueeze(0))
    expected = [int(q.argmax(dim=1)) for q in qs]
    assert list(a) == expected, f"epsilon=0 iken her dalın argmax'ı seçilmeli: beklenen {expected}, bulunan {list(a)}"
    ok("epsilon=0 iken en iyi komut seçiliyor")

    acts = np.array([agent.act(obs, epsilon=1.0) for _ in range(600)])
    for d, n in enumerate(BRANCHES):
        seen = set(acts[:, d].tolist())
        assert seen == set(range(n)), f"epsilon=1 iken dal {d} için 0 ile {n - 1} arasındaki bütün seçenekler görülmeli, görülen {seen}"
    ok("epsilon=1 iken her dal geçerli aralıkta rastgele seçiyor")


def check_4():
    print("Adım 4: compute_loss()")
    torch.manual_seed(0)
    agent = BDQAgent(OBS_DIM, BRANCHES, gamma=0.9)
    B = 16
    batch = {
        "obs": torch.randn(B, OBS_DIM),
        "actions": torch.stack([torch.randint(0, n, (B,)) for n in BRANCHES], dim=1),
        "rewards": torch.randn(B),
        "next_obs": torch.randn(B, OBS_DIM),
        "dones": torch.ones(B),
    }
    loss = agent.compute_loss(batch)
    assert loss.dim() == 0, "Kayıp tek bir sayı, yani skaler tensör olmalı"

    # done=1 iken hedef yalnızca ödülden oluşur. Kayıp elle hesaplanır ve karşılaştırılır.
    with torch.no_grad():
        qs = agent.q_net(batch["obs"])
        manual = torch.stack([
            torch.nn.functional.smooth_l1_loss(q.gather(1, batch["actions"][:, d : d + 1]).squeeze(1), batch["rewards"])
            for d, q in enumerate(qs)
        ]).mean()
    assert torch.allclose(loss, manual, atol=1e-5), f"done=1 iken kayıp {manual.item():.5f} olmalı, {loss.item():.5f} bulundu"
    ok("done=1 iken kayıp doğru, hedef r")

    loss.backward()
    assert all(p.grad is not None for p in agent.q_net.parameters()), "Gradyan online ağın tüm parametrelerine ulaşmalı"
    assert all(p.grad is None for p in agent.target_net.parameters()), "Hedef ağa gradyan gitmemeli, torch.no_grad kullanıldı mı?"
    ok("gradyan sadece online ağa gidiyor")

    batch["dones"] = torch.zeros(B)
    l1 = agent.compute_loss(batch).item()
    with torch.no_grad():
        for p in agent.target_net.parameters():
            p.add_(0.5)
    l2 = agent.compute_loss(batch).item()
    assert abs(l1 - l2) > 1e-6, "done=0 iken hedef, hedef ağın s' için tahminine bağlı olmalı"
    ok("done=0 iken hedef ağ kullanılıyor")


if __name__ == "__main__":
    steps = {"1": check_1, "2": check_2, "3": check_3, "4": check_4}
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for key in steps if which == "all" else [which]:
        try:
            steps[key]()
            print(f"Adım {key} geçti!\n")
        except NotImplementedError as e:
            print(f"  … {e} henüz yazılmadı.\n")
            break
        except AssertionError as e:
            print(f"  ✗ {e}\n")
            break
