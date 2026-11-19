"""Bu dosyada Q-learning ve SARSA tablo yerine tile coding ile uygulanır (Sutton ve Barto, bölüm 9.5 ve 10.1).

Tablo kullanan sürümde (klasik_rl.py) her durumun ayrı bir hücresi bulunur ve bir hücrede öğrenilen şey
komşusuna geçmez. Tile coding'de gözlem, birbirine göre biraz kaydırılmış birkaç ızgaraya yerleştirilir.
Durum her ızgarada bir kutuya düşer. Her kutunun her hareket için bir ağırlığı vardır. Bir hareketin değeri,
durumun düştüğü kutulardaki ağırlıklar toplanarak bulunur. Yakın durumlar kutularının çoğunu paylaştığı
için birinde öğrenilen diğerine de geçer.

Bütün bilgiler tek bir ızgaraya konsaydı ağırlık sayısı çok büyürdü. Bu nedenle bilgiler gruplara ayrılmıştır
(GROUPS) ve her grup kendi ızgaralarını kurar.

Sistemde üç kompresör bulunur. K1 ve K2 sabit hızda çalışır ve kapalı, boşta ya da yükte olabilir. VSD
(değişken hızlı sürücü) kompresörü kapalı ya da 5 devir kademesinden birinde olabilir. Bu 3 * 3 * 6 = 54
kombinasyon zaten sayılabilir olduğu için ızgaraya girmez. Kompresör durumunu kullanan gruplarda her
kombinasyonun ayrı kutuları vardır.

--extra-obs verilirse ajan K1 ve K2'nin kaç dakikadır aynı durumda olduğunu da görür (EXTRA_GROUP). İlk
tile coding ajanı kompresörleri saatlerce boşta tutmuştur. Baseline'ın "5 dakika boşta kalırsa kapat" kuralı
bu bilgiye dayanır ama ilk ajanda bu bilgi bulunmadığı için ajan boştaki bir kompresörün az önce mi yoksa
yarım saattir mi boşta olduğunu ayırt edememiştir.

Hareketler, 30 saniyelik karar süresi ve eğitim döngüsü klasik_rl.py ile aynıdır.

    uv run python klasik_rl.py --tablo tile --method sarsa --gamma 0.99 --alpha 0.1 --out runs/tile_sarsa
    uv run python klasik_rl.py --tablo tile --extra-obs --method sarsa --gamma 0.99 --alpha 0.1 --out runs/tile2_sarsa
"""

import numpy as np

from klasik_rl import N_ACTIONS, HoldPolicy, current, pressure

N_TILINGS = 8  # her gruptaki kaydırılmış ızgara sayısı

# Her bilginin alt sınırı, kutu genişliği ve bir ızgaradaki kutu sayısı sırayla verilir.
VARS = {
    "p": (5.6, 0.2, 12),  # basınç [bar], 5,6 ile 8,0 arası
    "dp": (-0.08, 0.02, 8),  # basıncın 10 saniyedeki değişimi [bar]
    "q": (0.0, 4.0, 8),  # talep [m³/dk], 0 ile 32 arası
    "hour": (0.0, 3.0, 8),  # günün saati, 3 saatlik kutular
    "starts": (0.0, 2.0, 4),  # en çok kalkış yapan motorun son bir saatteki kalkışı
    "t1": (0.0, 3.0, 10),  # K1'in kaç dakikadır aynı durumda olduğu, 30 dakikaya kadar
    "t2": (0.0, 3.0, 10),  # K2'nin kaç dakikadır aynı durumda olduğu
}

# Her grupta birlikte ızgaraya giren bilgiler yazılıdır. Yanında True olan gruplar kompresör durumunu da kullanır.
GROUPS = [
    (("p", "dp"), True),  # şu anki kompresörlerle basıncın ne yöne gittiğini gösterir
    (("p", "q"), True),  # bu talep ve basınçta şu anki kompresörlerin yetip yetmediğini gösterir
    (("q", "hour"), False),  # günün hangi saatinde ne kadar talep olduğunu gösterir
    (("starts",), True),  # saatlik kalkış limitine ne kadar yaklaşıldığını gösterir
]
EXTRA_GROUP = (("t1", "t2"), True)  # --extra-obs ile eklenen grup kompresörlerin kaç dakikadır bu durumda olduğunu gösterir
N_CONFIGS = 3 * 3 * 6  # kompresör durumlarının kombinasyon sayısı


def groups(extra: bool) -> list:
    """Kullanılan grupları döndürür. extra verilirse kompresörlerin aynı durumda geçirdiği süre de eklenir."""
    return GROUPS + [EXTRA_GROUP] if extra else GROUPS


def group_starts(extra: bool) -> tuple[np.ndarray, int]:
    """Her grubun ağırlık dizisinde nereden başladığını ve sondaki sabitle birlikte toplam ağırlık sayısını döndürür."""
    sizes = []
    for names, use_config in groups(extra):
        per_tiling = int(np.prod([VARS[n][2] + 1 for n in names])) * (N_CONFIGS if use_config else 1)
        sizes.append(per_tiling * N_TILINGS)
    return np.concatenate([[0], np.cumsum(sizes)[:-1]]), int(sum(sizes)) + 1
# Her ızgaranın her boyutta kutu genişliğinin ne kadarı kadar kaydırıldığı burada tutulur. Kitaptaki öneriye
# göre boyutlar farklı adımlarla (1, 3, 5, ...) kaydırıldığı için ızgaralar köşegen boyunca üst üste binmez.
OFFSETS = np.array([[(k * (2 * d + 1) / N_TILINGS) % 1.0 for d in range(2)] for k in range(N_TILINGS)])


def features(obs: np.ndarray, extra: bool = False) -> np.ndarray:
    """Gözlemi açık olan kutuların numaralarına çevirir. Her grupta N_TILINGS kutu açılır, ayrıca bir sabit hep açıktır."""
    hour = (np.arctan2(obs[3], obs[4]) * 24.0 / (2 * np.pi)) % 24.0  # gözlemde saat sin ve cos olarak bulunur
    values = {
        "p": pressure(obs),
        "dp": obs[1] * 0.2,
        "q": obs[2] * 30.0,
        "hour": hour,
        "starts": max(obs[9], obs[15], obs[20]) * 6.0,
    }
    if extra:  # gözlemin sonunda her kompresörün aynı durumda geçirdiği süre bulunur, 1 değeri 30 dakikaya karşılık gelir
        values["t1"], values["t2"] = obs[23] * 30.0, obs[24] * 30.0
    starts, n_features = group_starts(extra)
    k1, k2, vsd = current(obs)
    config = (k1 * 3 + k2) * 6 + vsd  # kompresör durumu, 0-53
    out = []
    for (names, use_config), start in zip(groups(extra), starts):
        dims = [VARS[n][2] + 1 for n in names]  # bir ızgaradaki kutu sayısı, kaydırma yüzünden bir fazladır
        idx = np.zeros(N_TILINGS, dtype=np.int64)  # her ızgarada durumun düştüğü kutu
        for d, n in enumerate(names):
            lo, width, count = VARS[n]
            x = np.clip((values[n] - lo) / width, 0.0, count - 1e-6)  # kutu genişliği cinsinden konum
            idx = idx * dims[d] + np.floor(x + OFFSETS[:, d]).astype(np.int64)
        per_tiling = int(np.prod(dims)) * (N_CONFIGS if use_config else 1)
        if use_config:
            idx = config * int(np.prod(dims)) + idx
        out.append(start + np.arange(N_TILINGS) * per_tiling + idx)
    out.append([n_features - 1])  # sabit her durumda açıktır
    return np.concatenate(out)


class TileAgent(HoldPolicy):
    def __init__(self, gamma: float = 0.99, extra: bool = False):
        self.gamma = gamma  # gelecekteki ödüllerin bir adımdaki (10 sn) ağırlığı
        self.extra = extra  # True ise kompresörlerin aynı durumda geçirdiği süre kullanılır
        n_features = group_starts(extra)[1]  # her hareket için ağırlık sayısı
        self.n_active = len(groups(extra)) * N_TILINGS + 1  # bir durumda açık olan kutu sayısı
        # Her hareket için her kutunun ağırlığı tutulur. Her 10 saniyede yaklaşık 0,25 enerji cezası alındığı için
        # sabitin ağırlığı -0,25 / (1 - gamma) ile başlar. Sıfırla başlasaydı denenmemiş hareketler olduğundan iyi
        # görünürdü.
        self.w = np.zeros((N_ACTIONS, n_features))
        self.w[:, -1] = -0.25 / (1.0 - gamma)
        super().__init__()

    def state(self, obs: np.ndarray) -> np.ndarray:
        """Gözlemi açık kutuların listesine çevirir. Bu ajan durum olarak bu listeyi kullanır."""
        return features(obs, self.extra)

    def values(self, f: np.ndarray) -> np.ndarray:
        """Her hareketin değerini açık kutulardaki ağırlıkları toplayarak hesaplar."""
        return self.w[:, f].sum(axis=1)

    def act(self, f: np.ndarray, epsilon: float) -> int:
        """epsilon olasılıkla rastgele bir hareket, aksi halde değeri en yüksek hareketi seçer (epsilon-greedy)."""
        if np.random.rand() < epsilon:
            return np.random.randint(N_ACTIONS)
        v = self.values(f)
        return int(np.random.choice(np.flatnonzero(v == v.max())))  # eşitlik varsa rastgele biri seçilir

    def update(self, f, a: int, r: float, f2, a2: int | None, alpha: float, discount: float):
        """Açık kutuların ağırlıklarını hedef ile tahmin arasındaki fark kadar kaydırır (semi-gradient güncelleme).

        Q-learning hedefi r + discount * max_a' Q(s', a') ile, SARSA ise r + discount * Q(s', a') ile hesaplar.
        Kaydırma açık kutulara eşit bölündüğü için alpha tablodaki öğrenme hızıyla aynı anlama gelir.
        """
        nxt = self.values(f2).max() if a2 is None else self.w[a2, f2].sum()
        delta = r + discount * nxt - self.w[a, f].sum()
        self.w[a, f] += alpha / self.n_active * delta

    def save(self, path):
        np.savez_compressed(path, w=self.w, gamma=self.gamma, extra=self.extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        agent = cls(float(d["gamma"]), bool(d["extra"]) if "extra" in d else False)
        agent.w = d["w"]
        return agent
