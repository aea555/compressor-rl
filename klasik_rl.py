"""Bu dosyada Q-learning ve SARSA tablo kullanılarak uygulanır.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızda çalışır ve kapalı, boşta ya da yükte
olabilir. VSD (değişken hızlı sürücü, 55 kW) kompresörünün devri ayarlanabilir.

Derin RL'de ajanın öğrendikleri bir sinir ağında tutulur. Burada bunun için bir tablo kullanılır.
Tablonun satırları durumlara, sütunları hareketlere karşılık gelir. Her hücrede, o durumda o hareket
yapılırsa bundan sonra toplam ne kadar ödül alınacağının tahmini (Q değeri) durur. Q-learning ve SARSA
bu tahmini güncellerken sonraki durumun değerini farklı hesaplar (TabularAgent.update).

Gözlem sürekli sayılardan oluştuğu için tabloya doğrudan girmez. Her bilgi birkaç aralığa bölünür ve
gözlem bir durum numarasına çevrilir. Varsayılan tabloda (genis) bilgiler aşağıdaki gibi bölünür.

    basınç          8 aralık (6,0 altı, 6,0-6,2, ..., 7,3 üstü)
    basıncın hızı   5 aralık (hızlı düşüş, yavaş düşüş, sabit, yavaş yükseliş, hızlı yükseliş)
    talep           6 aralık (sınırlar kompresör kapasitelerinden gelir, LAYOUTS)
    K1 ve K2        3'er değer (kapalı, boşta, yükte)
    VSD             6 değer (kapalı ya da ortamın izin verdiği 5 devir kademesi)
    saatlik kalkış  3 aralık (en çok kalkış yapan motorun son bir saatteki kalkışı 0-3, 4-5, 6 ve üstü)

Bu bölme 8 * 5 * 6 * 3 * 3 * 6 * 3 = 38880 durum verir. Günün saati ve motor sıcaklıkları tabloda yer almaz.

--tablo kompakt daha küçük bir tablo kullanır. Bu tabloda basıncın hızı 3 aralığa (hızlı düşüş, normal,
hızlı yükseliş), talep 4 aralığa (VSD yeter, K2 ve VSD yeter, K1 ve VSD yeter, hiçbiri yetmez), saatlik
kalkış da 2 aralığa (0-4, 5 ve üstü) bölünür. Bu bölme 8 * 3 * 4 * 3 * 3 * 6 * 2 = 10368 durum verir.
Tablo büyüdükçe her hücreyi öğrenmek için daha çok deneyim gerekir. Küçük tabloda bilgi azdır ama her
hücre daha iyi öğrenilir.

Her hareket o anki durumda yapılacak bir değişikliği gösterir (ACTIONS). Ajan ya her şeyi olduğu gibi bırakır
ya da tek bir kompresörde tek bir değişiklik yapar. Seçilen komut 30 saniye (3 adım) uygulanır, sonra yeni
karar verilir. Basınç 6,2 bar'ın altına inerse 30 saniye dolmadan yeniden karar verilir.

--uzun verilirse K1 ve K2 hareketleri o kadar dakika uygulanır. --uzun 5 ile "K2'yi kapat" hareketi K2'yi
5 dakika kapalı tutar. Böylece kapatmanın faydası tek bir hareketin içinde görülür. VSD hareketleri yine
30 saniye sürer.

--tablo kompakt2, kompakt tablonun basınç aralıklarını 6 bar civarında sıklaştırır (5,8 / 6,0 / 6,1 / 6,2 / 6,3 /
6,4 / 6,6 / 6,8 / 7,0 / 7,3). Eski tablolarda 6 bar'ın altı tek aralık olduğu için 5,99 bar ile 3 bar aynı
satıra düşer. Bu tablo 11 * 3 * 4 * 3 * 3 * 6 * 2 = 14256 durum verir.

Epsilon ajanın rastgele hareket seçme olasılığıdır. --eps-hizli verilirse epsilon iki parçada iner. İlk
eps_hizli adımda 1'den 0,1'e, sonra decay_steps'e kadar 0,1'den eps_end'e iner. Eski planda epsilon 24M adım
boyunca doğrusal iner. Bu planla yapılan eğitimin ilk yarısında basınç zamanın yarısından fazlasında 6 bar'ın
altında kalmış ve tablonun öğrenme hızı bu kötü durumlarda harcanmıştır. İyi kontrolün gezdiği durumlar ancak
öğrenme hızı çok düştükten sonra görülmüştür. Bu bulgu ve iki düzeltme bağımsız bir analizle elde edilmiştir
(bkz. ANALIZ_TABLO.md).

--maske verilirse etkisiz hareketler seçilemez. K1 zaten yükteyken "K1'i yüke al" ya da VSD en yüksek
kademedeyken "VSD'yi hızlandır" hareketi bu türdendir. Bu hareketler "olduğu gibi bırak" ile aynı işi yapar.

--acil-yok 6,2 bar kuralını kapatır. --sure-sec K1 ve K2 hareketlerinin süresini ajana seçtirir. Analizde
ajanın uzun hareketi kararların yalnızca %0,6'sında seçtiği görülmüştür. Kısa hareketten sonra yeniden seçim
hakkı olduğu için kısa hareket her zaman en az uzun hareket kadar değerlidir.

İlk sürümde (runs/klasik_*_v1) karar her 10 saniyede bir verilmiştir. Tabloda basıncın yalnızca yönü kullanılmış,
kalkış sayısı tabloya alınmamış ve talep eşit aralıklara bölünmüştür. Bu ajan basıncı korumuş ama motorları çok
sık açıp kapatmıştır. K2 yüke alındığında K2'nin durumu değiştiği için ajan tablonun başka bir satırına geçmiş
ve o satırın en iyi hareketi de bazen K2'yi geri boşa almak olmuştur.

    uv run python klasik_rl.py --method qlearning --out runs/klasik_q   # Q-learning ile eğitir
    uv run python klasik_rl.py --method sarsa --out runs/klasik_sarsa   # SARSA ile eğitir
    uv run python klasik_rl.py --tablo kompakt --gamma 0.98 --uzun 5 --out runs/g98_uzun_q_s0   # gürültüsü azaltılmış tabloyla eğitir
    uv run python klasik_rl.py --tablo kompakt2 --gamma 0.98 --acil-yok --eps-hizli 3000000 --maske --w-start 3 --out runs/k2_s0   # son tablo düzeniyle eğitir
    uv run python evaluate.py --episodes 20 --model runs/klasik_q/q_table.npz runs/yol_C/final_model.zip
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

P_EDGES = [6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.3]  # basınç aralıklarının sınırları [bar], iki tabloda da aynı

# dp basıncın 10 saniyedeki değişimini aralıklara böler [bar]. Baseline ile geçen bir günde adımların
# yarısı ±0,008 bar içinde kalır, %10'u 0,03 bar'dan hızlı değişir. q talebi böler [m³/dk]. VSD tek başına
# 9,5, K1 13, K2 ve VSD 17, K1 ve K2 20,5, K1 ve VSD 22,5 m³/dk basabilir. Ajan hangi kompresörlerin
# yeteceğini bu aralıklardan anlar. starts en çok kalkış yapan motorun son bir saatteki kalkışını böler.
LAYOUTS = {
    "genis": dict(p=P_EDGES, dp=[-0.03, -0.008, 0.008, 0.03], q=[9.5, 13.0, 17.0, 20.5, 22.5], starts=[4, 6]),
    "kompakt": dict(p=P_EDGES, dp=[-0.02, 0.02], q=[9.5, 17.0, 22.5], starts=[5]),
    # kompakt ile aynıdır, yalnızca basınç 6 bar civarında daha sık bölünür
    "kompakt2": dict(p=[5.8, 6.0, 6.1, 6.2, 6.3, 6.4, 6.6, 6.8, 7.0, 7.3], dp=[-0.02, 0.02], q=[9.5, 17.0, 22.5], starts=[5]),
}
P_REF = 7.0  # gözlemdeki basınç bu değerden farkı olarak verilir [bar], env.py'deki p_ref ile aynıdır
Q_CAPACITY = 30.0  # toplam kompresör kapasitesi [m³/dk], gözlemdeki talep buna bölünmüştür
MAX_STARTS = 6  # saatlik kalkış limiti, gözlemdeki kalkış sayısı buna bölünmüştür


def dims(layout: str) -> tuple:
    """Tablo düzeninde basınç, hız, talep, K1, K2, VSD ve kalkışın kaç değer alabildiğini döndürür."""
    L = LAYOUTS[layout]
    return (len(L["p"]) + 1, len(L["dp"]) + 1, len(L["q"]) + 1, 3, 3, 6, len(L["starts"]) + 1)


ACTIONS = [  # tablonun sütunları, ajan her kararda bunlardan birini seçer
    "her şeyi olduğu gibi bırak",
    "VSD'yi bir kademe hızlandır",
    "VSD'yi bir kademe yavaşlat",
    "K1'i yüke al",
    "K1'i boşa al",
    "K1'i kapat",
    "K2'yi yüke al",
    "K2'yi boşa al",
    "K2'yi kapat",
]
N_ACTIONS = len(ACTIONS)  # tablodaki hareket sayısı
HOLD = 3  # bir kararın uygulandığı adım sayısı, 3 adım 30 saniye eder
P_URGENT = 6.2  # basınç bunun altına inerse süre dolmadan yeniden karar verilir [bar]


def hold_for(a: int, hold_long: int, choose: bool = False) -> int:
    """Hareketin kaç adım uygulanacağını döndürür.

    choose True ise K1 ve K2 hareketlerinin iki kopyası bulunur. 3-8 arası hareketler 30 saniye, 9-14 arası
    hareketler hold_long adım sürer. Böylece hareketin ne kadar süreceğini ajan seçer.

    choose False ise VSD hareketleri ve "olduğu gibi bırak" hep 30 saniye, K1 ve K2 hareketleri hold_long adım
    sürer. hold_long HOLD'dan büyükse "K2'yi kapat" hareketi K2'yi kapatıp birkaç dakika kapalı tutar.
    Böylece kapatmanın faydası tek bir hareketin içinde görülür.
    """
    if choose:
        return hold_long if a >= 9 else HOLD
    return hold_long if a >= 3 else HOLD


def current(obs: np.ndarray) -> list[int]:
    """Gözlemden kompresörlerin şu anki durumunu [K1, K2, VSD kademesi] olarak okur."""
    k1 = int(np.argmax(obs[5:8]))  # 0 kapalı, 1 boşta, 2 yükte
    k2 = int(np.argmax(obs[11:14]))
    vsd = 0 if obs[17] < 0.5 else int(round((obs[18] - 0.2) / 0.2)) + 1  # 0 kapalı, 1-5 devir kademesi
    return [k1, k2, vsd]


def pressure(obs: np.ndarray) -> float:
    """Gözlemden hat basıncını okur [bar]."""
    return float(obs[0]) + P_REF


def discretize(obs: np.ndarray, layout: str = "genis") -> int:
    """Gözlemi (22 ya da 26 sayı) tablodaki durum numarasına çevirir.

    Gözlemin sırası env.py'deki _obs ile aynıdır. Önce basınç, basınç değişimi, talep ve saat (2 sayı) gelir.
    Sonra K1 için 6 sayı (3 sayı durum, sıcaklık, saatlik kalkış, kilit), K2 için 6 ve VSD için 5 sayı gelir.
    """
    dp = obs[1] * 0.2  # son adımdaki basınç değişimi [bar]
    q = obs[2] * Q_CAPACITY  # talep [m³/dk]
    starts = round(max(obs[9], obs[15], obs[20]) * MAX_STARTS)  # en çok kalkış yapan motorun son bir saatteki kalkışı
    k1, k2, vsd = current(obs)
    L = LAYOUTS[layout]
    parts = (
        int(np.searchsorted(L["p"], pressure(obs))),
        int(np.searchsorted(L["dp"], dp)),
        int(np.searchsorted(L["q"], q)),
        k1, k2, vsd,
        int(np.searchsorted(L["starts"], starts, side="right")),
    )
    return int(np.ravel_multi_index(parts, dims(layout)))


def to_command(a: int, obs: np.ndarray) -> np.ndarray:
    """Tablodaki hareketi (ACTIONS sırası) ortamın beklediği [2, 0, 3] gibi üç sayılık komuta çevirir."""
    cmd = current(obs)
    if a >= 9:  # süresi uzun olan kopyanın komutu kısa olanla aynıdır
        a -= 6
    if a == 1:
        cmd[2] = min(cmd[2] + 1, 5)
    elif a == 2:
        cmd[2] = max(cmd[2] - 1, 0)
    elif a >= 3:
        k, what = divmod(a - 3, 3)  # hangi kompresör (0 K1, 1 K2) ve ne yapılacağı
        cmd[k] = (2, 1, 0)[what]  # yükte, boşta ya da kapalı
    return np.array(cmd, dtype=np.int64)


def valid_actions(obs: np.ndarray, n_actions: int) -> np.ndarray:
    """Etkisiz olmayan hareketleri True olarak döndürür. "Olduğu gibi bırak" her zaman seçilebilir."""
    k1, k2, vsd = current(obs)
    m = np.ones(n_actions, dtype=bool)
    m[1], m[2] = vsd < 5, vsd > 0  # VSD en yüksek kademede hızlanamaz, kapalıyken yavaşlayamaz
    for a in range(3, n_actions):
        b = a - 6 if a >= 9 else a  # süresi uzun kopya kısa olanla aynı komutu verir
        k, what = divmod(b - 3, 3)
        m[a] = (2, 1, 0)[what] != (k1, k2)[k]  # kompresör zaten o durumdaysa hareket etkisizdir
    return m


def pick(agent, s: int, eps: float, obs: np.ndarray) -> int:
    """Ajanın hareketini seçer. Maske açıksa etkisiz hareketler seçilemez."""
    m = agent.mask_of(obs)
    return agent.act(s, eps) if m is None else agent.act(s, eps, m)


class HoldPolicy:
    """Testte her kararı eğitimdeki kadar uygular. Tablo ve tile coding ajanı bu sınıfı kullanır."""

    def __init__(self):
        self.cmd = None  # testte uygulanan komut
        self.held = 0  # bu komutun kaç adımdır uygulandığı
        self.limit = HOLD  # bu komutun kaç adım uygulanacağı

    def reset(self, env=None):
        """Testte her bölümün başında çağrılır."""
        self.cmd, self.held, self.limit = None, 0, HOLD

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Testte gözlemden komutu üretir, keşif yapmaz.

        Karar eğitimdeki kadar uygulanır. Acil kuralı açıkken basınç 6,2 bar'ın altına inerse hemen yeni
        karar verilir.
        """
        urgent = getattr(self, "urgent", True) and pressure(obs) < P_URGENT
        if self.cmd is None or self.held >= self.limit or urgent:
            a = pick(self, self.state(obs), 0.0, obs)
            self.cmd, self.held = to_command(a, obs), 0
            self.limit = hold_for(a, getattr(self, "hold_long", HOLD), getattr(self, "choose", False))
        self.held += 1
        return self.cmd

    def mask_of(self, obs: np.ndarray):
        """Maske açıksa seçilebilir hareketleri, kapalıysa None döndürür."""
        return valid_actions(obs, self.n_actions) if getattr(self, "mask", False) else None


class TabularAgent(HoldPolicy):
    def __init__(self, gamma: float = 0.995, layout: str = "genis", hold_long: int = HOLD,
                 urgent: bool = True, choose: bool = False, mask: bool = False):
        self.layout = layout  # tablo düzeni, LAYOUTS'taki anahtarlardan biri
        self.hold_long = hold_long  # K1 ve K2 hareketlerinin kaç adım sürdüğü
        self.urgent = urgent  # True ise basınç 6,2 bar'ın altına inince süre dolmadan yeniden karar verilir
        self.choose = choose  # True ise K1 ve K2 hareketlerinin süresini (30 sn ya da hold_long) ajan seçer
        self.mask = mask  # True ise etkisiz hareketler seçilemez
        self.n_actions = N_ACTIONS + 6 if choose else N_ACTIONS  # tablodaki hareket sayısı
        n_states = int(np.prod(dims(layout)))  # tablodaki durum sayısı
        self.q = np.zeros((n_states, self.n_actions))  # Q tablosu, her durum ve hareket için beklenen toplam ödül
        self.visits = np.zeros((n_states, self.n_actions), dtype=np.int64)  # her hücrenin kaç kez güncellendiği
        self.gamma = gamma  # gelecekteki ödüllerin bir adımdaki (10 sn) ağırlığı
        super().__init__()

    def act(self, s: int, epsilon: float, m: np.ndarray | None = None) -> int:
        """epsilon olasılıkla rastgele bir hareket, aksi halde tablodaki en iyi hareketi seçer (epsilon-greedy).

        Yalnızca en az bir kez denenmiş hareketlere bakılır. Tablo sıfırla başladığı için denenmemiş bir hareket,
        değeri hep eksi olan denenmiş hareketlerden iyi görünürdü. Hiçbir hareket denenmemişse her şey olduğu
        gibi bırakılır.
        """
        if m is None:
            m = np.ones(self.n_actions, dtype=bool)
        if np.random.rand() < epsilon:
            return int(np.random.choice(np.flatnonzero(m)))
        tried = (self.visits[s] > 0) & m
        if not tried.any():
            return 0
        row = np.where(tried, self.q[s], -np.inf)
        return int(np.random.choice(np.flatnonzero(row == row.max())))  # eşitlik varsa rastgele biri seçilir

    def update(self, s: int, a: int, r: float, s2: int, a2: int | None, alpha: float, discount: float):
        """Tablodaki tek bir hücreyi günceller.

        r kararın uygulandığı adımlarda toplanan ödülü, discount ise gamma'nın o adım sayısı kadar kuvvetini
        tutar. a2 verilmezse Q-learning, verilirse SARSA güncellemesi yapılır. Q-learning hedefi
        r + discount * max_a' Q(s', a') ile hesaplar ve sonraki durumdaki en iyi hareketin değerini kullanır.
        SARSA hedefi r + discount * Q(s', a') ile hesaplar ve sonraki durumda gerçekten seçilen hareketin
        değerini kullanır.

        Hiç denenmemiş hücredeki 0 gerçek bir tahmin olmadığı için hedefte yalnızca denenmiş hücreler
        kullanılır. Bir hücre ilk kez güncellenirken de 0 unutulur ve doğrudan hedef yazılır.
        """
        tried = self.visits[s2] > 0
        if not tried.any():
            # Sonraki durum hiç görülmediği için az önceki ödülün hep böyle süreceği varsayılır. Önceki sürümde
            # burada hücrenin kendi değeri kullanılmıştır. İlk ziyarette bu değer de 0 olduğundan bilinmeyen
            # durumlara gitmek neredeyse cezasız görünmüştür.
            nxt = r / (1.0 - discount)
        elif a2 is None or not tried[a2]:
            nxt = self.q[s2][tried].max()
        else:
            nxt = self.q[s2, a2]
        step = 1.0 if self.visits[s, a] == 0 else alpha
        self.q[s, a] += step * (r + discount * nxt - self.q[s, a])
        self.visits[s, a] += 1

    def state(self, obs: np.ndarray) -> int:
        """Gözlemi bu ajanın tablo düzenine göre durum numarasına çevirir."""
        return discretize(obs, self.layout)

    def save(self, path):
        np.savez_compressed(path, q=self.q, visits=self.visits, gamma=self.gamma, layout=self.layout,
                            hold_long=self.hold_long, urgent=self.urgent, choose=self.choose, mask=self.mask)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        agent = cls(float(d["gamma"]), str(d["layout"]) if "layout" in d else "genis",
                    int(d["hold_long"]) if "hold_long" in d else HOLD,
                    bool(d["urgent"]) if "urgent" in d else True, bool(d["choose"]) if "choose" in d else False,
                    bool(d["mask"]) if "mask" in d else False)
        agent.q, agent.visits = d["q"], d["visits"]
        return agent


def load_agent(path):
    """Kaydedilmiş tablo ya da tile coding ajanını yükler. Hangisi olduğu dosyanın içinden anlaşılır."""
    if "w" in np.load(path):
        from klasik_tile import TileAgent

        return TileAgent.load(path)
    return TabularAgent.load(path)


def train(args):
    from compressor_env import CompressorEnv, EnvConfig

    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    # varsayılandan farklı ortam ayarları PPO'daki gibi env_config.json'a yazılır
    overrides = {"w_start": args.w_start}
    if args.extra_obs:  # gözleme talep eğilimi ve kompresörlerin aynı durumda geçirdiği süre eklenir, bunları yalnızca tile coding kullanır
        overrides["extra_obs"] = True
    (args.out / "env_config.json").write_text(json.dumps(overrides, indent=2))
    (args.out / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    env = CompressorEnv(replace(EnvConfig(), **overrides))
    if args.tablo == "tile":
        from klasik_tile import TileAgent

        agent = TileAgent(args.gamma, args.extra_obs)
    else:
        agent = TabularAgent(args.gamma, args.tablo, round(args.uzun * 6) if args.uzun else HOLD,
                             urgent=not args.acil_yok, choose=args.sure_sec, mask=args.maske)
    sarsa = args.method == "sarsa"
    log = []  # her bölümün özeti, eğitim eğrisi için

    obs, _ = env.reset(seed=args.seed)
    s = agent.state(obs)
    eps = 1.0
    a = pick(agent, s, eps, obs)
    ep_count, step = 0, 0
    next_snap = args.ara_kayit  # bir sonraki ara kaydın alınacağı adım
    while step < args.timesteps:
        frac = min(step / args.decay_steps, 1.0)  # azalmanın ne kadar ilerlediği (0-1)
        if args.eps_hizli and step < args.eps_hizli:  # ilk eps_hizli adımda epsilon 1'den 0,1'e hızlı iner
            eps = 1.0 - 0.9 * step / args.eps_hizli
        elif args.eps_hizli:  # sonra 0,1'den eps_end'e yavaş iner
            eps = 0.1 + min((step - args.eps_hizli) / (args.decay_steps - args.eps_hizli), 1.0) * (args.eps_end - 0.1)
        else:
            eps = 1.0 + frac * (args.eps_end - 1.0)  # eski planda keşif olasılığı 1'den eps_end'e doğrusal iner
        alpha = args.alpha + frac * (args.alpha_end - args.alpha)  # öğrenme hızı da doğrusal iner

        # karar en fazla limit adım uygulanır, bu sürede gelen ödüller gamma ile indirilerek toplanır
        cmd = to_command(a, obs)
        limit = hold_for(a, getattr(agent, "hold_long", HOLD), getattr(agent, "choose", False))  # en fazla kaç adım
        r_sum, discount, k = 0.0, 1.0, 0
        while True:
            obs, r, terminated, truncated, info = env.step(cmd)
            step += 1
            k += 1
            r_sum += discount * r
            discount *= args.gamma
            done = terminated or truncated
            if done or k >= limit or (getattr(agent, "urgent", True) and pressure(obs) < P_URGENT):
                break
        s2 = agent.state(obs)
        a2 = pick(agent, s2, eps, obs)  # sonraki hareket şimdiden seçilir, SARSA'nın güncellemesi bu hareketi kullanır
        # Bölüm yalnızca süre dolunca biter. Bu gerçek bir son olmadığı için sonraki durumun değeri yine kullanılır.
        if args.ortalama and args.tablo != "tile":
            # Öğrenme hızı 1/n olarak alınır, n hücrenin güncellenme sayısıdır. Böylece hücre gördüğü bütün
            # sonuçların ortalaması olur ve tek tük şanslı ya da şanssız günler değeri az oynatır. Öğrenme hızı
            # alpha_end'in altına inmez.
            alpha = max(1.0 / (agent.visits[s, a] + 1), args.alpha_end)
        agent.update(s, a, r_sum, s2, a2 if sarsa else None, alpha, discount)
        s, a = s2, a2

        if done:
            ep_count += 1
            st = info["episode_stats"]
            log.append(dict(step=step, energy=st["energy_kwh"], low_p=st["low_p_min"], starts=st["starts"],
                            switches=st["load_switches"]))
            if ep_count % args.print_every == 0:
                last = log[-args.print_every:]
                print(f"adım {step:>9} | bölüm {ep_count:>5} | enerji {np.mean([x['energy'] for x in last]):>5.0f} kWh | "
                      f"6 bar altı {np.mean([x['low_p'] for x in last]):>5.1f} dk | kalkış {np.mean([x['starts'] for x in last]):>5.1f} | "
                      f"yük/boş {np.mean([x['switches'] for x in last]):>5.0f} | eps {eps:.2f}", flush=True)
            # ara kayıt, modelin eğitim boyunca nasıl geliştiğini sonradan test etmek için alınır
            if args.ara_kayit and step >= next_snap:
                (args.out / "ara").mkdir(exist_ok=True)
                agent.save(args.out / "ara" / f"q_{step}.npz")
                next_snap += args.ara_kayit
            obs, _ = env.reset()
            s = agent.state(obs)
            a = pick(agent, s, eps, obs)

    agent.save(args.out / "q_table.npz")
    (args.out / "egitim.json").write_text(json.dumps(log))
    if args.tablo == "tile":
        print(f"Kaydedildi: {args.out / 'q_table.npz'}")
    else:
        seen = (agent.visits.sum(axis=1) > 0).mean() * 100  # en az bir kez görülen durumların oranı [%]
        print(f"Kaydedildi: {args.out / 'q_table.npz'}. Durumların %{seen:.0f}'i en az bir kez görüldü.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["qlearning", "sarsa"], default="qlearning")
    # tablo düzeni seçilir, "tile" verilirse tablo yerine tile coding kullanılır (klasik_tile.py)
    ap.add_argument("--tablo", choices=list(LAYOUTS) + ["tile"], default="genis")
    ap.add_argument("--timesteps", type=int, default=30_000_000)  # toplam eğitim adımı, her adım 10 sn
    ap.add_argument("--decay-steps", type=int, default=24_000_000)  # epsilon ve öğrenme hızının azaldığı adım sayısı
    ap.add_argument("--eps-end", type=float, default=0.01)  # epsilon'un en düşük değeri
    ap.add_argument("--alpha", type=float, default=0.2)  # başlangıçtaki öğrenme hızı
    ap.add_argument("--alpha-end", type=float, default=0.01)  # sondaki öğrenme hızı
    ap.add_argument("--extra-obs", action="store_true")  # verilirse tile coding kompresörlerin kaç dakikadır aynı durumda olduğunu da görür
    # K1 ve K2 hareketlerinin kaç dakika süreceğini belirler, verilmezse hareketler 30 saniye sürer. Bu ayar
    # yalnızca tabloda kullanılır.
    ap.add_argument("--uzun", type=float, default=0)
    # basınç 6,2 bar'ın altına inince süre dolmadan yeniden karar verme kuralını kapatır
    ap.add_argument("--acil-yok", action="store_true")
    # verilirse K1 ve K2 hareketlerinin 30 saniye mi --uzun kadar mı süreceğini ajan seçer
    ap.add_argument("--sure-sec", action="store_true")
    # epsilon'un 1'den 0,1'e hızlı indiği adım sayısı, 0 verilirse eski doğrusal plan kullanılır
    ap.add_argument("--eps-hizli", type=int, default=0)
    ap.add_argument("--maske", action="store_true")  # verilirse etkisiz hareketler seçilemez
    ap.add_argument("--ortalama", action="store_true")  # verilirse öğrenme hızı her hücrede 1/(güncelleme sayısı) olur
    # gelecekteki ödüllerin 10 sn'lik bir adımdaki ağırlığı, 0,995 ile ajan yaklaşık 200 adım (33 dk) ileriye bakar
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--w-start", type=float, default=1.0)  # motor kalkışı cezası, PPO-C ile aynı
    ap.add_argument("--seed", type=int, default=0)
    # kaç adımda bir tablonun ara kaydının alınacağı (out/ara/q_<adım>.npz), 0 ise ara kayıt alınmaz
    ap.add_argument("--ara-kayit", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=500)  # kaç bölümde bir özet yazılacağı
    ap.add_argument("--out", type=Path, default=Path("runs/klasik_q"))
    train(ap.parse_args())
