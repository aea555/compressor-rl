"""Bu dosya fabrikanın hava tüketimini üretir.

Her bölümün başında vardiya boyunca her 10 saniyede ne kadar hava tüketileceği baştan hesaplanır.
Talep m³/dk cinsinden verilir. Taban talep gece düşük, gündüz yüksektir. Molalarda ve öğle arasında
talep düşer. Bunun üstüne rastgele zamanlarda devreye girip bir süre çalışan makineler ve küçük
dalgalanmalar eklenir. Her bölümde rastgele kısımlar değiştiği için ajan tek bir günü ezberleyemez.
"""

import numpy as np

# Her satırda vardiyanın başlangıç ve bitiş saati ile ortalama talebi [m³/dk] yazılıdır.
SHIFT_BASE = [(0, 6, 6.0), (6, 14, 20.0), (14, 22, 15.0), (22, 24, 6.0)]
# Her satırda molanın başlangıç ve bitiş saati ile talep çarpanı yazılıdır. 0,55 çarpanı talebi yüzde 45 düşürür.
BREAKS = [(10.0, 10.25, 0.7), (12.0, 12.5, 0.55), (18.0, 18.5, 0.6), (2.0, 2.5, 0.8)]


def _base_at(hour: float) -> float:
    """Verilen saatteki (0-24) taban talebi [m³/dk] döndürür. Mola saatindeyse talebi düşürür."""
    for start, end, q in SHIFT_BASE:
        if start <= hour < end:
            base = q
            break
    for start, end, k in BREAKS:
        if start <= hour < end:
            base *= k
    return base


class DemandProfile:
    """Bir bölümün hava talebini tutar. `profile[adım]` o adımdaki talebi verir."""

    def __init__(
        self,
        rng: np.random.Generator,  # rastgele sayı üreteci, aynı seed aynı günü üretir
        start_s: float,  # bölümün başladığı saat, gece yarısından itibaren [s]
        n_steps: int,  # bölümdeki adım sayısı
        dt: float,  # bir adımın süresi [s]
        scale_range=(0.85, 1.15),  # günün yoğunluğu, bütün talep bu aralıktan seçilen bir çarpanla ölçeklenir
        event_rate_per_h: float = 4.0,  # saatte ortalama kaç kez ek makine devreye girdiği
        event_size=(1.0, 4.0),  # devreye giren makinenin ek tüketimi [m³/dk]
        event_mean_min: float = 10.0,  # ek makinenin ortalama çalışma süresi [dk]
        noise_sigma: float = 0.8,  # dalgalanmanın büyüklüğü [m³/dk]
        noise_tau_s: float = 60.0,  # dalgalanmanın ne kadar yavaş değiştiği [s]
        q_max: float = 26.0,  # talebin üst sınırı [m³/dk], toplam kompresör kapasitesi (30) aşılmasın diye konur
    ):
        self.dt = dt
        t = start_s + np.arange(n_steps) * dt  # her adımın günün kaçıncı saniyesine denk geldiği [s]
        hours = (t / 3600.0) % 24.0  # her adımın saati (0-24)

        # 1) vardiyanın taban talebi günün yoğunluk çarpanıyla çarpılır
        base = np.array([_base_at(h) for h in hours]) * rng.uniform(*scale_range)
        # vardiya değişirken talep birden zıplamasın, 10 dakikaya yayılsın diye yumuşatılır
        k = max(1, int(600 / dt))  # 10 dakikadaki adım sayısı
        pad = np.pad(base, (k // 2, k - 1 - k // 2), mode="edge")
        base = np.convolve(pad, np.ones(k) / k, mode="valid")

        # 2) rastgele zamanlarda ek makineler devreye girer
        events = np.zeros(n_steps)  # her adımdaki ek tüketim [m³/dk]
        n_events = rng.poisson(event_rate_per_h * n_steps * dt / 3600.0)  # bu bölümde kaç kez makine devreye girdiği
        for _ in range(n_events):
            i0 = rng.integers(0, n_steps)  # makinenin devreye girdiği adım
            length = max(1, int(rng.exponential(event_mean_min * 60.0) / dt))  # kaç adım çalışacağı
            events[i0 : i0 + length] += rng.uniform(*event_size)

        # 3) yavaş değişen rastgele dalgalanma eklenir. Her adımdaki değer bir öncekine yakın kalır. Bu tür
        # gürültüye Ornstein-Uhlenbeck gürültüsü denir.
        noise = np.zeros(n_steps)  # her adımdaki dalgalanma [m³/dk]
        a = np.exp(-dt / noise_tau_s)  # önceki değerin ne kadarının korunduğu
        s = noise_sigma * np.sqrt(1 - a**2)  # her adımda eklenen yeni rastgelelik [m³/dk]
        for i in range(1, n_steps):
            noise[i] = a * noise[i - 1] + s * rng.standard_normal()

        self.q = np.clip(base + events + noise, 0.5, q_max)  # her adımdaki toplam talep [m³/dk]

    def __getitem__(self, step: int) -> float:
        return float(self.q[min(step, len(self.q) - 1)])

    def __len__(self) -> int:
        return len(self.q)
