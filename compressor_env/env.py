"""Bu dosyada basınçlı hava sistemi simüle edilir.

Ajan ya da klasik kontrolcü her 10 saniyede bir kompresörlere komut verir. Burada o 10 saniyede ne kadar
hava basıldığı, ne kadar tüketildiği, basıncın ne olduğu ve ne kadar elektrik harcandığı hesaplanır.
Bir bölüm 8 saatlik bir vardiyaya karşılık gelir.

Basınç aşağıdaki basit modelle hesaplanır.

    dP/dt = (P_atm / V) * (Q_üretilen - Q_tüketilen)

Kompresörler tüketimden fazla hava basarsa basınç yükselir, az basarsa düşer.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızlıdır ve kapalı, boşta ya da yükte olabilir.
Boştaki motor döner ama hava basmaz, buna rağmen tam gücün yüzde 30 kadarını çeker. VSD (değişken hızlı
sürücü, 55 kW) kompresörünün devri ayarlanabildiği için bu kompresör ihtiyaç kadar hava basabilir.

Ajanın her adımda verdiği komut [2, 0, 3] gibi üç sayıdan oluşur. Sayıların anlamı aşağıda verilir.

    K1    0 kapalı, 1 boşta, 2 yükte
    K2    0 kapalı, 1 boşta, 2 yükte
    VSD   0 kapalı, 1-5 devir kademesi (5 tam hız)
"""

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .demand import DemandProfile

P_ATM = 1.013  # atmosfer basıncı [bar]

OFF, UNLOAD, LOAD = 0, 1, 2  # sabit hızlı kompresörün durumları, komuttaki sayılarla aynıdır


@dataclass(frozen=True)
class FixedSpec:
    """Sabit hızlı bir kompresörün katalog değerlerini tutar."""

    name: str  # grafiklerde görünen isim, "K1" gibi
    rated_kw: float  # tam yükte çektiği güç [kW]
    flow_m3min: float  # yükteyken bastığı hava [m³/dk]
    unload_frac: float = 0.30  # boştayken tam gücün ne kadarını çektiği, 0.30 yüzde 30 anlamına gelir


@dataclass(frozen=True)
class VSDSpec:
    """Devri ayarlanabilen (VSD) kompresörün katalog değerlerini tutar."""

    name: str  # grafiklerde görünen isim
    rated_kw: float  # tam devirde çektiği güç [kW]
    flow_max_m3min: float  # tam devirde bastığı hava [m³/dk]
    min_speed: float = 0.2  # çalışabildiği en düşük devir, tam devrin oranı olarak
    n_levels: int = 5  # ajanın seçebileceği devir kademesi sayısı, kapalı hariç
    fixed_loss_frac: float = 0.12  # sürücü ve fan gibi devirden bağımsız kayıplar, tam gücün oranı olarak


@dataclass(frozen=True)
class EnvConfig:
    """Simülasyonun bütün ayarlarını tutar."""

    # Kompresörler
    fixed: tuple = (  # sabit hızlı kompresörler, sırayla isim, güç [kW] ve debi [m³/dk]
        FixedSpec("K1", 75.0, 13.0),
        FixedSpec("K2", 45.0, 7.5),
    )
    vsd: VSDSpec = VSDSpec("K3-VSD", 55.0, 9.5)  # devri ayarlanabilen kompresör

    # Zaman
    dt: float = 10.0  # bir adımın süresi [s], ajan bu aralıkla karar verir
    episode_hours: float = 8.0  # bir bölümün uzunluğu [saat], bir vardiyaya eşittir

    # Basınç ve hat
    volume_m3: float = 20.0  # tank ve boru hattının toplam hacmi [m³], hacim büyüdükçe basınç daha yavaş değişir
    p_init_range: tuple = (6.6, 7.2)  # bölüm başındaki basınç bu aralıktan rastgele seçilir [bar]
    p_min: float = 6.0  # basınç bunun altına inerse üretim durabilir ve ceza verilir [bar]
    p_margin: float = 6.2  # basınç bunun altına inince hafif ceza başlar [bar]
    p_max: float = 8.0  # basınç bunun üstündeyse kompresörler emniyet için boşa alınır [bar]
    p_relief: float = 8.5  # emniyet ventili basıncın bunu geçmesine izin vermez [bar]
    p_ref: float = 7.0  # güç ve tüketim hesabında normal kabul edilen basınç [bar]
    power_per_bar: float = 0.07  # basınç 1 bar artınca kompresör gücü yüzde 7 artar
    leak_m3min: float = 2.0  # hattaki kaçaklardan kaybolan hava [m³/dk]

    # motor sıcaklığı
    ambient_c: float = 25.0  # ortam sıcaklığı, kapalı motor buna doğru soğur [°C]
    temp_rise_c: float = 60.0  # tam yükte motorun ortamın ne kadar üstüne ısındığı [°C]
    temp_tau_s: float = 900.0  # ısınma ve soğumanın zaman sabiti, yaklaşık 15 dk [s]
    start_heat_c: float = 6.0  # her kalkışta kalkış akımından gelen ek ısınma [°C]
    trip_temp_c: float = 105.0  # motor bu sıcaklıkta kendini korumak için kapanır, buna trip denir [°C]
    trip_lockout_s: float = 600.0  # trip sonrası motorun açılamadığı süre [s]
    max_starts_per_hour: int = 6  # bir motor için saatte izin verilen en fazla kalkış

    # Ödül ağırlıkları
    w_energy: float = 1.0  # enerji cezasının ağırlığı
    w_low_p: float = 3.0  # basınç 6 bar altındaysa her adımda verilen ceza
    # Basınç 6,2 ile 6,0 bar arasındayken her adımda ceza verilir. Ceza 6,2 bar'da sıfırdır, 6,0 bar'da bu değere
    # çıkar. İkinci denemede ajan basıncı tam 6 bar sınırında tuttuğu için bu ceza eklenmiştir.
    w_margin: float = 0.5
    w_start: float = 0.3  # her motor kalkışı için ceza
    # Her yük-boş geçişi için ceza verilir. İlk denemede bu değer 0,03 iken ajan motoru kapatmak yerine sürekli
    # yük ile boş arasında gidip gelmeyi öğrenmiştir. Bu davranış vanayı aşındırdığı için ceza artırılmıştır.
    w_load_switch: float = 0.2
    w_excess_start: float = 2.0  # saatlik kalkış limiti aşılırsa verilen ek ceza
    w_trip: float = 10.0  # motor aşırı ısınıp trip olursa verilen ceza

    # Gözlem
    # True olursa ajan iki bilgi daha görür. Biri son 1 dakikadaki tüketim değişimi, diğeri her kompresörün
    # şu anki durumunda ne kadar kaldığıdır. Bu durumda gözlem 22 sayıdan 26 sayıya çıkar. Varsayılan değer
    # False olduğu için 22 sayı bekleyen eski modeller çalışmaya devam eder.
    extra_obs: bool = False

    # Aşağıdaki ayarlar yalnızca dayanıklılık testlerinde değiştirilir, eğitimde varsayılanlar kullanılır.
    demand_scale: float = 1.0  # hava talebinin çarpanı, 1.2 değeri talebi yüzde 20 artırır
    # Arıza sırasıyla kompresörün sıra numarası (0'dan başlar), başlangıç [saat] ve süre [saat] olarak verilir.
    # Boş bırakılırsa arıza olmaz. (1, 3.0, 1.0) verilirse K2 bölüm başladıktan 3 saat sonra bozulur ve 1 saat
    # çalışmaz. Arızalı kompresör trip kilidindeymiş gibi davranır ve ajan bu kilidi gözlemde görür.
    failure: tuple = ()


class CompressorEnv(gym.Env):
    """Ajanın çalıştığı ortamı Gymnasium arayüzüyle tanımlar."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = cfg = config or EnvConfig()  # bütün ayarlar
        self.n_fixed = len(cfg.fixed)  # sabit hızlı kompresör sayısı (2)
        self.n_comp = self.n_fixed + 1  # sabitler ve VSD ile toplam kompresör sayısı
        self.n_steps = int(cfg.episode_hours * 3600 / cfg.dt)  # bir bölümdeki adım sayısı (8 saatte 2880)
        # bütün kompresörler tam çalışınca basılan hava [m³/dk]
        self.q_capacity = sum(f.flow_m3min for f in cfg.fixed) + cfg.vsd.flow_max_m3min
        # bütün kompresörlerin toplam gücü [kW], enerji cezasını 0-1 aralığına getirmek için kullanılır
        self.p_rated_total = sum(f.rated_kw for f in cfg.fixed) + cfg.vsd.rated_kw

        # komutta her sabit kompresör için 3 seçenek, VSD için kapalı ve n_levels kademe bulunur
        self.action_space = spaces.MultiDiscrete([3] * self.n_fixed + [cfg.vsd.n_levels + 1])
        # gözlemde 5 genel değer, her sabit kompresör için 6 ve VSD için 5 değer bulunur (_obs)
        obs_dim = 5 + self.n_fixed * 6 + 5
        if cfg.extra_obs:
            obs_dim += 1 + self.n_comp  # talep eğilimi ve her kompresörün aynı durumda geçirdiği süre
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ yardımcılar
    def vsd_speed(self, level: int) -> float:
        """VSD devir kademesini (0-n_levels) devir oranına (0-1) çevirir.

        Varsayılan ayarlarda 0 kapalı anlamına gelir, 1 ise 0,2 devre karşılık gelir. Sonraki her kademe 0,2 ekler
        ve 5 tam hıza denk gelir.
        """
        v = self.cfg.vsd
        if level == 0:
            return 0.0
        if v.n_levels == 1:
            return 1.0
        return v.min_speed + (1.0 - v.min_speed) * (level - 1) / (v.n_levels - 1)

    def _clock_hour(self) -> float:
        """Fabrikadaki saati (0-24) döndürür. 13,5 değeri 13.30'a karşılık gelir."""
        return ((self.start_s + self.t) / 3600.0) % 24.0

    def _starts_last_hour(self, i: int) -> int:
        """i. kompresörün son bir saatte kaç kez çalıştırıldığını döndürür."""
        q = self.start_times[i]
        while q and q[0] <= self.t - 3600.0:  # bir saatten eski kalkışlar silinir
            q.popleft()
        return len(q)

    def _obs(self) -> np.ndarray:
        """Ajanın gördüğü gözlemi hazırlar.

        Sinir ağı küçük sayılarla daha rahat öğrendiği için değerler kabaca -1 ile 1 arasına ölçeklenir.
        """
        cfg = self.cfg
        h = self._clock_hour() * 2 * np.pi / 24.0  # saat açıya çevrildiği için 23.59 ile 00.00 yan yana düşer
        obs = [
            (self.p - cfg.p_ref) / 1.0,  # hat basıncının 7 bar'dan farkı
            self.dp / 0.2,  # son adımdaki basınç değişimi
            self.q_out / self.q_capacity,  # hava tüketiminin toplam kapasiteye oranı
            np.sin(h),  # saat, sin ve cos birlikte
            np.cos(h),
        ]
        for i in range(self.n_fixed):  # her sabit kompresör için 6 değer
            onehot = [0.0, 0.0, 0.0]  # kompresör hangi durumdaysa (kapalı, boşta, yükte) o eleman 1 olur
            onehot[self.state[i]] = 1.0
            obs += onehot
            obs += [
                (self.temp[i] - cfg.ambient_c) / cfg.temp_rise_c,  # motor sıcaklığı, 0 soğuk, 1 tam yük sıcaklığı
                self._starts_last_hour(i) / cfg.max_starts_per_hour,  # saatlik kalkış limitinin kullanılan kısmı
                self.lockout[i] / cfg.trip_lockout_s,  # trip kilidinden kalan süre, 0 ise kilit yok
            ]
        j = self.n_fixed  # VSD listelerde en sonda durur
        obs += [  # VSD için 5 değer
            float(self.state[j] > 0),  # açıksa 1, kapalıysa 0
            self.vsd_speed(self.state[j]),  # devir oranı (0-1)
            (self.temp[j] - cfg.ambient_c) / cfg.temp_rise_c,  # motor sıcaklığı
            self._starts_last_hour(j) / cfg.max_starts_per_hour,  # saatlik kalkış limitinin kullanılan kısmı
            self.lockout[j] / cfg.trip_lockout_s,  # trip kilidinden kalan süre
        ]
        if cfg.extra_obs:
            # talep eğilimi son 1 dakikada tüketimin ne kadar değiştiğini gösterir, 3 m³/dk artış +1 olarak görünür
            obs.append((self.q_hist[-1] - self.q_hist[0]) / self.q_capacity * 10.0)
            # her kompresörün şu anki durumunda geçirdiği süre eklenir, 0 durumun az önce değiştiğini, 1 en az 30 dakikadır
            # aynı kaldığını gösterir
            obs += [min((self.t - self.state_since[i]) / 1800.0, 1.0) for i in range(self.n_comp)]
        return np.clip(np.asarray(obs, dtype=np.float32), -5.0, 5.0)

    # ---------------------------------------------------------------- Gymnasium API
    def reset(self, *, seed=None, options=None):
        """Yeni bir bölüm başlatır. Her bölüm bir vardiyaya karşılık gelir."""
        super().reset(seed=seed)
        cfg = self.cfg
        options = options or {}
        # vardiyanın başladığı saat verilmezse rastgele seçilir
        start_hour = options.get("start_hour", self.np_random.uniform(0, 24))
        self.start_s = start_hour * 3600.0  # başlangıç saati, gece yarısından itibaren [s]
        # bu bölümdeki hava tüketimi baştan rastgele üretilir
        self.demand = DemandProfile(self.np_random, self.start_s, self.n_steps, cfg.dt)
        # Test senaryosunda talep ölçeklenir. Aynı seed aynı günü üretir ve gün yalnızca daha yoğun ya da sakin geçer.
        self.demand.q = self.demand.q * cfg.demand_scale

        self.step_i = 0  # şu anki adım numarası
        self.t = 0.0  # bölüm başından beri geçen süre [s]
        self.p = self.np_random.uniform(*cfg.p_init_range)  # hat basıncı [bar]
        self.dp = 0.0  # son adımdaki basınç değişimi [bar]
        self.q_out = self.demand[0]  # hattan çekilen hava [m³/dk]
        # Son bir dakikanın ve şimdiki adımın, toplam 7 adımın tüketimi [m³/dk] tutulur. Talep eğilimi bundan hesaplanır.
        self.q_hist = deque([self.q_out] * 7, maxlen=7)
        # Sabit kompresörün durumu OFF, UNLOAD ya da LOAD olur, VSD'nin durumu ise devir kademesidir.
        # Bölüm başında sabitler kapalıdır ve VSD orta kademede başlar.
        self.state = [OFF] * self.n_fixed + [(cfg.vsd.n_levels + 1) // 2]
        self.temp = [cfg.ambient_c] * self.n_comp  # motor sıcaklıkları [°C]
        self.lockout = [0.0] * self.n_comp  # trip kilidinden kalan süre [s], 0 ise kilit yok
        self.start_times = [deque() for _ in range(self.n_comp)]  # her motorun kalkış zamanları [s], saatlik limit için
        self.state_since = [0.0] * self.n_comp  # her kompresörün şu anki durumuna geçtiği an [s]
        # bölüm sonunda raporlanan toplamlar burada tutulur
        self.stats = dict(
            energy_kwh=0.0,  # harcanan elektrik [kWh]
            air_m3=0.0,  # basılan hava [m³]
            low_p_s=0.0,  # basıncın 6 bar altında kaldığı süre [s]
            margin_s=0.0,  # basıncın 6,2 bar altında kaldığı süre [s]
            starts=0,  # motor kalkışı sayısı
            load_switches=0,  # yük-boş geçişi sayısı
            excess_starts=0,  # saatlik limiti aşan kalkış sayısı
            trips=0,  # aşırı ısınmadan kapanma sayısı
            p_sum=0.0,  # basınçların toplamı [bar], ortalama basınç için
        )
        return self._obs(), {}

    def step(self, action):
        """Ajanın komutunu uygular ve simülasyonu 10 saniye ilerletir."""
        cfg = self.cfg
        action = np.asarray(action, dtype=np.int64)
        # aşağıdaki sayaçlar bu adımın ödül hesabında kullanılır
        starts = 0  # kapalıyken açılan motor sayısı
        load_switches = 0  # yük-boş geçişi yapan kompresör sayısı
        excess = 0  # saatlik limiti aşan kalkış sayısı
        trips = 0  # aşırı ısınıp kapanan motor sayısı

        # 0) test senaryosunda arıza süresi içindeyse arızalı kompresör kilitli tutulur
        if cfg.failure:
            i_f, start_h, dur_h = cfg.failure  # arızalı kompresör, başlangıç [saat], süre [saat]
            end_s = (start_h + dur_h) * 3600.0  # arızanın bittiği an [s]
            if start_h * 3600.0 <= self.t < end_s:
                # Kilit süresi olarak arızanın bitmesine kalan süre yazılır. Gözlem eğitimde görülen aralığın
                # dışına çıkmasın diye en fazla trip_lockout_s gösterilir ve değer her adımda yenilenir.
                self.lockout[i_f] = max(self.lockout[i_f], min(end_s - self.t, cfg.trip_lockout_s))

        # 1) komutlar uygulanır, kilit ve emniyet kuralları ajanın komutunu değiştirebilir
        for i in range(self.n_comp):
            is_vsd = i == self.n_fixed  # bu kompresör VSD mi
            cmd = int(action[i])  # ajanın bu kompresöre verdiği komut
            if self.lockout[i] > 0:
                cmd = OFF  # kilitli motor çalıştırılamaz
            elif self.p >= cfg.p_max and cmd > 0:
                cmd = 1 if is_vsd else UNLOAD  # basınç çok yüksek olduğu için VSD en düşük kademeye iner, sabitler boşa alınır
            prev = self.state[i]  # önceki durum
            if cmd != prev:
                self.state_since[i] = self.t
                if prev == OFF:  # kapalıyken açılan motor bir kalkış sayılır
                    starts += 1
                    self.start_times[i].append(self.t)
                    self.temp[i] += cfg.start_heat_c
                    if self._starts_last_hour(i) > cfg.max_starts_per_hour:
                        excess += 1
                elif cmd != OFF and not is_vsd:  # motor zaten döndüğü için bu değişiklik yük-boş geçişi sayılır
                    load_switches += 1
            self.state[i] = cmd

        # 2) basılan hava ve çekilen güç hesaplanır
        # basınç yükseldikçe kompresör daha çok zorlandığı için bu katsayı gücü artırır
        press_factor = 1.0 + cfg.power_per_bar * (max(self.p, cfg.p_min - 1.0) - cfg.p_ref)
        q_in = 0.0  # bu adımda basılan toplam hava [m³/dk]
        power = 0.0  # bu adımda çekilen toplam güç [kW]
        load_frac = []  # her motorun yük oranı (0-1), sıcaklık hesabı için
        for i, spec in enumerate(cfg.fixed):
            s = self.state[i]
            if s == LOAD:  # yükteki kompresör hava basar ve tam güç çeker
                q_in += spec.flow_m3min
                power += spec.rated_kw * press_factor
                load_frac.append(1.0)
            elif s == UNLOAD:  # boştaki kompresör hava basmaz ama yine de güç çeker
                power += spec.rated_kw * spec.unload_frac
                load_frac.append(0.35)
            else:  # kapalı
                load_frac.append(0.0)
        v = cfg.vsd
        speed = self.vsd_speed(self.state[self.n_fixed])  # VSD devir oranı (0-1)
        if speed > 0:  # VSD'nin debisi devirle orantılıdır, gücü ise sabit kayıp ile devirle artan kısmın toplamıdır
            q_in += speed * v.flow_max_m3min
            power += v.rated_kw * (v.fixed_loss_frac + (1 - v.fixed_loss_frac) * speed) * press_factor
        load_frac.append(speed)

        # 3) tüketim fabrikanın talebi ile kaçakların toplamıdır ve basınç yükseldikçe artar
        p_ratio = np.sqrt((self.p + P_ATM) / (cfg.p_ref + P_ATM))
        q_out = (self.demand[self.step_i] + cfg.leak_m3min) * p_ratio  # tüketilen hava [m³/dk]

        # 4) basılan ile tüketilen hava arasındaki fark basıncı değiştirir
        p_prev = self.p
        self.p += P_ATM / cfg.volume_m3 * (q_in - q_out) / 60.0 * cfg.dt
        self.p = float(np.clip(self.p, 0.0, cfg.p_relief))  # emniyet ventili basıncı p_relief ile sınırlar
        self.dp = self.p - p_prev
        self.q_out = q_out
        self.q_hist.append(q_out)

        # 5) motor sıcaklıkları güncellenir, fazla ısınan motor trip olur
        for i in range(self.n_comp):
            target = cfg.ambient_c + cfg.temp_rise_c * load_frac[i]  # bu yükte ulaşılacak sıcaklık [°C]
            self.temp[i] += (target - self.temp[i]) * cfg.dt / cfg.temp_tau_s  # sıcaklık hedefe yavaşça yaklaşır
            self.lockout[i] = max(0.0, self.lockout[i] - cfg.dt)  # kilit süresi azalır
            if self.temp[i] >= cfg.trip_temp_c and self.lockout[i] == 0:
                trips += 1
                self.state[i] = OFF
                self.state_since[i] = self.t
                self.lockout[i] = cfg.trip_lockout_s

        # 6) ödüldeki her terim bir cezadır ve ajan toplam cezayı azaltmaya çalışır
        e_kwh = power * cfg.dt / 3600.0  # bu adımda harcanan enerji [kWh]
        e_scale = self.p_rated_total * cfg.dt / 3600.0  # hepsi tam güçte çalışsaydı harcanacak enerji [kWh]
        reward = -cfg.w_energy * e_kwh / e_scale  # enerji cezası, 0 ile -1 arasında
        if self.p < cfg.p_min:
            # 6 bar altı cezası basınç düştükçe artar, basınç 5 bar'ın altına indikten sonra sabit kalır. Sınır
            # konmadığında eğitimin başında ceza çok büyüyüp öğrenmeyi bozar.
            reward -= cfg.w_low_p * (1.0 + 5.0 * min(cfg.p_min - self.p, 1.0))
        if self.p < cfg.p_margin:
            # Güvenlik payı cezası 6,2 bar'da sıfırdır, 6,0 bar'da w_margin olur ve daha aşağıda sabit kalır.
            closeness = min((cfg.p_margin - self.p) / (cfg.p_margin - cfg.p_min), 1.0)
            reward -= cfg.w_margin * closeness
        reward -= (
            cfg.w_start * starts
            + cfg.w_load_switch * load_switches
            + cfg.w_excess_start * excess
            + cfg.w_trip * trips
        )

        # 7) bölüm toplamları güncellenir
        st = self.stats
        st["energy_kwh"] += e_kwh
        st["air_m3"] += q_in * cfg.dt / 60.0
        st["low_p_s"] += cfg.dt if self.p < cfg.p_min else 0.0
        st["margin_s"] += cfg.dt if self.p < cfg.p_margin else 0.0
        st["starts"] += starts
        st["load_switches"] += load_switches
        st["excess_starts"] += excess
        st["trips"] += trips
        st["p_sum"] += self.p

        self.step_i += 1
        self.t += cfg.dt
        truncated = self.step_i >= self.n_steps  # vardiya bitti mi
        # bu adımın ayrıntıları grafikler ve analiz için verilir
        info = dict(
            pressure=self.p,  # hat basıncı [bar]
            demand=q_out,  # tüketilen hava [m³/dk]
            supply=q_in,  # basılan hava [m³/dk]
            power_kw=power,  # çekilen güç [kW]
            state=list(self.state),  # kompresör durumları
            vsd_speed=speed,  # VSD devir oranı (0-1)
            hour=self._clock_hour(),  # fabrikadaki saat (0-24)
        )
        if truncated:
            info["episode_stats"] = self.episode_stats()
        # Gymnasium dönüş değerlerini gözlem, ödül, terminated, truncated ve info sırasıyla bekler. terminated hep
        # False döner, basınç düşse de bölüm bitirilmez ve yalnızca ceza verilir.
        return self._obs(), float(reward), False, truncated, info

    def episode_stats(self) -> dict:
        """Bölüm sonundaki özet değerleri döndürür. Raporlarda bu değerler kullanılır."""
        st = self.stats
        return dict(
            energy_kwh=st["energy_kwh"],  # toplam enerji [kWh]
            air_m3=st["air_m3"],  # toplam basılan hava [m³]
            # Özgül güç [kW/(m³/dk)] 1 m³/dk hava için harcanan gücü gösterir. Sektörde verimlilik bu değerle
            # ölçülür ve düşük olması iyidir.
            specific_power=st["energy_kwh"] / max(st["air_m3"], 1e-6) * 60.0,
            low_p_min=st["low_p_s"] / 60.0,  # 6 bar altında geçen süre [dk]
            margin_min=st["margin_s"] / 60.0,  # 6,2 bar altında geçen süre [dk]
            starts=st["starts"],  # toplam motor kalkışı
            load_switches=st["load_switches"],  # toplam yük-boş geçişi
            excess_starts=st["excess_starts"],  # limiti aşan kalkış sayısı
            trips=st["trips"],  # aşırı ısınmadan kapanma sayısı
            mean_p=st["p_sum"] / max(self.step_i, 1),  # ortalama basınç [bar]
        )
