"""Bu dosyada klasik kaskad kontrol uygulanır. RL modelleri bu kontrolcüyle karşılaştırılır.

Kaskad kontrol fabrikalarda bugün de kullanılır ve sabit kurallarla çalışır. İçinde öğrenme yoktur.

Sistemde üç kompresör bulunur. K1 (75 kW) ve K2 (45 kW) sabit hızlıdır. VSD (değişken hızlı sürücü, 55 kW)
kompresörünün ise devri ayarlanabilir. Sabit kompresörler sırayla devreye girer. Basınç alt eşiğin altına
düşünce kompresör yüke alınır, üst eşiği geçince boşa alınır. 5 dakika boşta kalan motor durdurulur.

İnce ayarı VSD yapar. Devrini bir PI kontrolcü ayarlar ve basıncı 6,9 bar civarında tutmaya çalışır.
PI kontrolcü hedef basınçla ölçülen basınç arasındaki farka (P, oransal kısım) ve bu farkın zamanla
birikmesine (I, integral kısım) bakar.
"""

import numpy as np

from .env import LOAD, OFF, UNLOAD, CompressorEnv


class CascadeController:
    def __init__(
        self,
        # Her sabit kompresörün yüke alma ve boşa alma eşikleri [bar] burada verilir.
        # İlk satır K1'e, ikinci satır K2'ye aittir. K2'nin eşikleri daha düşük olduğu için önce K1 devreye girer.
        bands=((6.5, 7.3), (6.3, 7.1)),
        vsd_setpoint: float = 6.9,  # VSD'nin tutmaya çalıştığı basınç [bar]
        kp: float = 0.6,  # PI oransal kazancı, basınç farkı değişince devrin ne kadar değişeceği [1/bar]
        ki: float = 0.004,  # PI integral kazancı, fark sürdükçe devrin ne hızla değişeceği [1/(bar*s)]
        idle_off_s: float = 300.0,  # bu kadar boşta kalan kompresör kapatılır [s]
    ):
        self.bands = bands
        self.sp = vsd_setpoint
        self.kp = kp
        self.ki = ki
        self.idle_off_s = idle_off_s

    def reset(self, env: CompressorEnv):
        """Her bölümün başında kontrolcünün hafızasını sıfırlar."""
        self.idle = [0.0] * env.n_fixed  # her sabit kompresörün ne zamandır boşta olduğu [s]
        self.speed_cmd = env.vsd_speed(env.state[env.n_fixed])  # PI'ın hesapladığı VSD devri (0-1)
        self.e_prev = self.sp - env.p  # önceki adımdaki basınç farkı, hedef eksi ölçülen [bar]
        self.vsd_on = env.state[env.n_fixed] > 0  # VSD açık mı

    def act(self, env: CompressorEnv) -> np.ndarray:
        """Ortamdaki basıncı ve kompresör durumlarını okuyup komut üretir."""
        p, dt = env.p, env.cfg.dt  # şu anki basınç [bar] ve adım süresi [s]
        action = []
        # sabit kompresörler basınç eşiklerine göre yüke ya da boşa alınır, gerekirse kapatılır
        for i, (p_on, p_off) in enumerate(self.bands):
            s = env.state[i]  # kompresörün şu anki durumu
            if p < p_on:
                cmd = LOAD  # basınç düşük olduğu için yüke alınır
            elif p > p_off and s != OFF:
                cmd = UNLOAD  # basınç yüksek olduğu için boşa alınır
            else:
                cmd = s  # basınç iki eşiğin arasında olduğu için durum değişmez
            if cmd == UNLOAD:
                self.idle[i] += dt
                if self.idle[i] >= self.idle_off_s:
                    cmd = OFF  # uzun süre boşta kalan kompresör boşa enerji harcamasın diye kapatılır
            else:
                self.idle[i] = 0.0
            action.append(cmd)

        # VSD devri PI kontrolle hesaplanır
        e = self.sp - p  # basınç farkı [bar], pozitif olduğunda basınç hedefin altındadır ve devir artırılır
        self.speed_cmd += self.kp * (e - self.e_prev) + self.ki * e * dt
        self.speed_cmd = float(np.clip(self.speed_cmd, 0.0, 1.0))  # devir 0 ile 1 arasında tutulur
        self.e_prev = e
        # VSD'nin durma ve başlama eşikleri farklı tutulduğu için motor sürekli açılıp kapanmaz. PI devri en düşük
        # devrin yarısının altına inerse ve basınç hedefin 0,25 bar üstündeyse VSD durur. Basınç hedefin altına
        # inince tekrar başlar.
        min_speed = env.cfg.vsd.min_speed
        if self.vsd_on and self.speed_cmd < min_speed * 0.5 and p > self.sp + 0.25:
            self.vsd_on = False
        elif not self.vsd_on and p < self.sp:
            self.vsd_on = True
            self.speed_cmd = max(self.speed_cmd, min_speed)
        action.append(self._speed_to_level(env, self.speed_cmd) if self.vsd_on else 0)
        return np.asarray(action, dtype=np.int64)

    @staticmethod
    def _speed_to_level(env: CompressorEnv, speed: float) -> int:
        """Devir oranını (0-1) en yakın VSD kademesine (1-n_levels) yuvarlar."""
        speeds = np.array([env.vsd_speed(l) for l in range(1, env.cfg.vsd.n_levels + 1)])
        return int(np.argmin(np.abs(speeds - speed))) + 1
