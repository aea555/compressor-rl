"""Bu dosya PPO ajanını eğitir.

PPO (Proximal Policy Optimization) bir derin RL yöntemidir. Eğitim için Stable-Baselines3 kütüphanesi kullanılır.
Ajan her adımda K1, K2 ve VSD için ayrı ayrı seçim yaptığı için komut MultiDiscrete tipindedir. K1 ve K2 sabit
hızlı kompresörlerdir, VSD (değişken hızlı sürücü) kompresörünün ise devri ayarlanabilir.

    uv run python train_ppo.py --timesteps 2000000 --n-envs 8   # ajanı eğitir
    uv run tensorboard --logdir runs                            # eğitim eğrilerini gösterir

ONERILEN_DEGISIKLIKLER.md'deki aşamalar aşağıdaki ayarlarla eğitilmiştir.

    A   --w-start 1.0
    B   --w-start 1.0 --extra-obs
    C   --w-start 1.0 --extra-obs --normalize-reward --lr-decay
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from compressor_env import CompressorEnv, EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)  # toplam eğitim adımı, her adım 10 sn
    ap.add_argument("--n-envs", type=int, default=8)  # aynı anda çalışan ortam sayısı, eğitimi hızlandırır
    ap.add_argument("--seed", type=int, default=0)  # aynı seed aynı sonucu verir
    ap.add_argument("--out", type=Path, default=Path("runs/ppo"))  # model ve kayıtların yazılacağı klasör
    # modelin kaç adımda bir ara kopyasının (checkpoint) kaydedileceği
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    # Aşağıdaki seçenekler aşamalarda eklenmiştir. Verilmezlerse eski davranış korunur.
    ap.add_argument("--w-start", type=float, default=None)  # değişiklik 1, motor kalkışı cezası (varsayılan 0.3)
    ap.add_argument("--extra-obs", action="store_true")  # değişiklik 2, gözleme talep eğilimi ve kompresörlerin aynı durumda geçirdiği süre eklenir
    ap.add_argument("--normalize-reward", action="store_true")  # değişiklik 3a, ödüller otomatik ölçeklenir
    ap.add_argument("--lr-decay", action="store_true")  # değişiklik 3b, öğrenme hızı eğitimin sonuna doğru sıfıra iner
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Varsayılandan farklı ortam ayarları env_config.json dosyasına da yazılır. Böylece evaluate.py modeli
    # doğru ortamla açabilir.
    overrides = {}
    if args.w_start is not None:
        overrides["w_start"] = args.w_start
    if args.extra_obs:
        overrides["extra_obs"] = True
    (args.out / "env_config.json").write_text(json.dumps(overrides, indent=2))
    # hangi klasörün hangi ayarla eğitildiği sonradan anlaşılsın diye bütün eğitim ayarları da kaydedilir
    (args.out / "train_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    env_kwargs = dict(config=replace(EnvConfig(), **overrides))

    # eğitim ortamlarının her biri ayrı bir süreçte çalışır
    env = make_vec_env(CompressorEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    # eğitim sırasında ajanı denemek için ayrı bir ortam kurulur, bu ortam farklı seed ile başka günler görür
    eval_env = make_vec_env(CompressorEnv, n_envs=1, seed=args.seed + 1000, env_kwargs=env_kwargs)
    if args.normalize_reward:
        # Ödüller o ana kadar görülen ödüllerin büyüklüğüne bölünüp 1 civarına getirilir. Gözlemler ortamda
        # zaten ölçekli olduğu için onlara dokunulmaz (norm_obs=False). Böylece kaydedilen model test edilirken
        # ek bir dosyaya gerek kalmaz.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.995)
        # Kütüphane test ortamının da aynı sarmalayıcıda olmasını ister. Burada ödül ölçeklenmez ve test
        # ödülleri gerçek değerleriyle raporlanır.
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)

    # Öğrenme hızı ya sabit 3e-4 olur ya da eğitim boyunca 3e-4'ten 0'a doğrusal iner. progress_remaining
    # eğitimin kalan kısmını gösterir, başta 1.0, sonda 0.0 olur.
    learning_rate = (lambda progress_remaining: 3e-4 * progress_remaining) if args.lr_decay else 3e-4

    model = PPO(
        "MlpPolicy",  # tam bağlantılı (MLP) sinir ağı
        env,
        n_steps=1024,  # bir güncellemeden önce her ortamdan toplanan adım sayısı
        batch_size=512,  # her gradyan adımında kullanılan örnek sayısı
        n_epochs=10,  # toplanan veriden kaç tur geçileceği
        learning_rate=learning_rate,  # öğrenme hızı
        gamma=0.995,  # gelecekteki ödüllerin ağırlığı, ajan 10 sn'lik adımlarla yaklaşık 200 adım (33 dk) ileriye bakar
        gae_lambda=0.95,  # avantaj tahmininde yanlılık (bias) ile varyans arasındaki dengeyi ayarlar
        ent_coef=0.01,  # ajan erkenden tek bir davranışa takılmasın diye keşfi teşvik eder
        clip_range=0.2,  # ajanın davranışının bir güncellemede en fazla ne kadar değişebileceği
        # İki ayrı ağ bulunur. pi komutu seçer, vf durumun ne kadar iyi olduğunu tahmin eder.
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(args.out / "tb"),  # eğitim eğrileri buraya yazılır
        seed=args.seed,
        verbose=1,
    )
    # Yaklaşık her 50 bin adımda ajan 15 günde denenir ve en iyisi best_model.zip olarak saklanır. Önceden
    # 5 gün kullanılmıştır ama sonuç çok oynadığı için şansa iyi geçen bir test kötü bir modeli seçtirebilirdi.
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=15,
        eval_freq=max(50_000 // args.n_envs, 1),
        best_model_save_path=str(args.out),
        log_path=str(args.out),
        deterministic=True,
    )
    # Her checkpoint_every adımda model checkpoints/ppo_<adım>_steps.zip olarak ayrıca kaydedilir.
    # best_model yalnızca toplam ödüle bakar ve her seferinde üzerine yazılır. Ara kopyalar sayesinde
    # eğitimden sonra modeller enerji ve kalkış sayısına göre de karşılaştırılabilir.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),  # sayaç her ortamda ayrı saydığı için n_envs'e bölünür
        save_path=str(args.out / "checkpoints"),
        name_prefix="ppo",
    )
    model.learn(total_timesteps=args.timesteps, callback=[eval_cb, checkpoint_cb], progress_bar=True)
    model.save(args.out / "final_model")
    print(f"Kaydedildi: {args.out / 'final_model.zip'}, en iyisi: {args.out / 'best_model.zip'}")
    print(f"Ara modeller: {args.out / 'checkpoints'}")


if __name__ == "__main__":
    main()
