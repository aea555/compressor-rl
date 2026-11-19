using UnityEngine;

// Bu sınıf sesleri ses dosyası kullanmadan kodla üretir. Motor uğultusu, hava sesi, boşa almadaki tahliye sesi,
// motor kalkışı, basınç 6 bar'ın altına inince çalan zil ve buton sesi burada tanımlıdır. Sesler kısık tutulur.
// Soldaki istasyonun sesi soldan, sağdakinin sesi sağdan gelir.
public class Sfx
{
    const int Rate = 44100;  // örnekleme hızı [Hz]
    static AudioClip[] hum;  // kompresörlerin uğultusu (her birinin tonu farklı)
    static AudioClip hiss, blowoff, start, alarm, click;
    static AudioSource ui;   // arayüz sesleri için
    static System.Random rng = new System.Random(7);

    public static float Master = 0.7f;  // genel ses seviyesi (0-1)
    public static bool Muted;

    readonly AudioSource[] humSrc = new AudioSource[3];
    readonly AudioSource hissSrc, oneShot;
    float lastEvent = -10f, lastAlarm = -10f;  // son olay ve alarm sesinin zamanı [sn], sesler çok sık çalmasın diye

    public static void Init(GameObject host)
    {
        hum = new[] { Hum(50f), Hum(62.5f), Hum(75f) };
        hiss = Loop(Noise(2.5f, 0.04f, 0.3f));  // hava akışının hışırtısı, yaklaşık 280 Hz üstü kısılır
        blowoff = Blowoff();
        start = Start();
        alarm = Chime();
        click = ClickClip();
        ui = host.AddComponent<AudioSource>();
        ui.playOnAwake = false;
    }

    public static void Click() { if (!Muted) ui.PlayOneShot(click, 0.25f * Master); }

    public Sfx(GameObject host, float pan)
    {
        for (int k = 0; k < 3; k++)
        {
            humSrc[k] = Source(host, hum[k], pan);
        }
        hissSrc = Source(host, hiss, pan);
        oneShot = host.AddComponent<AudioSource>();
        oneShot.panStereo = pan;
        oneShot.playOnAwake = false;
    }

    static AudioSource Source(GameObject host, AudioClip clip, float pan)
    {
        var s = host.AddComponent<AudioSource>();
        s.clip = clip;
        s.loop = true;
        s.volume = 0f;
        s.panStereo = pan;
        s.playOnAwake = false;
        s.time = (float)rng.NextDouble() * clip.length;  // döngüler üst üste binmesin diye rastgele bir yerden başlar
        s.Play();
        return s;
    }

    // Her karede çağrılır. state değerinde 0 kapalı, 1 boşta, 2 yükte anlamına gelir. vsdSpeed 0-1 arasındadır,
    // flow toplam hava akışıdır [m³/dk].
    public void UpdateLoops(int[] state, float vsdSpeed, float flow, float dt)
    {
        float m = Muted ? 0f : Master;
        for (int k = 0; k < 3; k++)
        {
            float vol, pitch;
            if (k == 2) { vol = vsdSpeed > 0 ? 0.05f + 0.12f * vsdSpeed : 0f; pitch = 0.55f + 0.45f * vsdSpeed; }
            else { vol = state[k] == 2 ? 0.15f : state[k] == 1 ? 0.07f : 0f; pitch = state[k] == 1 ? 0.93f : 1f; }
            var s = humSrc[k];
            s.volume = Mathf.MoveTowards(s.volume, vol * m, dt * 0.6f);   // ses yavaşça açılıp kapansın diye seviye adım adım değişir
            s.pitch = Mathf.MoveTowards(s.pitch, Mathf.Max(pitch, 0.5f), dt * 0.5f);
        }
        hissSrc.volume = Mathf.MoveTowards(hissSrc.volume, (0.02f + 0.05f * Mathf.Clamp01(flow / 30f)) * m, dt * 0.5f);
    }

    // Olay seslerini çalar. Oynatma hızı yüksekse sesler kısılır ve seyrekleşir.
    public void Blowoff(float playbackSpeed) => Event(blowoff, 0.22f, playbackSpeed);
    public void MotorStart(float playbackSpeed) => Event(start, 0.32f, playbackSpeed);

    void Event(AudioClip clip, float vol, float playbackSpeed)
    {
        if (Muted || Time.time - lastEvent < 0.35f) return;
        lastEvent = Time.time;
        float scale = playbackSpeed <= 120f ? 1f : playbackSpeed <= 300f ? 0.7f : 0.45f;
        oneShot.PlayOneShot(clip, vol * scale * Master);
    }

    public void Alarm()
    {
        if (Muted || Time.time - lastAlarm < 3f) return;
        lastAlarm = Time.time;
        oneShot.PlayOneShot(alarm, 0.3f * Master);
    }

    // Motor uğultusunu f0 frekansı, harmonikleri ve biraz gürültüyle üretir. 2 sn'lik döngü başa sarınca tık yapmaz.
    static AudioClip Hum(float f0)
    {
        int n = Rate * 2;  // f0'lar 0,5 Hz'in katı olduğu için 2 sn tam periyoda denk gelir
        float[] noise = Noise(2f, 0.02f, 1f);
        var data = new float[n];
        float[] amp = { 0.55f, 0.3f, 0.14f, 0.06f };
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate, v = 0f;
            for (int h = 0; h < amp.Length; h++) v += amp[h] * Mathf.Sin(2 * Mathf.PI * f0 * (h + 1) * t);
            v *= 0.85f + 0.15f * Mathf.Sin(2 * Mathf.PI * 1f * t);  // hafif dalgalanma
            data[i] = 0.35f * v + 0.25f * noise[i];
        }
        return Clip("hum", data);
    }

    // Başa sarılabilen bir gürültü üretir. lp küçüldükçe ses boğuklaşır.
    static float[] Noise(float seconds, float lp, float gain)
    {
        int n = (int)(Rate * seconds), fade = Rate / 4;
        var raw = new float[n + fade];
        float y1 = 0f, y = 0f, prev = 0f;
        for (int i = 0; i < raw.Length; i++)
        {
            // Tek süzgeçte tiz kısım fazla kaldığı için gürültü iki kez süzülür.
            y1 += lp * ((float)rng.NextDouble() * 2f - 1f - y1);
            y += lp * (y1 - y);
            float hp = y - prev * 0.995f;  // çok alçak titreşimler atılır
            prev = y;
            raw[i] = hp;
        }
        float max = 1e-6f;
        foreach (var v in raw) max = Mathf.Max(max, Mathf.Abs(v));
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = raw[i] / max * gain;
        for (int i = 0; i < fade; i++)  // sonu başına yumuşakça bağlanır
        {
            float a = (float)i / fade;
            data[i] = data[i] * a + raw[n + i] / max * gain * (1 - a);
        }
        return data;
    }

    static AudioClip Loop(float[] data) => Clip("loop", data);

    // Tahliye sesi için hızlı başlayıp yavaşça sönen bir gürültü üretir
    static AudioClip Blowoff()
    {
        int n = Rate;
        var data = new float[n];
        float y = 0f;
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            float lp = Mathf.Lerp(0.35f, 0.04f, t);
            y += lp * ((float)rng.NextDouble() * 2f - 1f - y);
            float env = Mathf.Clamp01(t / 0.03f) * Mathf.Exp(-t * 3.5f);
            data[i] = y * env * 0.9f;
        }
        return Clip("blowoff", data);
    }

    // Motor kalkışı için kısa bir tık, alçak bir ses ve yükselen bir vınlama üretir
    static AudioClip Start()
    {
        int n = (int)(Rate * 0.9f);
        var data = new float[n];
        float y = 0f, phase = 0f;
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            y += 0.3f * ((float)rng.NextDouble() * 2f - 1f - y);
            float clickEnv = Mathf.Exp(-t * 300f);
            float thump = Mathf.Sin(2 * Mathf.PI * 48f * t) * Mathf.Exp(-t * 9f);
            phase += 2 * Mathf.PI * Mathf.Lerp(70f, 150f, Mathf.Clamp01(t / 0.8f)) / Rate;
            float whine = Mathf.Sin(phase) * 0.15f * Mathf.Clamp01(t / 0.1f) * Mathf.Exp(-t * 2.5f);
            data[i] = 0.3f * y * clickEnv + 0.6f * thump + whine;
        }
        return Clip("start", data);
    }

    // Mi ve La notalarından oluşan yumuşak bir zil sesi üretir
    static AudioClip Chime()
    {
        int n = (int)(Rate * 1.6f);
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            data[i] = Note(t, 0f, 659.3f) + Note(t, 0.35f, 880f);
        }
        return Clip("alarm", data);
    }

    static float Note(float t, float t0, float f)
    {
        if (t < t0) return 0f;
        float u = t - t0, env = Mathf.Clamp01(u / 0.01f) * Mathf.Exp(-u * 3f);
        return 0.3f * env * (Mathf.Sin(2 * Mathf.PI * f * u) + 0.25f * Mathf.Sin(4 * Mathf.PI * f * u));
    }

    static AudioClip ClickClip()
    {
        int n = Rate / 30;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            float t = (float)i / Rate;
            data[i] = 0.5f * Mathf.Sin(2 * Mathf.PI * 1500f * t) * Mathf.Exp(-t * 180f);
        }
        return Clip("click", data);
    }

    static AudioClip Clip(string name, float[] data)
    {
        var c = AudioClip.Create(name, data.Length, 1, Rate, false);
        c.SetData(data, 0);
        return c;
    }
}
