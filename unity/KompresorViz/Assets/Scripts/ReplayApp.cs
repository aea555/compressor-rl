using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering.Universal;

// Uygulamanın ana scripti budur. Kayıtları okur, sahneyi kurar, zamanı ve ışığı ilerletir, arayüzü çizer.
// Rehber penceresi ReplayApp.Guide.cs dosyasında, fareyle üstüne gelince çıkan açıklamalar ReplayApp.Hints.cs dosyasında bulunur.
//
// Boşluk tuşu oynatmayı durdurur ya da sürdürür, sol ve sağ ok hızı değiştirir, R baştan başlatır, Tab özeti, G rehberi açar, M sesi kapatır.
// Ekran görüntüsü ve video kaydı için aşağıdaki komut satırı seçenekleri kullanılır.
//   -model ppo_w3 -scenario gece    açılışta bu kaydı seçer
//   -summary, -guide veya -terms    özet, rehber ya da terimler sayfası açık başlar
//   -hintat 1320,280                fare bu noktadaymış gibi açıklamayı gösterir [1080p piksel]
//   -shots 1,3.5,6 -shotdir d       bölümün bu saatlerinde ekran görüntüsü alıp kapanır
//   -record 0.5,25 -shotdir d       0,5. saatten başlayarak 25 sn video karesi kaydedip kapanır
public partial class ReplayApp : MonoBehaviour
{
    ReplayIndex index;               // hangi model ve senaryoların kaydı var
    int modelIdx, scenIdx;           // seçili model ve senaryonun sırası
    ReplayData data;                 // şu an oynatılan kayıt
    StationView[] stations;          // 0 baseline, 1 ajan
    Camera[] cams;                   // istasyonların kameraları, ekranın sol ve sağ yarısı
    Vector3[] camBase;               // kameraların asıl konumu [m], hafifçe bunun etrafında sallanırlar
    Light sun;                       // güneş ya da ay ışığı
    float t;                         // şu anki zaman [adım], ondalıklı olabilir
    float speed = 120f;              // oynatma hızı, gerçek 1 saniyede geçen simülasyon süresi [sn]
    bool paused;                     // durduruldu mu?
    bool showSummary;                // özet ekranı açık mı?
    bool summaryAutoShown;           // bölüm bitince özet bir kez kendiliğinden açıldı mı?
    List<string> insights = new();   // özet ekranında modelin ne yaptığını anlatan cümleler
    float ui;                        // arayüz ölçeği, 1080p ekranda 1

    static readonly Vector3 CamOffset = new Vector3(1.15f, 11.5f, -18.5f);  // kameranın istasyona göre konumu [m]
    static readonly Vector3 CamTarget = new Vector3(1.15f, 3.2f, 0.5f);     // kameranın baktığı nokta [m]
    static readonly float[] Speeds = { 30f, 60f, 120f, 300f, 600f, 1200f };  // seçilebilecek hızlar
    string ReplayDir => Path.Combine(Application.streamingAssetsPath, "replays");

    void Start()
    {
        // Ekrandaki ondalık sayılar Türkçedeki gibi virgülle yazılır, diğer biçimler değişmez.
        var culture = (CultureInfo)CultureInfo.InvariantCulture.Clone();
        culture.NumberFormat.NumberDecimalSeparator = ",";
        CultureInfo.CurrentCulture = CultureInfo.DefaultThreadCurrentCulture = culture;
        string shots = null, shotDir = ".", model = null, scen = null, record = null;
        var args = System.Environment.GetCommandLineArgs();
        for (int k = 0; k < args.Length; k++)
        {
            string next = k + 1 < args.Length ? args[k + 1] : null;
            if (args[k] == "-shots") shots = next;
            if (args[k] == "-record") record = next;  // "başlangıç saati,süre [sn]", örneğin "0.5,25"
            if (args[k] == "-shotdir") shotDir = next;
            if (args[k] == "-model") model = next;
            if (args[k] == "-scenario") scen = next;
            if (args[k] == "-summary") showSummary = summaryAutoShown = true;
            if (args[k] == "-guide") showGuide = summaryAutoShown = true;
            if (args[k] == "-terms") { showGuide = summaryAutoShown = true; guidePage = 1; }
            if (args[k] == "-hintat" && next != null) { var xy = next.Split(','); hintAt = new Vector2(float.Parse(xy[0], CultureInfo.InvariantCulture), float.Parse(xy[1], CultureInfo.InvariantCulture)); }
        }
        index = JsonUtility.FromJson<ReplayIndex>(File.ReadAllText(Path.Combine(ReplayDir, "index.json")));
        modelIdx = Mathf.Max(0, System.Array.FindIndex(index.models, m => m.key == model));
        scenIdx = Mathf.Max(0, System.Array.FindIndex(index.scenarios, s => s.key == scen));
        LoadReplay();

        Look.Init();
        Sfx.Init(gameObject);
        stations = new[] {
            new StationView(data, data.runs[0], new Vector3(-16f, 0, 0), new Color(0.62f, 0.64f, 0.68f), new Sfx(gameObject, -0.7f)),
            new StationView(data, data.runs[1], new Vector3(16f, 0, 0), new Color(0.6f, 0.66f, 0.66f), new Sfx(gameObject, 0.7f)),
        };

        // Ekran ikiye bölünür ve her istasyonu kendi kamerası karşıdan gösterir.
        cams = new Camera[2];
        camBase = new Vector3[2];
        for (int k = 0; k < 2; k++)
        {
            var cam = k == 0 ? Camera.main : new GameObject("Kamera2").AddComponent<Camera>();
            Vector3 o = stations[k].Origin;
            cam.transform.position = o + CamOffset;
            cam.transform.LookAt(o + CamTarget);
            camBase[k] = cam.transform.position;
            cam.fieldOfView = 50f;
            cam.rect = new Rect(k * 0.5f, 0f, 0.5f, 1f);  // ekranın sol ya da sağ yarısı
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Theme.Bg;
            var urp = cam.GetUniversalAdditionalCameraData();
            urp.renderPostProcessing = true;  // parlama, ton eşleme ve vinyet için
            urp.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
            urp.antialiasingQuality = AntialiasingQuality.High;
            cams[k] = cam;
        }
        sun = new GameObject("Güneş").AddComponent<Light>();
        sun.type = LightType.Directional;
        sun.shadows = LightShadows.Soft;
        sun.shadowStrength = 0.85f;
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;

        if (shots != null) { Sfx.Muted = true; StartCoroutine(TakeShots(shots, shotDir)); }
        if (record != null) { Sfx.Muted = true; StartCoroutine(Record(record, shotDir)); }
    }

    // Seçili model ve senaryonun kaydını okuyup istasyonlara verir
    void LoadReplay()
    {
        string m = index.models[modelIdx].key, s = index.scenarios[scenIdx].key;
        var entry = System.Array.Find(index.files, e => e.model == m && e.scenario == s);
        data = JsonUtility.FromJson<ReplayData>(File.ReadAllText(Path.Combine(ReplayDir, entry.file)));
        if (stations != null)
            for (int k = 0; k < 2; k++) stations[k].SetRun(data, data.runs[k]);
        t = 0f;
        summaryAutoShown = showSummary;
        BuildInsights();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space)) paused = !paused;
        if (Input.GetKeyDown(KeyCode.R)) Restart();
        if (Input.GetKeyDown(KeyCode.Tab)) { showSummary = !showSummary; showGuide = false; }
        if (Input.GetKeyDown(KeyCode.G)) { showGuide = !showGuide; showSummary = false; }
        if (Input.GetKeyDown(KeyCode.M)) Sfx.Muted = !Sfx.Muted;
        if (Input.GetKeyDown(KeyCode.RightArrow)) ChangeSpeed(+1);
        if (Input.GetKeyDown(KeyCode.LeftArrow)) ChangeSpeed(-1);

        if (!paused) t = Mathf.Min(t + Time.deltaTime * speed / data.dt, data.Steps - 1);
        if (t >= data.Steps - 1 && !summaryAutoShown && !showGuide) showSummary = summaryAutoShown = true;  // bölüm bitti
        foreach (var s in stations) s.UpdateView(t, Time.deltaTime, paused ? 0f : speed);
        UpdateLighting();

        // Sahne durağan görünmesin diye kameralar çok yavaş sallanır.
        float tt = Time.time;
        for (int k = 0; k < 2; k++)
        {
            cams[k].transform.position = camBase[k] + new Vector3(Mathf.Sin(tt * 0.11f) * 0.35f, Mathf.Sin(tt * 0.07f) * 0.15f, 0f);
            cams[k].transform.LookAt(stations[k].Origin + CamTarget);
        }
    }

    void Restart() { t = 0f; summaryAutoShown = false; showSummary = false; }

    void ChangeSpeed(int dir)
    {
        int si = System.Array.IndexOf(Speeds, speed);
        speed = Speeds[Mathf.Clamp(si + dir, 0, Speeds.Length - 1)];
    }

    // Işığı fabrika saatine göre ayarlar. Gündüz sarımsı güneş, gece mavimsi ay ışığı kullanılır.
    void UpdateLighting()
    {
        int i = Mathf.Clamp(Mathf.FloorToInt(t), 0, data.Steps - 1);
        float h = data.runs[0].hour[i];                       // fabrika saati (0-24)
        float elev = Mathf.Sin((h - 6f) / 12f * Mathf.PI);   // güneşin yüksekliği, 6'da ve 18'de 0, 12'de 1
        float day = Mathf.Clamp01(elev * 1.6f + 0.15f);      // 0 gece, 1 gündüz
        float dusk = Mathf.Clamp01(1f - Mathf.Abs(elev) * 3f) * day;  // sabah ve akşam turuncusunun oranı (0-1)
        sun.intensity = Mathf.Lerp(0.25f, 1.7f, day);
        sun.color = Color.Lerp(Color.Lerp(new Color(0.55f, 0.65f, 1f), new Color(1f, 0.96f, 0.9f), day), new Color(1f, 0.6f, 0.35f), dusk * 0.6f);
        sun.transform.rotation = Quaternion.Euler(Mathf.Lerp(35f, 58f, day), -35f + (h - 12f) * 6f, 0f);
        RenderSettings.ambientSkyColor = Color.Lerp(new Color(0.16f, 0.19f, 0.3f), new Color(0.55f, 0.6f, 0.7f), day);
        RenderSettings.ambientEquatorColor = Color.Lerp(new Color(0.12f, 0.13f, 0.18f), new Color(0.42f, 0.42f, 0.44f), day);
        RenderSettings.ambientGroundColor = Color.Lerp(new Color(0.06f, 0.06f, 0.08f), new Color(0.2f, 0.19f, 0.18f), day);
        foreach (var s in stations) s.SetLighting(1f - day);
    }

    // Video için oynatmayı saniyede 30 kare olarak kaydeder ve uygulamayı kapatır.
    // Time.captureFramerate her karede zamanı 1/30 sn ilerlettiği için kayıt bilgisayarın hızından etkilenmez.
    IEnumerator Record(string spec, string dir)
    {
        Directory.CreateDirectory(dir);
        var p = spec.Split(',');
        float hour = float.Parse(p[0], CultureInfo.InvariantCulture);   // kaydın başladığı an, bölüm başından [saat]
        float secs = float.Parse(p[1], CultureInfo.InvariantCulture);   // kayıt süresi [sn]
        for (int k = 0; k < 60; k++) yield return null;  // pencere ve arayüz otursun diye 60 kare beklenir
        Time.captureFramerate = 30;
        paused = false;
        t = Mathf.Min(hour * 3600f / data.dt, data.Steps - 1);
        int frames = Mathf.RoundToInt(secs * 30f);
        for (int f = 0; f < frames; f++)
        {
            yield return new WaitForEndOfFrame();
            var tex = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(Path.Combine(dir, $"f_{f:00000}.jpg"), tex.EncodeToJPG(92));
            Destroy(tex);
        }
        Application.Quit();
    }

    // Komut satırında verilen saatlerde ekran görüntüsü alıp uygulamayı kapatır
    IEnumerator TakeShots(string shots, string dir)
    {
        Directory.CreateDirectory(dir);
        foreach (var h in shots.Split(','))
        {
            float hour = float.Parse(h, CultureInfo.InvariantCulture);
            float target = Mathf.Min(hour * 3600f / data.dt, data.Steps - 1);  // istenen an [adım]
            paused = false;
            t = Mathf.Max(0f, target - 12f);  // duman ve buhar oluşsun diye oynatma 12 adım önceden başlar
            for (int k = 0; k < 90; k++) yield return null;
            paused = true;
            t = target;
            for (int k = 0; k < 5; k++) yield return null;
            yield return new WaitForEndOfFrame();
            var tex = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(Path.Combine(dir, $"shot_{h}.png"), tex.EncodeToPNG());
            Debug.Log($"Ekran görüntüsü: saat {h}");
            Destroy(tex);
        }
        Application.Quit();
    }

    // Bölümün tamamına bakıp modelin baseline'dan farkını birkaç cümleyle anlatır.
    // Cümleler seçili kayda göre değişir.
    void BuildInsights()
    {
        insights.Clear();
        ReplayRun b = data.runs[0], a = data.runs[1];
        int n = data.Steps - 1;
        string good = Theme.Hex(Theme.Good), bad = Theme.Hex(Theme.Bad);
        float de = b.energy[n] - a.energy[n];  // toplam tasarruf [kWh] (negatifse ajan daha çok harcadı)
        insights.Add(de >= 0
            ? $"Bu vardiyada baseline'dan <color=#{good}>%{de / b.energy[n] * 100:0.0} daha az</color> enerji harcadı, yani {de:0} kWh."
            : $"Bu vardiyada baseline'dan <color=#{bad}>%{-de / b.energy[n] * 100:0.0} daha fazla</color> enerji harcadı, yani {-de:0} kWh.");

        float dp = b.sumMeanP - a.sumMeanP;  // ortalama basınç farkı [bar]
        float sp = b.ePressure[n] - a.ePressure[n];  // basınç kaleminden tasarruf [kWh]
        if (Mathf.Abs(dp) > 0.1f)
            insights.Add(dp > 0
                ? $"Basıncı baseline'dan ortalama {dp:0.00} bar düşük tutuyor ({a.sumMeanP:0.00} bar). Basınç düşük olunca kompresörler daha az güç çekiyor ve kaçak azalıyor. Bundan gelen tasarruf <color=#{good}>{sp:0} kWh</color>."
                : $"Basıncı baseline'dan ortalama {-dp:0.00} bar yüksek tutuyor, bu da <color=#{bad}>{-sp:0} kWh</color> fazladan enerji demek.");

        float su = b.eUnload[n] - a.eUnload[n];  // boşta çalışma kaleminden tasarruf [kWh]
        if ((su > 5f && a.eUnload[n] < 0.5f * b.eUnload[n]) || su < -5f)
            insights.Add(su > 0
                ? $"Gerekmeyen kompresörü boşta bekletmek yerine kapatıyor. Boşta çalışmaya baseline {b.eUnload[n]:0} kWh, model {a.eUnload[n]:0} kWh harcadı."
                : $"Kompresörleri baseline'dan daha çok boşta çalıştırıyor. Boşta çalışmaya model {a.eUnload[n]:0} kWh, baseline {b.eUnload[n]:0} kWh harcadı.");

        // Her kompresörün yükte geçirdiği süre oranı karşılaştırılır.
        float bk1 = Share(b.k1), ak1 = Share(a.k1), bk2 = Share(b.k2), ak2 = Share(a.k2);
        if (ak2 - bk2 > 0.1f && ak1 < bk1)
            insights.Add($"Küçük K2'yi daha çok, büyük K1'i daha az çalıştırıyor. Yükte geçen süre K2 için %{ak2 * 100:0} (baseline %{bk2 * 100:0}), K1 için %{ak1 * 100:0} (baseline %{bk1 * 100:0}). Talep düşükken küçük kompresörü seçiyor.");
        else if (ak2 - bk2 > 0.1f)
            insights.Add($"K2'yi baseline'dan daha sık kullanıyor. K2'nin yükte geçen süresi %{ak2 * 100:0}, baseline'da %{bk2 * 100:0}.");
        float spr = b.eProduction[n] - a.eProduction[n];  // hangi makinelerin çalıştığı kaleminden tasarruf [kWh]
        if (Mathf.Abs(spr) > 3f)
            insights.Add(spr > 0
                ? $"Aynı havayı daha verimli kompresörlerle üretiyor ve bundan <color=#{good}>{spr:0} kWh</color> kazanıyor."
                : $"Seçtiği kompresörler baseline'ınkinden biraz daha az verimli, bu <color=#{bad}>{-spr:0} kWh</color> fazlaya mal oluyor.");

        int flick = 0;  // 30 sn içinde geri alınan karar sayısı [adet]
        foreach (var w in a.evWhy) if (IsFlicker(w)) flick++;
        int bs = b.starts[n], as_ = a.starts[n];
        string startTxt = as_ <= bs
            ? $"Motorları <color=#{good}>{as_}</color> kez çalıştırdı, baseline {bs} kez. Motorlar daha az yıpranıyor."
            : $"Motorları <color=#{bad}>{as_}</color> kez çalıştırdı, baseline {bs} kez. Tasarrufun karşılığında motorlar daha çok yıpranıyor.";
        if (flick > 3) startTxt += $" <color=#{bad}>{flick} kararını 30 sn içinde geri aldı</color>, yani kararsız davranıyor.";
        if (a.sumExcess > 0) startTxt += $" Saatlik kalkış sınırını {a.sumExcess:0} kez aştı.";
        insights.Add(startTxt);

        float bl = b.lowMin[n], al = a.lowMin[n];  // 6 bar altı süre [dk]
        insights.Add(al <= bl + 0.5f
            ? (bl > 0 ? $"Basınç {al:0} dk boyunca 6 bar'ın altında kaldı, baseline'da bu süre {bl:0} dk." : "Basınç hiç 6 bar'ın altına inmedi, üretim aksamadı.")
            : $"<color=#{bad}>Basınç {al:0} dk boyunca 6 bar'ın altında kaldı</color>, baseline'da {bl:0} dk. Basıncı düşük tutmanın riski bu.");
    }

    static bool IsFlicker(string why) => why.StartsWith("30 sn içinde geri alındı");  // karar günlüğündeki satırın titreme olup olmadığını döndürür

    static float Share(int[] states)  // yükte geçen sürenin oranını döndürür (0-1)
    {
        int c = 0;
        foreach (var s in states) if (s == 2) c++;
        return (float)c / states.Length;
    }

    // Arayüz Unity'nin IMGUI sistemiyle her karede baştan çizilir.
    float U(float v) => v * ui;  // 1080p için verilen ölçüyü ekran boyuna çevirir [piksel]
    int Fs(float v) => Mathf.RoundToInt(v * ui);  // yazı boyutu [piksel]
    static string Clock(float h) => $"{(int)h:00}:{(int)(h % 1 * 60):00}";
    static string State(int s) => s == 2 ? "yükte" : s == 1 ? "boşta" : "kapalı";

    bool Button(Rect r, string text, bool on = false)
    {
        bool hit = GUI.Button(r, text, on ? Theme.BtnOn : Theme.Btn);
        if (hit) Sfx.Click();
        return hit;
    }

    // Bir başlık ve yan yana butonlar çizer. Tıklanan butonun sırasını döndürür, tıklanmadıysa -1.
    int Segment(ref float x, float y, string caption, string[] labels, int selected, string[] tips = null)
    {
        GUI.Label(new Rect(x, y + U(10), U(80), U(20)), caption, Theme.Caps);
        x += Theme.Caps.CalcSize(new GUIContent(caption)).x + U(10);
        int clicked = -1;
        for (int k = 0; k < labels.Length; k++)
        {
            float w = Theme.Btn.CalcSize(new GUIContent(labels[k])).x + U(6);
            var br = new Rect(x, y, w, U(32));
            if (Button(br, labels[k], k == selected)) clicked = k;
            if (tips != null && tips[k] != null) Hint(br, tips[k], true);
            x += w + U(5);
        }
        x += U(22);
        return clicked;
    }

    void OnGUI()
    {
        if (data == null || stations == null) return;
        ui = Screen.height / 1080f;
        Theme.Ensure(ui);
        int i = Mathf.Clamp(Mathf.FloorToInt(t), 0, data.Steps - 1);
        float W = Screen.width, H = Screen.height;

        DrawHeader(i, W);
        Theme.Rect(new Rect(W / 2 - U(1), U(113), U(2), H), new Color(1f, 1f, 1f, 0.12f));  // iki yarıyı ayıran çizgi
        for (int s = 0; s < 2; s++) DrawStation(stations[s], s, i);
        DrawDifference(i);
        DrawSavings(i);
        DrawTimeline(i);
        if (showSummary) DrawSummary();
        if (showGuide) DrawGuide();
        DrawHint();
    }

    // Üst çubuğu çizer. Çubukta başlık, saat ve oynatma butonları, altında da model ve senaryo seçimi bulunur.
    void DrawHeader(int i, float W)
    {
        Theme.Rect(new Rect(0, 0, W, U(112)), new Color(0.03f, 0.04f, 0.055f, 0.85f));
        Theme.Rect(new Rect(0, U(112), W, U(1)), Theme.Line);
        // Sol üstte logo ve başlık çizilir.
        Theme.Dot(new Rect(U(20), U(16), U(14), U(14)), Theme.Accent);
        Theme.Dot(new Rect(U(28), U(24), U(14), U(14)), Theme.Good);
        GUI.Label(new Rect(U(52), U(9), U(600), U(26)), "Kompresör Sıralama", Theme.H2);
        GUI.Label(new Rect(U(52), U(34), U(700), U(20)), $"RL ödevi, {data.title}", Theme.Tiny);

        // Ortada fabrika saati gösterilir.
        float h = data.runs[0].hour[i];
        var clockR = new Rect(W / 2 - U(90), U(8), U(180), U(46));
        GUI.Box(clockR, GUIContent.none, Theme.Pill);
        GUI.Label(new Rect(clockR.x + U(18), clockR.y + U(17), U(60), U(16)), h >= 6 && h < 18 ? "GÜNDÜZ" : "GECE", Theme.Caps);
        GUI.Label(new Rect(clockR.x + U(74), clockR.y + U(3), U(100), U(40)), Clock(h), Theme.Big);

        // Sağdaki butonlar sağdan sola dizilir.
        float x = W - U(20), bh = U(32), y = U(14);
        x -= U(104); if (Button(new Rect(x, y, U(104), bh), "Özet  (Tab)", showSummary)) { showSummary = !showSummary; showGuide = false; }
        x -= U(110); if (Button(new Rect(x, y, U(104), bh), "Rehber  (G)", showGuide)) { showGuide = !showGuide; showSummary = false; }
        x -= U(100); if (Button(new Rect(x, y, U(94), bh), Sfx.Muted ? "Ses kapalı" : "Ses açık", !Sfx.Muted)) Sfx.Muted = !Sfx.Muted;
        x -= U(82); if (Button(new Rect(x, y, U(76), bh), "Baştan")) Restart();
        x -= U(38); if (Button(new Rect(x, y, U(32), bh), "+")) ChangeSpeed(+1);
        x -= U(72);
        GUI.Label(new Rect(x, y, U(72), bh), $"×{speed:0}", new GUIStyle(Theme.H2) { alignment = TextAnchor.MiddleCenter });
        Hint(new Rect(x, y, U(72), bh), $"Oynatma hızı. ×{speed:0} hızda gerçek 1 saniyede fabrikada {speed / 60f:0.#} dakika geçiyor.", true);
        x -= U(32); if (Button(new Rect(x, y, U(32), bh), "−")) ChangeSpeed(-1);
        x -= U(98); if (Button(new Rect(x, y, U(90), bh), paused ? "Oynat" : "Durdur", paused)) paused = !paused;

        // Alt satıra model ve senaryo seçimi çizilir.
        float sx = U(20), sy = U(68);
        int cm = Segment(ref sx, sy, "MODEL", System.Array.ConvertAll(index.models, m => m.@short), modelIdx,
                         System.Array.ConvertAll(index.models, ModelHint));
        int cs = Segment(ref sx, sy, "SENARYO", System.Array.ConvertAll(index.scenarios, s => s.label), scenIdx,
                         System.Array.ConvertAll(index.scenarios, s => ScenarioHint(s.key)));
        if (cm >= 0 && cm != modelIdx) { modelIdx = cm; LoadReplay(); }
        if (cs >= 0 && cs != scenIdx) { scenIdx = cs; LoadReplay(); }
    }

    // Bir istasyonun bilgi panelini, karar günlüğünü ve sahne etiketlerini çizer
    void DrawStation(StationView st, int s, int i)
    {
        var r = st.run;
        float W = Screen.width, half = s * W / 2;
        var panel = new Rect(half + U(16), U(124), U(440), U(272));
        GUI.Box(panel, GUIContent.none, Theme.Panel);
        float px = panel.x + U(16), py = panel.y + U(12), pw = panel.width - U(32);

        // Panelin başlığı ve sağ üstteki etiket çizilir.
        GUI.Label(new Rect(px, py - U(2), pw, U(32)), r.name, Theme.H1);
        string tag = s == 0 ? "REFERANS" : "RL AJANI";
        float tw = Theme.Caps.CalcSize(new GUIContent(tag)).x + U(16);
        var tagR = new Rect(panel.xMax - tw - U(14), py + U(6), tw, U(20));
        Theme.RoundRect(tagR, s == 0 ? new Color(1, 1, 1, 0.1f) : new Color(Theme.Accent.r, Theme.Accent.g, Theme.Accent.b, 0.35f));
        GUI.Label(new Rect(tagR.x + U(8), tagR.y + U(4), tw, U(16)), s == 0 ? tag : $"<color=#cfe3ff>{tag}</color>", Theme.Caps);
        string desc = s == 0 ? "Fabrikalarda yaygın olarak kullanılıyor, sabit basınç eşiklerine göre çalışıyor." : data.modelDesc;
        GUI.Label(new Rect(px, py + U(32), pw, U(34)), desc, new GUIStyle(Theme.Tiny) { wordWrap = true });
        Hint(new Rect(panel.x, panel.y, panel.width, U(70)), s == 0
            ? "Baseline, fabrikalarda bugün kullanılan klasik kontrolü gösteriyor. Basınç bir eşiğin altına inince bir kompresörü daha yüke alıyor, üst eşiği geçince boşa alıyor. 5 dakika boşta kalan motoru durduruyor. VSD ise basıncı 6,9 bar'da tutmaya çalışıyor."
            : ModelHint(index.models[modelIdx]));

        // Göstergeler 3 sütun ve 2 satır halinde dizilir.
        float p = r.pressure[i];  // tank basıncı [bar]
        string Unit(string u) => $" <size={Fs(14)}><color=#{Theme.Hex(Theme.Muted)}>{u}</color></size>";
        var tiles = new (string label, string value)[] {
            ("BASINÇ", $"<color=#{Theme.Hex(st.PressureColor(p))}>{p:0.00}</color>{Unit("bar")}"),
            ("GÜÇ", $"{r.power[i]:0}{Unit("kW")}"),
            ("TALEP", $"{r.demand[i]:0.0}{Unit("m³/dk")}"),
            ("ENERJİ", $"{r.energy[i]:0}{Unit("kWh")}"),
            ("MOTOR KALKIŞI", $"{r.starts[i]}"),
            ("6 BAR ALTI", r.lowMin[i] > 0 ? $"<color=#{Theme.Hex(Theme.Bad)}>{r.lowMin[i]:0}</color>{Unit("dk")}" : $"0{Unit("dk")}"),
        };
        float tileW = (pw - U(16)) / 3f, tileH = U(50);
        for (int k = 0; k < tiles.Length; k++)
        {
            var tr = new Rect(px + (k % 3) * (tileW + U(8)), py + U(70) + (k / 3) * (tileH + U(6)), tileW, tileH);
            GUI.Box(tr, GUIContent.none, Theme.Card);
            GUI.Label(new Rect(tr.x + U(10), tr.y + U(6), tileW, U(14)), tiles[k].label, Theme.Caps);
            GUI.Label(new Rect(tr.x + U(10), tr.y + U(19), tileW, U(28)), tiles[k].value, Theme.Value);
            Hint(tr, TileHints[k]);
        }

        // Son 1 saatin basınç grafiği çizilir. Kırmızı çizgi 6 bar'ı, sarı çizgi 6,2 bar'ı gösterir.
        // Mor çizgi VSD devrinin 1 dakikalık ortalamasıdır. Grafiğin tabanı kapalı, tepesi tam devir anlamına gelir.
        var g = new Rect(px, py + U(184), pw, U(70));
        GUI.Box(g, GUIContent.none, Theme.Card);
        const float lo = 5.6f, hi = 7.6f;  // grafiğin basınç aralığı [bar]
        float Y(float v) => g.y + U(4) + (g.height - U(8)) * (1f - Mathf.Clamp01((v - lo) / (hi - lo)));
        Theme.Rect(new Rect(g.x + U(6), Y(data.pMin), g.width - U(12), U(1.5f)), new Color(1f, 0.3f, 0.25f, 0.8f));
        Theme.Rect(new Rect(g.x + U(6), Y(data.pMargin), g.width - U(12), U(1)), new Color(1f, 0.75f, 0.2f, 0.45f));
        int window = Mathf.RoundToInt(3600f / data.dt);           // 1 saatteki adım sayısı
        int cols = Mathf.RoundToInt((g.width - U(12)) / U(2));     // grafikteki sütun sayısı
        float gb = g.yMax - U(5), gh = g.height - U(26);           // VSD çizgisinin tabanı ve en fazla yüksekliği [piksel], tepe başlığın altında kalır
        int avg = Mathf.Max(1, Mathf.RoundToInt(60f / data.dt));  // 1 dakikadaki adım sayısı
        float prevY = -1f;  // bir önceki sütundaki VSD çizgisinin yüksekliği [piksel]
        for (int c = 0; c < cols; c++)
        {
            int k = i - window + c * window / cols;
            if (k < 0) continue;
            float sum = 0f; int n = 0;
            for (int j = Mathf.Max(0, k - avg + 1); j <= k; j++, n++) sum += r.vsdSpeed[j];
            float vy = gb - gh * sum / n;  // VSD devrinin 1 dakikalık ortalaması [piksel]
            float top = prevY < 0 ? vy : Mathf.Min(vy, prevY), bot = prevY < 0 ? vy : Mathf.Max(vy, prevY);
            Theme.Rect(new Rect(g.x + U(6) + c * U(2), top - U(1), U(2.4f), bot - top + U(2)), VsdColor);  // sütunlar birbirine bağlansın diye arası doldurulur
            prevY = vy;
        }
        for (int c = 0; c < cols; c++)
        {
            int k = i - window + c * window / cols;
            if (k < 0) continue;
            float v = r.pressure[k];
            Theme.Rect(new Rect(g.x + U(6) + c * U(2), Y(v) - U(1.5f), U(2.4f), U(3)), st.PressureColor(v));
        }
        GUI.Label(new Rect(g.x + U(8), g.y + U(5), U(200), U(14)), "SON 1 SAATTE BASINÇ", Theme.Caps);
        float lw = Theme.Caps.CalcSize(new GUIContent("SON 1 SAATTE BASINÇ")).x;
        Theme.Rect(new Rect(g.x + U(16) + lw, g.y + U(10.5f), U(14), U(2.5f)), VsdColor);  // çizginin küçük örneği
        GUI.Label(new Rect(g.x + U(32) + lw, g.y + U(5), U(200), U(14)), "VSD DEVRİ", Theme.Caps);
        Hint(g, "Son bir saatteki basıncı gösteriyor. Kırmızı çizgi 6 bar, sarı çizgi 6,2 bar. Çizgi bu ikisinin biraz üstünde ve düz giderse iyi. " +
                "Mor çizgi VSD'nin devrini 1 dakikalık ortalama olarak gösteriyor. Çizgi yükseldikçe VSD daha hızlı dönüp daha çok hava basıyor. En alt kapalı, en üst tam devir demek.");
        GUI.Label(new Rect(g.xMax - U(44), Y(data.pMin) - U(15), U(40), U(14)), "6 bar", new GUIStyle(Theme.Tiny) { fontSize = Fs(11) });

        DrawEventLog(r, i, new Rect(panel.xMax + U(10), panel.y, W / 2 - panel.width - U(42), panel.height), s == 0);

        // Sahnedeki kompresör, tank ve üretim etiketleri çizilir.
        string[] st3 = { State(r.k1[i]), State(r.k2[i]), r.vsdSpeed[i] > 0 ? $"%{r.vsdSpeed[i] * 100:0} devir" : "kapalı" };
        Color[] sc = { StateColor(r.k1[i]), StateColor(r.k2[i]), r.vsdSpeed[i] > 0 ? StationView.LoadColor : StateColor(0) };
        float[] temps = { r.temp1[i], r.temp2[i], r.temp3[i] };  // motor sıcaklıkları [°C]
        if (st.Failed(t)) { st3[data.failComp] = "ARIZALI"; sc[data.failComp] = StationView.FailColor; }
        for (int k = 0; k < 3; k++)
            Chip(cams[s], st.CompLabel(k), $"<b>{data.compNames[k]}</b>  <color=#9aa4b2>{data.compKw[k]:0} kW</color>",
                 $"{st3[k]}, {temps[k]:0}°C", sc[k], CompHint(k));
        Chip(cams[s], st.tankLabelPos, "<b>Hava tankı</b>", $"{p:0.00} bar", st.PressureColor(p),
             "Kompresörlerin bastığı hava burada birikiyor. Basılan hava tüketimden fazlaysa basınç yükseliyor, azsa düşüyor.");
        bool low = p < data.pMin;
        Chip(cams[s], st.factoryLabelPos, "<b>Üretim</b>", low ? "aksıyor" : $"{r.demand[i]:0.0} m³/dk", low ? Theme.Bad : StationView.LoadColor,
             "Fabrikanın hava tüketimi. Basınç 6 bar'ın altına inerse buradaki makineler düzgün çalışmıyor.");
    }

    static readonly Color VsdColor = new Color(0.72f, 0.58f, 1f, 0.9f);  // basınç grafiğindeki VSD devri çizgisi

    static Color StateColor(int s) => s == 2 ? StationView.LoadColor : s == 1 ? StationView.UnloadColor : new Color(0.45f, 0.48f, 0.52f);

    // Sahnedeki bir noktanın üstüne adını ve durumunu gösteren iki satırlık etiket koyar
    void Chip(Camera cam, Vector3 world, string title, string status, Color dot, string tip = null)
    {
        var sp = cam.WorldToScreenPoint(world);
        if (sp.z < 0) return;
        float w = Mathf.Max(Theme.Small.CalcSize(new GUIContent(title)).x, Theme.Tiny.CalcSize(new GUIContent(status)).x + U(14)) + U(22);
        var r = new Rect(sp.x - w / 2, Screen.height - sp.y - U(22), w, U(44));
        GUI.Box(r, GUIContent.none, Theme.Pill);
        GUI.Label(new Rect(r.x + U(11), r.y + U(4), w, U(18)), title, Theme.Small);
        Theme.Dot(new Rect(r.x + U(11), r.y + U(27), U(8), U(8)), dot);
        GUI.Label(new Rect(r.x + U(24), r.y + U(23), w, U(16)), status, Theme.Tiny);
        if (tip != null) Hint(r, tip);
    }

    // Karar günlüğünü çizer. Şu ana kadarki kararlar en yenisi üstte olacak şekilde dizilir, son 5 dakikadakiler vurgulanır.
    void DrawEventLog(ReplayRun r, int i, Rect box, bool baseline)
    {
        GUI.Box(box, GUIContent.none, Theme.Panel);
        GUI.Label(new Rect(box.x + U(14), box.y + U(12), box.width, U(20)), "Karar günlüğü", Theme.H2);
        GUI.Label(new Rect(box.x + U(138), box.y + U(17), box.width, U(16)), "en yeni üstte", Theme.Tiny);
        Hint(new Rect(box.x, box.y, box.width, U(40)), baseline
            ? "Baseline'ın verdiği kararlar. Her satırda kararın saati, o anki basınç ve talep, altında da kararı hangi kuralın verdirdiği var."
            : "Modelin verdiği kararlar. Model kurallarını söyleyemediği için alttaki sebepler kararın verildiği duruma bakılarak tahmin edildi. Kırmızı satırlar 30 saniye içinde geri alınan kararlar.");
        int last = System.Array.BinarySearch(r.evStep, i);  // şu ana kadarki son kararın sırası
        last = last >= 0 ? last : ~last - 1;
        while (last + 1 < r.evStep.Length && r.evStep[last + 1] <= i) last++;
        float ey = box.y + U(44), rowH = U(44);
        if (last < 0) GUI.Label(new Rect(box.x + U(14), ey, box.width, U(20)), "Henüz karar yok", Theme.Tiny);
        var whyStyle = new GUIStyle(Theme.Tiny) { fontSize = Fs(13) };
        var timeStyle = new GUIStyle(Theme.Small) { normal = { textColor = Theme.Muted } };
        for (int k = last; k >= 0 && ey + rowH <= box.yMax - U(4); k--, ey += rowH)
        {
            bool recent = i - r.evStep[k] < 300f / data.dt;  // son 5 dakika içinde mi?
            var row = new Rect(box.x + U(10), ey, box.width - U(20), rowH - U(5));
            if (recent) Theme.RoundRect(row, new Color(1f, 1f, 1f, 0.06f));
            string why = r.evWhy[k];
            Color wc = baseline ? new Color(0.62f, 0.68f, 0.78f) : IsFlicker(why) ? Theme.Bad : Theme.Good;
            Theme.RoundRect(new Rect(row.x, row.y + U(3), U(3), row.height - U(6)), wc);  // renkli sol çizgi
            GUI.Label(new Rect(row.x + U(10), row.y + U(3), U(50), U(18)), Clock(r.hour[r.evStep[k]]), timeStyle);
            GUI.Label(new Rect(row.x + U(56), row.y + U(3), row.width, U(18)), r.evText[k], Theme.Small);
            GUI.Label(new Rect(row.x + U(56), row.y + U(21), row.width - U(60), U(16)), $"<color=#{Theme.Hex(wc)}>{why}</color>", whyStyle);
        }
    }

    // İki taraf o an farklı kompresörlerle çalışıyorsa bunu sahnenin altında gösterir
    void DrawDifference(int i)
    {
        string b = Combo(data.runs[0], i), a = Combo(data.runs[1], i);
        if (b == a) return;
        float W = Screen.width;
        string txt = $"<color=#{Theme.Hex(Theme.Warn)}>FARKLI KARAR</color>      Baseline: {b}      Model: {a}";
        var sz = Theme.Body.CalcSize(new GUIContent(txt));
        var r = new Rect(W / 2 - sz.x / 2 - U(18), Screen.height - U(254), sz.x + U(36), U(36));  // alt panelin hemen üstü
        Theme.RoundRect(r, new Color(0.22f, 0.17f, 0.04f, 0.92f));
        Theme.RoundRect(new Rect(r.x, r.y, U(4), r.height), Theme.Warn);
        GUI.Label(new Rect(r.x + U(18), r.y + (r.height - sz.y) / 2, sz.x, sz.y), txt, Theme.Body);
        Hint(r, "İki taraf şu an farklı kompresörlerle hava basıyor. Yazılanlar yükteki kompresörler, boşta bekleyenler ayrıca belirtiliyor.");
    }

    string Combo(ReplayRun r, int i)  // yükte olan kompresörleri yazar, örneğin "K1 ve K3-VSD"
    {
        var parts = new List<string>();
        if (r.k1[i] == 2) parts.Add(data.compNames[0]);
        if (r.k2[i] == 2) parts.Add(data.compNames[1]);
        if (r.vsdSpeed[i] > 0) parts.Add(data.compNames[2]);
        string s = parts.Count > 0 ? string.Join(" ve ", parts) : "hiçbiri";
        int idle = (r.k1[i] == 1 ? 1 : 0) + (r.k2[i] == 1 ? 1 : 0);  // boşta bekleyen kompresör sayısı
        return idle > 0 ? s + $", {idle} tanesi boşta" : s;
    }

    // Alt paneli çizer. Panel şu ana kadarki enerji farkını ve bu farkın nereden geldiğini gösterir.
    void DrawSavings(int i)
    {
        float W = Screen.width, H = Screen.height;
        ReplayRun b = data.runs[0], a = data.runs[1];
        var box = new Rect(U(16), H - U(204), W - U(32), U(116));
        GUI.Box(box, GUIContent.none, Theme.Panel);

        float e0 = b.energy[i], e1 = a.energy[i];            // şu ana kadarki enerji [kWh]
        float save = e0 > 1f ? (e0 - e1) / e0 * 100f : 0f;  // modelin tasarrufu [%]
        Color col = save >= 0 ? Theme.Good : Theme.Bad;
        GUI.Label(new Rect(box.x + U(20), box.y + U(14), U(400), U(16)), "ŞU ANA KADARKİ ENERJİ FARKI", Theme.Caps);
        string pct = $"<color=#{Theme.Hex(col)}>{(save >= 0 ? "−" : "+")}%{Mathf.Abs(save):0.0}</color>";
        GUI.Label(new Rect(box.x + U(20), box.y + U(30), U(400), U(44)), pct, Theme.Big);
        float pw = Theme.Big.CalcSize(new GUIContent(pct)).x;
        GUI.Label(new Rect(box.x + U(32) + pw, box.y + U(45), U(300), U(24)),
            $"{Mathf.Abs(e0 - e1):0} kWh {(e0 >= e1 ? "daha az" : "daha fazla")}", Theme.Body);
        float dp = b.power[i] - a.power[i];  // anlık güç farkı [kW]
        GUI.Label(new Rect(box.x + U(20), box.y + U(84), U(420), U(18)),
            $"Model şu an {Mathf.Abs(dp):0} kW daha {(dp >= 0 ? "az" : "fazla")} güç çekiyor", Theme.Tiny);
        Hint(new Rect(box.x, box.y, U(460), box.height),
            "Vardiyanın başından beri modelin baseline'a göre ne kadar az ya da fazla enerji harcadığını gösteriyor. Yeşilse model daha az harcamış.");

        // Fark üç kaynaktan gelir. Yeşil çubuk modelin daha az, kırmızı çubuk daha fazla harcadığı kalemi gösterir.
        var items = new (string name, string hint, float kwh, string tip)[] {
            ("Boşta dönen motorlar", "hava basmadan güç çeken kompresörler", b.eUnload[i] - a.eUnload[i],
             "K1 ve K2 boştayken hava basmadığı halde gücünün %30'unu çekiyor. Gerekmeyen kompresörü boşta bekletmek yerine kapatan model bu kalemde kazanıyor."),
            ("Gereğinden yüksek basınç", "basıncın 6,2 bar'ın üstünde tutulması", b.ePressure[i] - a.ePressure[i],
             "Basınç 6,2 bar'ın üstüne çıktıkça kompresörler daha çok güç çekiyor, kaçaklardan da daha çok hava gidiyor. Basıncı düşük tutan model bu kalemde kazanıyor."),
            ("Hangi makinelerin çalıştığı", "havanın hangi kompresörlerle basıldığı", b.eProduction[i] - a.eProduction[i],
             "Diğer iki kalemden geriye kalan fark burada görünüyor. Aynı havayı daha verimli kompresörlerle basmak burada kazanç olarak görünüyor, örneğin talep düşükken K1 yerine K2 ile VSD'yi kullanmak. Bir vardiyada toplam 900 kWh civarı harcandığı için birkaç kWh'lik fark önemsiz."),
        };
        float scale = 1f;  // çubuğun tamamı kaç kWh, en büyük kaleme göre
        foreach (var it in items) scale = Mathf.Max(scale, Mathf.Abs(it.kwh));
        float cx = box.x + U(470), cw = (box.xMax - cx - U(14)) / 3f;
        GUI.Label(new Rect(cx, box.y + U(14), U(600), U(16)), "FARK NEREDEN GELİYOR?", Theme.Caps);
        float capW = Theme.Caps.CalcSize(new GUIContent("FARK NEREDEN GELİYOR?")).x;
        GUI.Label(new Rect(cx + capW + U(10), box.y + U(12), U(600), U(16)), "modelin baseline'a göre harcadığı enerji", Theme.Tiny);
        for (int k = 0; k < 3; k++)
        {
            var it = items[k];
            var card = new Rect(cx + k * cw, box.y + U(34), cw - U(10), U(70));
            GUI.Box(card, GUIContent.none, Theme.Card);
            bool zero = Mathf.Abs(it.kwh) < 3f;  // 3 kWh'den küçük fark önemsiz sayılır
            Color c = zero ? Theme.Muted : it.kwh >= 0 ? Theme.Good : Theme.Bad;
            string val = zero ? "fark yok" : it.kwh >= 0 ? $"{it.kwh:0} kWh az" : $"{-it.kwh:0} kWh fazla";
            Hint(card, it.tip);
            GUI.Label(new Rect(card.x + U(12), card.y + U(9), cw, U(20)), it.name, Theme.Small);
            var vs = Theme.H2.CalcSize(new GUIContent(val));
            GUI.Label(new Rect(card.xMax - vs.x - U(12), card.y + U(6), vs.x, U(24)), $"<color=#{Theme.Hex(c)}>{val}</color>", Theme.H2);
            var track = new Rect(card.x + U(12), card.y + U(35), card.width - U(24), U(8));
            Theme.RoundRect(track, new Color(1f, 1f, 1f, 0.07f));
            float fw = track.width * Mathf.Abs(it.kwh) / scale;
            if (fw > U(4)) Theme.RoundRect(new Rect(track.x, track.y, fw, track.height), c);
            GUI.Label(new Rect(card.x + U(12), card.y + U(48), card.width, U(16)), it.hint, Theme.Tiny);
        }
    }

    // Zaman çubuğunu çizer. Arıza süresi kırmızıyla gösterilir. Çubuğa tıklanınca oynatma o ana atlar.
    void DrawTimeline(int i)
    {
        float W = Screen.width, H = Screen.height;
        Theme.Rect(new Rect(0, H - U(80), W, U(80)), new Color(0.03f, 0.04f, 0.055f, 0.85f));
        var track = new Rect(U(24), H - U(64), W - U(48), U(8));
        Theme.RoundRect(track, new Color(1f, 1f, 1f, 0.1f));
        float total = (data.Steps - 1) * data.dt / 3600f;  // bölüm uzunluğu [saat]
        float elapsed = i * data.dt / 3600f;               // geçen süre [saat]
        if (data.shadeStart >= 0)
            Theme.RoundRect(new Rect(track.x + track.width * data.shadeStart / total, track.y, track.width * (data.shadeEnd - data.shadeStart) / total, track.height), new Color(1f, 0.3f, 0.25f, 0.85f));
        Theme.RoundRect(new Rect(track.x, track.y, Mathf.Max(U(8), track.width * elapsed / total), track.height), Theme.Accent);
        float hx = track.x + track.width * elapsed / total;
        Theme.Dot(new Rect(hx - U(8), track.y - U(4), U(16), U(16)), Color.white);
        // Her saat başına fabrika saati yazılır.
        float h0 = data.runs[0].hour[0];
        var tick = new GUIStyle(Theme.Tiny) { alignment = TextAnchor.UpperCenter, fontSize = Fs(12) };
        for (int k = 0; k <= Mathf.RoundToInt(total); k++)
        {
            float x = track.x + track.width * k / total;
            GUI.Label(new Rect(x - U(20), track.y + U(13), U(40), U(16)), Clock(Mathf.Repeat(h0 + k, 24f)), tick);
        }
        Hint(new Rect(track.x, track.y - U(10), track.width, U(28)), data.shadeStart >= 0
            ? "Oynatmanın vardiyanın neresinde olduğunu gösteriyor. Tıklayınca o ana gidiliyor. Kırmızı kısım arızanın sürdüğü saatleri gösteriyor."
            : "Oynatmanın vardiyanın neresinde olduğunu gösteriyor. Tıklayınca o ana gidiliyor.");
        var e = Event.current;
        if (e.type == EventType.MouseDown && new Rect(track.x, track.y - U(10), track.width, U(28)).Contains(e.mousePosition))
        {
            t = Mathf.Clamp01((e.mousePosition.x - track.x) / track.width) * (data.Steps - 1);
            e.Use();
        }
        // En alta renklerin anlamı ve tuşlar yazılır.
        float lx = U(24), ly = H - U(24);
        foreach (var (c, name) in new[] { (StationView.LoadColor, "yükte"), (StationView.UnloadColor, "boşta"), (StateColor(0), "kapalı"), (StationView.FailColor, "arıza") })
        {
            Theme.Dot(new Rect(lx, ly + U(3), U(9), U(9)), c);
            GUI.Label(new Rect(lx + U(14), ly - U(1), U(60), U(16)), name, Theme.Tiny);
            lx += U(70);
        }
        GUI.Label(new Rect(W - U(1000), ly - U(1), U(976), U(16)), "Açıklama için fareyle üstüne gelin    Boşluk durdur    Oklar hız    R baştan    Tab özet    G rehber    M ses",
                  new GUIStyle(Theme.Tiny) { alignment = TextAnchor.UpperRight });
    }

    // Özet ekranını çizer. Ekranda bölüm sonundaki değerler ve modelin ne yaptığını anlatan cümleler yer alır.
    void DrawSummary()
    {
        float W = Screen.width, H = Screen.height;
        Theme.Rect(new Rect(0, U(113), W, H), new Color(0.01f, 0.015f, 0.025f, 0.75f));
        var box = new Rect((W - U(1240)) / 2, U(150), U(1240), U(620));
        Theme.RoundRect(box, new Color(0.07f, 0.085f, 0.11f, 1f));
        GUI.Box(box, GUIContent.none, Theme.Panel);
        ReplayRun b = data.runs[0], a = data.runs[1];
        int n = data.Steps - 1;
        float x = box.x + U(32), y = box.y + U(26);
        GUI.Label(new Rect(x, y, box.width, U(16)), "VARDİYA ÖZETİ", Theme.Caps);
        GUI.Label(new Rect(x, y + U(16), box.width, U(36)), $"{a.name}  <color=#{Theme.Hex(Theme.Muted)}>ve</color>  {b.name}", Theme.H1);
        GUI.Label(new Rect(x, y + U(54), box.width, U(20)), $"{data.title}, {n * data.dt / 3600f:0} saatin tamamı", Theme.Tiny);
        if (Button(new Rect(box.xMax - U(120), box.y + U(24), U(96), U(32)), "Kapat")) showSummary = false;

        // Sol tarafa bölüm sonu değerlerinin tablosu çizilir.
        var rows = new (string name, float bv, float av, string fmt, string unit)[] {
            ("Enerji", b.energy[n], a.energy[n], "0", "kWh"),
            ("Özgül güç", b.sumSpecific, a.sumSpecific, "0.00", "kW/(m³/dk)"),
            ("Ortalama basınç", b.sumMeanP, a.sumMeanP, "0.00", "bar"),
            ("6 bar altı süre", b.lowMin[n], a.lowMin[n], "0", "dk"),
            ("Motor kalkışı", b.starts[n], a.starts[n], "0", "adet"),
            ("Yük/boş geçişi", b.sumSwitches, a.sumSwitches, "0", "adet"),
            ("Limit aşan kalkış", b.sumExcess, a.sumExcess, "0", "adet"),
            ("Boşta çalışma enerjisi", b.eUnload[n], a.eUnload[n], "0", "kWh"),
            ("Yüksek basınç enerjisi", b.ePressure[n], a.ePressure[n], "0", "kWh"),
        };
        float ty = y + U(96), rh = U(40), tw = U(640);
        float[] colX = { 0, 270, 390, 500 };  // sütunların soldan uzaklığı [1080p piksel]
        string[] head = { "DEĞER", "BASELINE", "MODEL", "FARK" };
        for (int c = 0; c < 4; c++) GUI.Label(new Rect(x + U(12) + U(colX[c]), ty, U(200), U(16)), head[c], Theme.Caps);
        for (int r = 0; r < rows.Length; r++)
        {
            var row = rows[r];
            var rr = new Rect(x, ty + U(22) + r * rh, tw, rh - U(4));
            if (r % 2 == 0) Theme.RoundRect(rr, new Color(1, 1, 1, 0.035f));
            float d = row.av - row.bv;
            Color col = Mathf.Abs(d) < 1e-3f ? Theme.Muted : d < 0 ? Theme.Good : Theme.Bad;  // hepsinde düşük değer iyi
            float yy = rr.y + U(8);
            GUI.Label(new Rect(rr.x + U(12), yy, U(250), U(22)), row.name, Theme.Body);
            GUI.Label(new Rect(rr.x + U(12) + U(colX[1]), yy, U(110), U(22)), row.bv.ToString(row.fmt), Theme.Body);
            GUI.Label(new Rect(rr.x + U(12) + U(colX[2]), yy, U(110), U(22)), row.av.ToString(row.fmt), Theme.Body);
            GUI.Label(new Rect(rr.x + U(12) + U(colX[3]), yy, U(140), U(22)),
                $"<color=#{Theme.Hex(col)}>{(d > 0 ? "+" : "")}{d.ToString(row.fmt)}</color> <size={Fs(13)}><color=#{Theme.Hex(Theme.Muted)}>{row.unit}</color></size>", Theme.Body);
        }

        // Sağ tarafa modelin ne yaptığını anlatan cümleler yazılır.
        float ix = x + tw + U(40), iw = box.xMax - ix - U(32), iy = ty;
        GUI.Label(new Rect(ix, iy, iw, U(16)), "BU MODEL NE YAPIYOR?", Theme.Caps);
        iy += U(26);
        var wrap = new GUIStyle(Theme.Body) { wordWrap = true, fontSize = Fs(16) };
        foreach (var s in insights)
        {
            var content = new GUIContent(s);
            float hh = wrap.CalcHeight(content, iw - U(22));
            Theme.Dot(new Rect(ix, iy + U(7), U(7), U(7)), Theme.Accent);
            GUI.Label(new Rect(ix + U(20), iy, iw - U(22), hh), content, wrap);
            iy += hh + U(12);
        }

        GUI.Label(new Rect(x, box.yMax - U(60), box.width - U(64), U(44)),
            "Modellerin çoğunun kuralları doğrudan okunamıyor. Bu yüzden buradaki açıklamalar davranışlarına bakılarak çıkarıldı. İki kontrolcü aynı gün çalıştırıldı ve enerji farkı " +
            "boşta dönen motorlar, gereğinden yüksek basınç ve hangi makinelerin çalıştığı olarak üçe ayrıldı. Karar günlüğündeki sebepler de kararın verildiği duruma bakılarak yazıldı.",
            new GUIStyle(Theme.Tiny) { wordWrap = true });
    }
}
