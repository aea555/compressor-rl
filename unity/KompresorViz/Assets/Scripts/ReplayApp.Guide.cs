using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Bu dosya G tuşuyla açılan rehber penceresini çizer. Rehber hangi değerin iyi olduğunu, iyi bir kontrolcünün
// nasıl davranması gerektiğini ve seçili senaryoda hangi modelin buna en yakın olduğunu gösterir.
public partial class ReplayApp
{
    bool showGuide;  // rehber açık mı?
    int guidePage;   // rehberin hangi sayfası açık, 0 sonuçlar, 1 terimler

    // Bir modelin seçili senaryodaki sonucunu tutar
    class ModelScore
    {
        public string name;     // model adı (kısa)
        public float savePct;   // baseline'a göre enerji tasarrufu [%], eksi ise daha çok harcamış
        public int starts;      // motor kalkışı [adet]
        public float lowMin;    // 6 bar altında geçen süre [dk]
        public int flicker;     // 30 sn içinde geri alınan karar sayısı [adet]
        public bool[] ok;       // enerji, kalkış, 6 bar altı ve kararlılık ölçütlerini geçti mi
        public int OkCount { get { int c = 0; foreach (var o in ok) if (o) c++; return c; } }
    }
    List<ModelScore> scores;  // seçili senaryodaki bütün modeller
    int scoresFor = -1;       // scores hangi senaryo için hesaplandı
    int baseStarts;           // baseline'ın kalkış sayısı [adet]
    float baseLow;            // baseline'ın 6 bar altı süresi [dk]

    const int FlickerLimit = 3;  // bundan fazla geri alınan karar varsa model kararsız sayılır [adet]
    static readonly string[] CriteriaNames = { "enerji", "kalkış", "6 bar altı", "kararlılık" };

    // 6 bar altı için izin verilen süreyi döndürür [dk]. Arızada baseline da 6 bar altına inebildiği için
    // baseline'ın süresinden %10 ve 1 dk fazlasına izin verilir.
    static float LowTolerance(float baseLow) => baseLow * 1.1f + 1f;

    // Seçili senaryodaki bütün modellerin kayıtlarını okuyup ölçütlere göre puanlar
    void ComputeScores()
    {
        if (scoresFor == scenIdx) return;
        scoresFor = scenIdx;
        scores = new List<ModelScore>();
        string scen = index.scenarios[scenIdx].key;
        foreach (var m in index.models)
        {
            var entry = System.Array.Find(index.files, e => e.model == m.key && e.scenario == scen);
            var d = JsonUtility.FromJson<ReplayData>(File.ReadAllText(Path.Combine(ReplayDir, entry.file)));
            ReplayRun b = d.runs[0], a = d.runs[1];
            int n = d.Steps - 1;
            int flick = 0;
            foreach (var w in a.evWhy) if (IsFlicker(w)) flick++;
            baseStarts = b.starts[n];
            baseLow = b.lowMin[n];
            var s = new ModelScore
            {
                name = m.@short,
                savePct = (b.energy[n] - a.energy[n]) / b.energy[n] * 100f,
                starts = a.starts[n],
                lowMin = a.lowMin[n],
                flicker = flick,
            };
            s.ok = new[] { s.savePct > 0f, s.starts <= baseStarts, s.lowMin <= LowTolerance(baseLow), s.flicker <= FlickerLimit };
            scores.Add(s);
        }
    }

    // En çok ölçütü geçen modeli döndürür. Eşitlik varsa enerjide daha iyi olan seçilir.
    ModelScore Best()
    {
        ModelScore best = null;
        foreach (var s in scores)
            if (best == null || s.OkCount > best.OkCount || (s.OkCount == best.OkCount && s.savePct > best.savePct)) best = s;
        return best;
    }

    static string Missing(ModelScore s)  // geçemediği ölçütleri yazar
    {
        var m = new List<string>();
        for (int k = 0; k < s.ok.Length; k++) if (!s.ok[k]) m.Add(CriteriaNames[k]);
        return string.Join(" ve ", m);
    }

    void DrawGuide()
    {
        ComputeScores();
        float W = Screen.width, H = Screen.height;
        Theme.Rect(new Rect(0, U(113), W, H), new Color(0.01f, 0.015f, 0.025f, 0.8f));
        var box = new Rect((W - U(1560)) / 2, U(126), U(1560), H - U(126) - U(92));
        Theme.RoundRect(box, new Color(0.07f, 0.085f, 0.11f, 1f));
        GUI.Box(box, GUIContent.none, Theme.Panel);
        float x = box.x + U(32), y = box.y + U(22);
        GUI.Label(new Rect(x, y, box.width, U(16)), "REHBER", Theme.Caps);
        GUI.Label(new Rect(x, y + U(14), box.width, U(36)), guidePage == 0 ? "Sonuçlar nasıl okunur?" : "Terimler", Theme.H1);
        if (Button(new Rect(box.xMax - U(120), box.y + U(22), U(96), U(32)), "Kapat")) showGuide = false;
        if (Button(new Rect(box.xMax - U(236), box.y + U(22), U(104), U(32)), "Terimler", guidePage == 1)) guidePage = 1;
        if (Button(new Rect(box.xMax - U(352), box.y + U(22), U(104), U(32)), "Sonuçlar", guidePage == 0)) guidePage = 0;
        y += U(64);
        if (guidePage == 1) { DrawTerms(box, x, y); return; }

        float colW = (box.width - U(64) - U(40)) / 2f;
        float leftX = x, rightX = x + colW + U(40);
        var nameStyle = new GUIStyle(Theme.Body) { fontSize = Fs(16) };
        var text = new GUIStyle(Theme.Tiny) { wordWrap = true, fontSize = Fs(14) };
        string good = Theme.Hex(Theme.Good), bad = Theme.Hex(Theme.Bad), muted = Theme.Hex(Theme.Muted);

        // Sol sütuna değerler ve hedefleri yazılır.
        GUI.Label(new Rect(leftX, y, colW, U(16)), "DEĞERLER", Theme.Caps);
        var metrics = new (string name, string target, string reason)[] {
            ("6 bar altı süre", "sıfır olmalı",
             "Basınç 6 bar'ın altına inince fabrikadaki makineler düzgün çalışmıyor. Enerji tasarrufu ne kadar büyük olursa olsun bu kabul edilmez."),
            ("Enerji ve özgül güç", "düşük olmalı",
             "Asıl amaç elektrik tüketimini azaltmak. Özgül güç, dakikada 1 m³ hava için kaç kW harcandığını gösterir."),
            ("Ortalama basınç", "6,2 bar'a yakın olmalı",
             "Basınç gereğinden yüksekse kompresörler daha çok güç çeker ve kaçaklardan daha çok hava kaçar. 1 bar fazlası yaklaşık %7 daha fazla güç demek."),
            ("Boşta çalışma enerjisi", "düşük olmalı",
             "Boşta çalışan kompresör hava basmadığı halde gücünün %30'unu harcar. Klasik kaskad en çok enerjiyi burada kaybediyor."),
            ("Motor kalkışı", "az olmalı",
             "Her kalkışta motor ısınır, parçalar aşınır. Bir motor saatte en fazla 6 kez çalıştırılabilir."),
            ("Yük/boş geçişi ve titreme", "az olmalı",
             "Sık geçiş vanaları yıpratır. Bir kararın 30 saniye içinde geri alınması, modelin kararsız olduğunu gösterir."),
            ("6,2 bar altı süre", "kısa olmalı",
             "Bu süre boyunca yedek basınç az kalır. Biraz olması normal, ama uzun sürerse ani bir talep artışında basınç 6 bar'ın altına düşebilir."),
        };
        float ly = y + U(24);
        foreach (var m in metrics)
        {
            float hh = text.CalcHeight(new GUIContent(m.reason), colW - U(24));
            var row = new Rect(leftX, ly, colW, hh + U(34));
            Theme.RoundRect(row, new Color(1f, 1f, 1f, 0.035f));
            GUI.Label(new Rect(row.x + U(12), row.y + U(7), colW, U(22)), $"{m.name}  <color=#{muted}>{m.target}</color>", nameStyle);
            GUI.Label(new Rect(row.x + U(12), row.y + U(29), colW - U(24), hh), m.reason, text);
            ly += row.height + U(6);
        }

        // Sağ sütuna iyi bir kontrolcünün ne yapması gerektiği yazılır.
        GUI.Label(new Rect(rightX, y, colW, U(16)), "İYİ BİR KONTROLCÜ NE YAPAR?", Theme.Caps);
        var ideal = new (string what, string reason)[] {
            ("Basıncı 6 bar'ın altına düşürmez.", "Arıza ya da ani talep artışı olduğunda önce yeni bir kompresör devreye alır."),
            ("Basıncı 6,2 ile 6,5 bar arasında tutar.", "Daha yüksek basınç boşa enerji harcamak demek. 6,2 bar'ın altına sık inmek de riskli."),
            ("Talebe uygun kompresörü seçer.", "Talep düşükken büyük K1 (75 kW) yerine küçük K2 (45 kW) ile VSD yeterli olur."),
            ("Dalgalanmayı VSD ile karşılar.", "VSD'nin hızı ayarlanabilir. Sabit hızlı kompresörler ise en verimli oldukları tam yükte çalışır."),
            ("Gerekmeyen kompresörü kapatır.", "Boşta bekleyen kompresör hava basmadan güç çeker."),
            ("Kompresörleri sık açıp kapatmaz.", "Az sonra yine gerekecek bir kompresörü kapatmak, motoru boşuna yıpratır."),
        };
        float ry = y + U(24);
        var whatStyle = new GUIStyle(Theme.Body) { wordWrap = true, fontSize = Fs(16) };
        for (int k = 0; k < ideal.Length; k++)
        {
            float tw = colW - U(44);
            float h1 = whatStyle.CalcHeight(new GUIContent(ideal[k].what), tw);
            float h2 = text.CalcHeight(new GUIContent(ideal[k].reason), tw);
            var num = new Rect(rightX, ry + U(2), U(26), U(26));
            Theme.Dot(num, new Color(Theme.Accent.r, Theme.Accent.g, Theme.Accent.b, 0.35f));
            GUI.Label(num, $"{k + 1}", new GUIStyle(Theme.Small) { alignment = TextAnchor.MiddleCenter, font = Theme.SemiBold });
            GUI.Label(new Rect(rightX + U(40), ry + U(2), tw, h1), ideal[k].what, whatStyle);
            GUI.Label(new Rect(rightX + U(40), ry + U(4) + h1, tw, h2), ideal[k].reason, text);
            ry += h1 + h2 + U(16);
        }

        // Sağ sütunun altında modeller seçili senaryoda karşılaştırılır.
        float my = Mathf.Max(ry, y + U(24)) + U(8);
        string scenName = index.scenarios[scenIdx].label.ToUpper(new System.Globalization.CultureInfo("tr-TR"));
        GUI.Label(new Rect(rightX, my, colW, U(16)), $"MODELLER, {scenName}", Theme.Caps);
        my += U(22);
        string[] head = { "MODEL", "ENERJİ", "KALKIŞ", "6 BAR ALTI", "TİTREME" };
        float[] cx = { 0, 130, 250, 360, 480 };  // sütunların soldan uzaklığı [1080p piksel]
        for (int c = 0; c < head.Length; c++) GUI.Label(new Rect(rightX + U(12) + U(cx[c]), my, U(140), U(16)), head[c], Theme.Caps);
        my += U(18);
        var best = Best();
        var cell = new GUIStyle(Theme.Small);
        string C(bool ok, string s) => $"<color=#{(ok ? good : bad)}>{s}</color>";
        foreach (var s in scores)
        {
            var row = new Rect(rightX, my, colW, U(26));
            if (s == best)  // en iyi modelin satırı
            {
                Theme.RoundRect(row, new Color(1f, 1f, 1f, 0.08f));
                Theme.RoundRect(new Rect(row.x, row.y + U(3), U(3), row.height - U(6)), Theme.Good);
            }
            float ty = row.y + U(4);
            GUI.Label(new Rect(row.x + U(12), ty, U(120), U(20)), s.name, cell);
            GUI.Label(new Rect(row.x + U(12) + U(cx[1]), ty, U(120), U(20)), C(s.ok[0], $"{(s.savePct >= 0 ? "−" : "+")}%{Mathf.Abs(s.savePct):0.0}"), cell);
            GUI.Label(new Rect(row.x + U(12) + U(cx[2]), ty, U(120), U(20)), C(s.ok[1], $"{s.starts}") + $" <color=#{muted}>/ {baseStarts}</color>", cell);
            GUI.Label(new Rect(row.x + U(12) + U(cx[3]), ty, U(120), U(20)), C(s.ok[2], $"{s.lowMin:0} dk") + (baseLow > 0 ? $" <color=#{muted}>/ {baseLow:0}</color>" : ""), cell);
            GUI.Label(new Rect(row.x + U(12) + U(cx[4]), ty, U(120), U(20)), C(s.ok[3], $"{s.flicker}"), cell);
            my += U(27);
        }
        GUI.Label(new Rect(rightX, my + U(2), colW, U(18)),
            "Yeşil değerler baseline kadar ya da ondan iyi. Eğik çizgiden sonraki sayı baseline'ın değerini gösteriyor.",
            new GUIStyle(Theme.Tiny) { fontSize = Fs(12) });
        my += U(22);
        string verdict = best.OkCount == 4
            ? $"Bu senaryoda en iyi sonuç <color=#{good}>{best.name}</color> modelinde. Dört ölçütü de geçenler arasında en çok enerjiyi o tasarruf ediyor."
            : $"Bu senaryoda dört ölçütü de geçen model yok. En yakını <color=#{good}>{best.name}</color>, {Missing(best)} ölçütünü geçemiyor.";
        if (baseLow > 0)
            verdict += $" <color=#{muted}>Arızada kapasite yetmediği için baseline da {baseLow:0} dk 6 bar altına iniyor. Modellerin ondan belirgin şekilde kötü olmaması yeterli.</color>";
        GUI.Label(new Rect(rightX, my, colW, U(80)), verdict, new GUIStyle(Theme.Body) { wordWrap = true, fontSize = Fs(15) });

        // En alta 20 test gününün ortalaması yazılır. Sayılar pareto.py, bdq.py ve klasik_rl.py sonuçlarından alınmıştır. Q tablolarının sayıları dört seed'in ortalamasıdır.
        var foot = new Rect(x, box.yMax - U(84), box.width - U(64), U(64));
        Theme.RoundRect(foot, new Color(Theme.Accent.r, Theme.Accent.g, Theme.Accent.b, 0.12f));
        Theme.RoundRect(new Rect(foot.x, foot.y, U(4), foot.height), Theme.Accent);
        GUI.Label(new Rect(foot.x + U(18), foot.y + U(10), foot.width - U(36), foot.height - U(12)),
            "Tek gün yanıltıcı olabilir. 20 test gününün ortalamasında baseline'ı hem enerjide hem kalkışta iki model geçiyor. PPO w=3 %7,5, CEM %5,4 daha az " +
            "enerji harcıyor. Kalkış sayıları baseline'ın 11,6'sına karşı 9,6 ve 8,8. PPO w=0,3 en az enerjiyi harcıyor ama 3,5 kat fazla kalkış yapıyor. " +
            "PPO w=10'un tasarrufu %3'te kalıyor. PPO w=1 baseline'dan fazla kalkış yapıyor. BDQ %3,6 tasarruf ediyor ama çok kararsız. Q w=3 %7,1, Q w=6 %6,2 tasarruf ediyor ama ikisi de baseline'dan fazla kalkış yapıyor. Q tile baseline'ı geçemiyor.",
            new GUIStyle(Theme.Small) { wordWrap = true });
    }
}
