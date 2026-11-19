using UnityEngine;

// Bu dosya fare bir öğenin üstüne gelince çıkan kısa açıklamaları ve bu açıklamaların metinlerini içerir.
public partial class ReplayApp
{
    string hint;          // bu karede gösterilecek açıklama, yoksa null
    float hintOver;       // farenin aynı öğe üstünde durduğu süre [s]
    string hintPrev;      // önceki karede farenin üstünde olduğu öğenin açıklaması
    Vector2? hintAt;      // ekran görüntüsü için farenin yerine kullanılan nokta [1080p piksel], -hintat ile verilir

    Vector2 Mouse => hintAt is Vector2 v ? v * ui : Event.current.mousePosition;  // fare konumu [piksel]

    // Fare r'nin içindeyse açıklamayı bu kare için kaydeder. Özet ya da rehber açıkken
    // altta kalan öğelerin açıklaması çıkmasın diye "onTop" verilmeyenler yok sayılır.
    void Hint(Rect r, string text, bool onTop = false)
    {
        if (!onTop && (showSummary || showGuide)) return;
        if (r.Contains(Mouse)) hint = text;
    }

    // OnGUI'nin en sonunda çağrılır. Fare kısa bir süre aynı yerde kalınca açıklamayı farenin yanında çizer.
    void DrawHint()
    {
        if (Event.current.type != EventType.Repaint) return;
        hintOver = hint != null && hint == hintPrev ? hintOver + Time.unscaledDeltaTime : 0f;
        hintPrev = hint;
        if (hint == null || (hintOver < 0.25f && hintAt == null)) { hint = null; return; }

        var style = new GUIStyle(Theme.Small) { wordWrap = true, fontSize = Fs(15), clipping = TextClipping.Overflow };
        var content = new GUIContent(hint);
        float w = Mathf.Min(U(380), style.CalcSize(content).x + U(4));  // metin genişliği [piksel]
        // Küçük pencerede yazı ölçülenden sık bölünebildiği için yükseklik biraz dar bir genişlikle ölçülür
        // ve her satıra pay eklenir. Böylece yazı kutudan taşmaz.
        float h = style.CalcHeight(content, w - U(16));
        h += Mathf.Ceil(h / Mathf.Max(1f, style.lineHeight)) * U(3);
        Vector2 m = Mouse;
        var r = new Rect(m.x + U(16), m.y + U(18), w + U(28), h + U(22));
        if (r.xMax > Screen.width - U(8)) r.x = m.x - r.width - U(8);  // sağa sığmıyorsa farenin soluna alınır
        if (r.yMax > Screen.height - U(8)) r.y = m.y - r.height - U(8);  // alta sığmıyorsa farenin üstüne alınır
        // Yuvarlak dolgu tek başına yarı saydam kaldığı için içi ayrıca düz renkle doldurulur.
        var fill = new Color(0.09f, 0.11f, 0.15f, 1f);
        Theme.RoundRect(r, fill);
        Theme.Rect(new Rect(r.x + U(4), r.y + U(4), r.width - U(8), r.height - U(8)), fill);
        GUI.Box(r, GUIContent.none, Theme.Panel);  // ince kenarlık
        Theme.RoundRect(new Rect(r.x, r.y, U(3), r.height), Theme.Accent);
        GUI.Label(new Rect(r.x + U(14), r.y + U(11), w, h), hint, style);
        hint = null;
    }

    // Aşağıdaki fonksiyon ve diziler açıklama metinlerini üretir.

    string ModelHint(ReplayIndex.Model m) =>
        (m.key.StartsWith("ppo")
            ? "PPO ile eğitildi. Sinir ağı hangi komutun verileceğini doğrudan öğreniyor. w, motor kalkışına verilen cezanın ağırlığını gösteriyor. w büyüdükçe model motoru açıp kapamaktan kaçınıyor. "
            : m.key == "bdq"
            ? "BDQ ile eğitildi. Ağ her komuta bir puan veriyor ve en yüksek puanlıyı seçiyor. "
            : m.key.StartsWith("q_w")
            ? "Sinir ağı kullanmıyor. Basınç, talep ve kompresörlerin durumu aralıklara bölünüyor ve her durum için tabloda bir satır ayrılıyor. Her satırda 9 hareketin puanı tutuluyor. w, motor kalkışına verilen cezanın ağırlığını gösteriyor. "
            : m.key == "q_tile"
            ? "Sinir ağı kullanmıyor. Tablo yerine tile coding kullanıyor, böylece birbirine yakın durumlar puanlarını paylaşıyor. "
            : "Sinir ağı kullanmıyor. Kaskad kontrolün eşiklerini, VSD'nin hedef basıncını ve boşta bekleme sürelerini öğrendi. Bunun için birçok farklı ayar denendi ve her turda en çok ödül toplayanlara yaklaşıldı. ")
        + m.desc;

    static string ScenarioHint(string key) => key switch
    {
        "gece" => "Gece vardiyası 22:00'de başlıyor ve 8 saat sürüyor. Talep düşük, çoğu zaman tek kompresör yetiyor.",
        "gunduz" => "Gündüz vardiyası 06:00'da başlıyor ve 8 saat sürüyor. Talep yüksek, molalarda ve öğle arasında düşüyor.",
        "k2_ariza" => "Gündüz vardiyasında K2 bir saat boyunca bozuk. Modeller eğitimde böyle bir arıza hiç görmedi.",
        "talep_+20" => "Fabrika %20 daha fazla hava tüketiyor, yeni bir üretim hattı eklenmiş gibi. Modeller bunu eğitimde görmedi.",
        "kacak_x2" => "Hattaki kaçaklar iki katına çıkmış, bakımı yapılmamış bir tesis gibi. Modeller bunu eğitimde görmedi.",
        _ => null,
    };

    string CompHint(int k) => k switch
    {
        0 => $"Sabit hızda çalışıyor, gücü {data.compKw[0]:0} kW. Ya tam güçle hava basıyor (yükte), ya hava basmadan dönüyor (boşta) ya da kapalı duruyor. Boştayken de gücünün %30'unu çekiyor.",
        1 => $"Sabit hızda çalışıyor, gücü {data.compKw[1]:0} kW. K1'den küçük olduğu için talep düşükken daha uygun.",
        _ => $"Bu kompresörün devri ayarlanabiliyor (VSD, variable speed drive). Gücü {data.compKw[2]:0} kW. İhtiyaç kadar hava basabildiği için basıncın ince ayarını bu kompresör yapıyor.",
    };

    static readonly string[] TileHints = {
        "Hava tankındaki basıncı gösteriyor. Basınç 6 bar'ın altına inerse üretim aksıyor. En iyisi 6,2 ile 6,5 bar arasında kalması. Daha yüksek basınç boşa enerji harcatıyor.",
        "Kompresörlerin şu an çektiği toplam elektrik gücünü gösteriyor.",
        "Fabrikanın şu an tükettiği havayı kaçaklarla birlikte gösteriyor. Birimi m³/dk, yani dakikada kaç metreküp hava gittiği.",
        "Vardiyanın başından beri harcanan elektriği gösteriyor. Düşük olması iyi.",
        "Kapalı motorların kaç kez çalıştırıldığını gösteriyor. Her kalkış motoru ısıtıp yıpratıyor. Bir motor saatte en fazla 6 kez çalıştırılabiliyor.",
        "Basıncın 6 bar'ın altında kaldığı toplam süreyi gösteriyor. Sıfır olmalı.",
    };

    // Rehberin "Terimler" sayfasındaki terimler ve açıklamaları
    static readonly (string term, string text)[] Terms = {
        ("Baseline", "Fabrikalarda bugün kullanılan klasik kontrol baseline olarak alındı. Kompresörleri sabit basınç eşiklerine bakarak sırayla açıp kapatıyor. Modeller bununla karşılaştırıldı."),
        ("Model, RL ajanı", "Pekiştirmeli öğrenmeyle (RL) eğitilen kontrolcüye model ya da ajan deniyor. Kompresörlere komut veriyor. Enerji harcadıkça ve basınç düştükçe ceza alıyor, zamanla cezayı azaltmayı öğreniyor. Öğrendikleri derin RL'de (PPO, BDQ) bir sinir ağında, klasik RL'de bir tabloda ya da birkaç ayarda tutuluyor."),
        ("PPO", "Proximal Policy Optimization'ın kısaltması. Sinir ağı hangi komutu vereceğini doğrudan öğreniyor. Her güncellemede davranışını sadece biraz değiştirdiği için eğitim kararlı ilerliyor. Stable-Baselines3 kütüphanesinden alındı."),
        ("BDQ", "Branching Dueling Q-Network'ün kısaltması. Sinir ağı her komutun ne kadar iyi olduğunu bir puanla tahmin edip en yüksek puanlıyı seçiyor. Her kompresörün ağda ayrı bir dalı var. PyTorch ile sıfırdan yazıldı."),
        ("Q-learning", "Klasik RL yöntemlerinden biri. Her durumda her hareketin ne kadar iyi olduğunu deneyerek tahmin ediyor. Q w=3 ve Q w=6 bu tahminleri bir tabloda tutuyor ve kalkış cezaları farklı. Q tile tablo yerine tile coding kullanıyor, böylece yakın durumlar birbirinden öğreniyor."),
        ("CEM", "Cross-Entropy Method'un kısaltması, klasik RL yöntemlerinden biri. Kontrolcünün ayarlarını değiştirerek birçok aday deniyor. En çok ödül toplayanları seçip onlara benzeyen yeni adaylar üretiyor. Vize için bu model seçildi."),
        ("w, kalkış cezası", "Eğitimde her motor kalkışına verilen cezanın ağırlığını gösteriyor. w büyüdükçe model motoru açıp kapamaktan kaçınıyor ama enerji tasarrufu azalıyor."),
        ("K1 ve K2", "K1 ve K2 sabit hızda çalışıyor. Büyük olan K1 75 kW, küçük olan K2 45 kW. Ya tam güçle çalışıyorlar, ya boşta dönüyorlar ya da kapalı duruyorlar."),
        ("VSD, K3", "Variable speed drive'ın kısaltması. K3'ün devri ayarlanabildiği için ihtiyaç kadar hava basabiliyor. Talepteki küçük değişimleri bu kompresör karşılıyor."),
        ("Yükte, boşta, kapalı", "Yükteki kompresör hava basıyor. Boştaki motor dönüyor ama hava basmıyor, yine de gücünün %30'unu çekiyor. Kapalı olan hiç güç çekmiyor."),
        ("Motor kalkışı", "Kapalı bir motorun çalıştırılmasına deniyor. Kalkış motoru ısıtıp yıprattığı için saatte en fazla 6 kalkışa izin var."),
        ("Titreme", "Bir kararın 30 saniye içinde geri alınmasına deniyor. Modelin kararsız olduğunu gösteriyor ve vanaları boşuna yıpratıyor."),
        ("Trip", "Motor çok ısınınca kendini korumak için kapanıyor. Sonra 10 dakika çalıştırılamıyor."),
        ("bar", "Basınç birimi. Hattaki basıncın 6 bar'ın üstünde kalması gerekiyor."),
        ("kW ve kWh", "kW o anki gücü, kWh harcanan enerjiyi gösteriyor. 100 kW güçle bir saat çalışan makine 100 kWh enerji harcıyor."),
        ("m³/dk", "Dakikada kaç metreküp hava basıldığını ya da tüketildiğini gösteriyor."),
        ("Özgül güç", "Dakikada 1 m³ hava için kaç kW harcandığını gösteriyor. Sektörde verimlilik buna bakılarak ölçülüyor, düşük olması iyi."),
        ("Senaryo", "Modelin test edildiği durumu anlatıyor. Gece ve gündüz vardiyasında koşullar eğitimdekiyle aynı. Arıza, fazla talep ve kaçak senaryolarını modeller eğitimde hiç görmedi."),
    };

    void DrawTerms(Rect box, float x, float y)
    {
        float colW = (box.width - U(64) - U(40)) / 2f;
        var nameStyle = new GUIStyle(Theme.Body) { fontSize = Fs(17), font = Theme.SemiBold };
        var text = new GUIStyle(Theme.Tiny) { wordWrap = true, fontSize = Fs(15) };
        int half = Terms.Length / 2 - 1;  // sol sütundaki metinler daha uzun olduğu için solda iki terim az gösterilir
        for (int col = 0; col < 2; col++)
        {
            float cx = x + col * (colW + U(40)), cy = y;
            for (int k = col * half; k < (col == 0 ? half : Terms.Length); k++)
            {
                float hh = text.CalcHeight(new GUIContent(Terms[k].text), colW - U(24));
                var row = new Rect(cx, cy, colW, hh + U(40));
                Theme.RoundRect(row, new Color(1f, 1f, 1f, 0.035f));
                GUI.Label(new Rect(row.x + U(12), row.y + U(8), colW, U(22)), Terms[k].term, nameStyle);
                GUI.Label(new Rect(row.x + U(12), row.y + U(33), colW - U(24), hh), Terms[k].text, text);
                cy += row.height + U(6);
            }
        }
    }
}
