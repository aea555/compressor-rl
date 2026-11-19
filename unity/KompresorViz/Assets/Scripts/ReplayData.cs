using System;

// export_replay.py'nin yazdığı JSON dosyaları bu sınıflara okunur.
// JsonUtility alanları isimle eşleştirdiği için alan adları JSON'daki anahtarlarla birebir aynı olmalıdır.

[Serializable]
public class ReplayRun
{
    public string name;        // kontrolcünün ekrandaki adı, örneğin "Klasik kaskad" ya da "PPO, kalkış cezası 3"
    public float[] hour;       // fabrika saati (0-24)
    public float[] pressure;   // hat basıncı [bar]
    public float[] demand;     // tüketilen hava [m³/dk]
    public float[] power;      // çekilen elektrik gücü [kW]
    public int[] k1;           // K1'in durumu, 0 kapalı, 1 boşta, 2 yükte
    public int[] k2;           // K2'nin durumu, 0 kapalı, 1 boşta, 2 yükte
    public float[] vsdSpeed;   // VSD'nin devri, 0 kapalı, çalışırken 0,2-1
    public float[] temp1;      // K1 motor sıcaklığı [°C]
    public float[] temp2;      // K2 motor sıcaklığı [°C]
    public float[] temp3;      // VSD motor sıcaklığı [°C]
    public float[] energy;     // bölüm başından beri harcanan enerji [kWh]
    public int[] starts;       // bölüm başından beri motor kalkışı [adet]
    public float[] lowMin;     // bölüm başından beri 6 bar altında geçen süre [dk]

    // Bölüm başından beri harcanan enerji üç kaleme ayrılır [kWh]. Üçünün toplamı energy'ye eşittir.
    public float[] eUnload;     // boşta çalışan (hava basmayan) kompresörlerin harcadığı
    public float[] ePressure;   // 6,2 bar'dan yüksek basınçta çalışmanın ek maliyeti
    public float[] eProduction; // asıl hava üretimine giden kalan kısım, hangi makinelerin çalıştığına bağlıdır

    // Bu diziler karar günlüğündeki her olayın adımını, metnini ve sebebini ya da yorumunu tutar.
    public int[] evStep;
    public string[] evText;
    public string[] evWhy;

    // Aşağıdaki alanlar bölüm sonu özetini tutar.
    public float sumSpecific;  // özgül güç [kW/(m³/dk)]
    public float sumMeanP;     // ortalama basınç [bar]
    public float sumSwitches;  // yük/boş geçişi [adet]
    public float sumExcess;    // saatlik limiti aşan kalkış [adet]
}

[Serializable]
public class ReplayData
{
    public string title;       // senaryonun açıklaması
    public string model;       // ajanın ekrandaki adı
    public string modelDesc;   // ajanın tek cümlelik özeti
    public float dt;           // adım süresi [s]
    public float pMin;         // alt basınç sınırı [bar] (6 bar)
    public float pMargin;      // güvenlik payı sınırı [bar] (6,2 bar)
    public float pMax;         // üst basınç sınırı [bar]
    public float shadeStart;   // arızanın bölüm başından kaç saat sonra başladığı, arıza yoksa -1
    public float shadeEnd;     // arızanın bittiği saat, arıza yoksa -1
    public int failComp;       // arızalı kompresörün sırası (0 K1, 1 K2), arıza yoksa -1
    public string[] compNames; // kompresör isimleri (K1, K2, K3-VSD)
    public float[] compKw;     // anma güçleri [kW]
    public float[] compFlow;   // tam yükteki debiler [m³/dk]
    public float ambientC;     // ortam sıcaklığı [°C]
    public float tripC;        // motorun trip olduğu sıcaklık [°C]
    public ReplayRun[] runs;   // [0] baseline, [1] ajan

    public int Steps => runs[0].pressure.Length;  // bölümdeki adım sayısı
}

// ReplayIndex, replays/index.json dosyasını okur. Bu dosya hangi model ve senaryonun kaydı olduğunu tutar.
[Serializable]
public class ReplayIndex
{
    [Serializable] public class Model { public string key, label, @short, desc; }  // @short JSON'daki "short" anahtarı, butonda yazan kısa ad
    [Serializable] public class Scenario { public string key, label; }
    [Serializable] public class Entry { public string model, scenario, file; }

    public Model[] models;
    public Scenario[] scenarios;
    public Entry[] files;
}
