using System.Collections.Generic;
using UnityEngine;

// Bu sınıf bir kompresör istasyonunun 3D sahnesini kurar ve oynatır. Sahnede salon, 3 kompresör, hava tankı, fabrika,
// borular ve efektler bulunur. Biri baseline'ı, diğeri modeli oynatan iki istasyon vardır. Bütün objeler kodla oluşturulur.
public class StationView
{
    public static readonly Color OffColor = new Color(0.22f, 0.24f, 0.27f);    // kapalı, koyu gri
    public static readonly Color UnloadColor = new Color(1f, 0.62f, 0.12f);    // boşta, turuncu
    public static readonly Color LoadColor = new Color(0.25f, 0.85f, 0.42f);   // yükte, yeşil
    public static readonly Color FailColor = new Color(1f, 0.15f, 0.12f);      // arızalı, kırmızı
    static readonly Color AirColor = new Color(0.35f, 0.8f, 1f);              // borulardaki hava akışı
    static readonly Color PipeColor = new Color(0.33f, 0.55f, 0.78f);         // basınçlı hava borusu, sanayide genelde mavi

    const float TankLow = 5.5f, TankHigh = 7.5f;          // gözetleme borusunun gösterdiği en düşük ve en yüksek basınç [bar]
    const float GlassBottom = 0.75f, GlassHeight = 2.5f;  // gözetleme borusunun başladığı yükseklik ve boyu [m]

    // Bir kompresörün sahnedeki parçalarını tutar
    class Comp
    {
        public Transform shake;               // gövde grubu, çalışırken hafifçe titrer
        public Transform rotor;               // üstte dönen fan
        public Material strip, screen;        // durum rengini gösteren ışıklı şerit ve ön panel ekranı
        public Material[] tower;              // ışık kulesinin kırmızı, turuncu ve yeşil lambaları
        public Transform tempFill;            // sıcaklık göstergesinin dolu kısmı
        public Material tempMat;
        public float tempBottom, tempH;       // sıcaklık göstergesinin alt kenarı ve tam boyu [m]
        public ParticleSystem mist, haze;     // boşa almada tahliye buharı, fandan çıkan sıcak hava
        public float fanAngle;                // fanın açısı [derece]
        public FlowPath path;                 // kompresörden tanka giden hava yolu
        public Vector3 labelPos;              // etiketin dünyadaki konumu
    }

    // Bu sınıf bir hava yolunu tutar. Borudaki akış yol boyunca kayan ışıklı halkalarla gösterilir.
    class FlowPath
    {
        public Vector3[] pts;       // yolun köşe noktaları (istasyona göre yerel)
        public float length;        // yolun toplam uzunluğu [m]
        public Transform[] rings;   // akan halkalar
        public float phase;         // halkaların yol üzerindeki ilerlemesi [m]
    }

    public ReplayRun run;             // bu istasyonun oynattığı kayıt
    ReplayData data;
    readonly Transform root;          // istasyonun kök objesi (tüm parçalar bunun altında)
    readonly Comp[] comps = new Comp[3];
    readonly Sfx sfx;                 // bu istasyonun sesleri
    Transform glassFill, needle, piston;
    Material glassFillMat, beaconMat, factoryWindows, wallWindows;
    Light beaconLight;
    List<Light> ceilingLights;        // tavan floresanlarının ışıkları
    const float FixtureY = 5.6f;      // floresan armatürlerinin yüksekliği [m]
    ParticleSystem chimney, safetyValve;
    FlowPath demandPath;              // tanktan fabrikaya giden hava yolu
    readonly List<Transform> boxes = new();  // konveyördeki koliler
    Vector3 beltStart;                // konveyörün başlangıç noktası (yerel)
    float beltLen = 3.2f;             // konveyör uzunluğu [m]
    float beltPos, pistonPhase;       // konveyör ve pres animasyonunun ilerlemesi
    float pistonX, pistonZ;           // presin konumu (yerel)
    int lastStep = -1;                // en son işlenen adım (olayları yakalamak için)
    public Vector3 tankLabelPos, factoryLabelPos, titlePos;  // etiket konumları (dünya)

    public StationView(ReplayData data, ReplayRun run, Vector3 origin, Color floorTint, Sfx sfx)
    {
        this.data = data;
        this.run = run;
        this.sfx = sfx;
        root = new GameObject(run.name).transform;
        root.position = origin;
        BuildHall(floorTint);

        float[] xs = { -5.2f, -2.4f, 0.2f };  // kompresörlerin x konumları
        for (int i = 0; i < 3; i++) comps[i] = BuildCompressor(i, xs[i]);

        Vector3 tankPos = new Vector3(3.4f, 0f, -0.2f), fPos = new Vector3(6.8f, 0f, 0.4f);
        BuildTank(tankPos);
        BuildFactory(fPos);

        // Borular her kompresörden ön kolektöre, oradan tanka gider. Tanktan fabrikaya ayrı bir boru çıkar.
        float headerZ = -2.2f, y = 0.42f;
        Pipe(new Vector3(xs[0], y, headerZ), new Vector3(tankPos.x, y, headerZ));
        Pipe(new Vector3(tankPos.x, y, headerZ), new Vector3(tankPos.x, y, tankPos.z - 0.75f));
        for (int i = 0; i < 3; i++)
        {
            float x = xs[i];
            Pipe(new Vector3(x, y, 0.3f), new Vector3(x, y, headerZ));
            comps[i].path = MakePath(new[] {
                new Vector3(x, y, 0.3f), new Vector3(x, y, headerZ),
                new Vector3(tankPos.x, y, headerZ), new Vector3(tankPos.x, y, tankPos.z - 0.75f) }, 9);
        }
        var a = new Vector3(tankPos.x + 0.75f, y, tankPos.z);
        var b = new Vector3(fPos.x - 1.3f, y, tankPos.z);
        Pipe(a, b);
        demandPath = MakePath(new[] { a, b }, 4);
        titlePos = root.TransformPoint(new Vector3(0.6f, 0f, -4.6f));
    }

    // BuildHall salonun zeminini, duvarlarını ve tavan ışıklarını kurar.
    void BuildHall(Color tint)
    {
        Prim(PrimitiveType.Cube, "Zemin", root, new Vector3(0.6f, -0.05f, 1f), new Vector3(30f, 0.1f, 18f),
             Look.Concrete(tint, new Vector2(7f, 4.2f)));
        // Zemine sarı yürüme yolu çizgileri, tank önüne uyarı şeridi konur.
        var paint = Look.Lit(new Color(0.95f, 0.75f, 0.12f), 0.3f);
        foreach (float z in new[] { -3.3f, 3.0f })
            Prim(PrimitiveType.Cube, "Çizgi", root, new Vector3(0.9f, 0.005f, z), new Vector3(16.5f, 0.01f, 0.12f), paint);
        Prim(PrimitiveType.Cube, "Uyarı", root, new Vector3(3.4f, 0.006f, -1.4f), new Vector3(2.2f, 0.01f, 0.18f), Look.Stripes(new Vector2(4f, 0.4f)));

        // Arka duvar, kolonlar, pencere şeridi ve duvar lambaları kurulur. Pencere şeridi gündüz gökyüzü, gece karanlık görünür.
        var wall = Look.Concrete(new Color(0.55f, 0.58f, 0.63f), new Vector2(6f, 2f));
        Prim(PrimitiveType.Cube, "Duvar", root, new Vector3(0.6f, 3.5f, 5.4f), new Vector3(30f, 7f, 0.3f), wall);
        var pillar = Look.Lit(new Color(0.3f, 0.32f, 0.36f), 0.3f, 0.2f);
        wallWindows = Look.Glow(new Color(0.65f, 0.8f, 1f), 1.2f);
        var lamp = Look.Glow(new Color(1f, 0.9f, 0.75f), 3f);
        for (int k = -2; k <= 3; k++)
        {
            float x = k * 5f - 1.9f;
            Prim(PrimitiveType.Cube, "Kolon", root, new Vector3(x, 3.5f, 5.1f), new Vector3(0.5f, 7f, 0.5f), pillar);
            Prim(PrimitiveType.Cube, "Pencere", root, new Vector3(x + 2.5f, 4.6f, 5.24f), new Vector3(3.6f, 1.4f, 0.02f), wallWindows);
            Prim(PrimitiveType.Cube, "Lamba", root, new Vector3(x, 5.9f, 4.8f), new Vector3(0.9f, 0.12f, 0.3f), lamp);
        }
        // Tavan floresanları iki sıra halinde üçer tanedir. Her birinin altında aşağı bakan geniş açılı
        // bir ışık bulunur. Fabrikada olduğu gibi gündüz de gece de yanarlar.
        var housing = Look.Lit(new Color(0.75f, 0.77f, 0.8f), 0.5f, 0.6f);
        var tube = Look.Glow(new Color(0.88f, 0.94f, 1f), 4f);
        var rod = Look.Lit(new Color(0.2f, 0.21f, 0.23f), 0.4f, 0.7f);
        ceilingLights = new List<Light>();
        foreach (float z in new[] { -3.0f, 1.8f })
            foreach (float x in new[] { -4.2f, 1.2f, 6.6f })
            {
                var p = new Vector3(x, FixtureY, z);
                Prim(PrimitiveType.Cube, "Armatür", root, p + Vector3.up * 0.06f, new Vector3(3.2f, 0.1f, 0.42f), housing);
                foreach (float dz in new[] { -0.1f, 0.1f })  // iki floresan tüp
                {
                    var tr = Prim(PrimitiveType.Cylinder, "Floresan", root, p + new Vector3(0f, -0.03f, dz), new Vector3(0.07f, 1.5f, 0.07f), tube);
                    tr.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
                }
                foreach (float dx in new[] { -1.3f, 1.3f })  // tavana uzanan askılar
                    Prim(PrimitiveType.Cylinder, "Askı", root, p + new Vector3(dx, 2f, 0f), new Vector3(0.025f, 1.95f, 0.025f), rod);

                var l = new GameObject("Floresan ışığı").AddComponent<Light>();
                l.type = LightType.Spot;
                l.transform.SetParent(root, false);
                l.transform.localPosition = p + Vector3.down * 0.15f;
                l.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
                l.spotAngle = 150f;
                l.innerSpotAngle = 60f;
                l.range = 14f;
                l.color = new Color(0.92f, 0.96f, 1f);  // soğuk beyaz
                l.shadows = LightShadows.None;
                ceilingLights.Add(l);
            }
    }

    Comp BuildCompressor(int i, float x)
    {
        var c = new Comp();
        bool vsd = i == 2;
        float s = Mathf.Pow(data.compKw[i] / 75f, 0.4f);  // boyut ölçeği, 75 kW'lık kompresörde 1
        var size = new Vector3(2.2f, 1.7f, 1.8f) * s;
        var basePos = new Vector3(x, 0f, 1.2f);
        Prim(PrimitiveType.Cube, "Kızak", root, basePos + Vector3.up * 0.07f, new Vector3(size.x + 0.25f, 0.14f, size.z + 0.25f),
             Look.Lit(new Color(0.12f, 0.13f, 0.15f), 0.3f, 0.6f));

        c.shake = new GameObject(data.compNames[i]).transform;
        c.shake.SetParent(root, false);
        c.shake.localPosition = basePos + Vector3.up * 0.14f;
        var body = Look.Lit(vsd ? new Color(0.78f, 0.82f, 0.88f) : new Color(0.84f, 0.85f, 0.86f), 0.35f, 0.1f);
        var dark = Look.Lit(new Color(0.13f, 0.14f, 0.16f), 0.4f, 0.3f);
        float fz = -size.z / 2 - 0.006f;  // ön yüzün z'si (kameraya bakan yüz)
        Prim(PrimitiveType.Cube, "Gövde", c.shake, new Vector3(0, size.y / 2, 0), size, body);
        Prim(PrimitiveType.Cube, "Kapak", c.shake, new Vector3(0, size.y + 0.03f, 0), new Vector3(size.x + 0.05f, 0.06f, size.z + 0.05f), dark);
        Prim(PrimitiveType.Cube, "Etek", c.shake, new Vector3(0, 0.05f, 0), new Vector3(size.x + 0.04f, 0.1f, size.z + 0.04f), dark);
        // Gövdeyi saran ışıklı durum şeridi yeşil, turuncu ya da sönük yanar.
        c.strip = Look.Glow(OffColor, 0f);
        Prim(PrimitiveType.Cube, "Şerit", c.shake, new Vector3(0, size.y * 0.78f, 0), new Vector3(size.x + 0.02f, 0.07f, size.z + 0.02f), c.strip);
        // Ön sağa havalandırma panjuru konur.
        for (int k = 0; k < 6; k++)
            Prim(PrimitiveType.Cube, "Panjur", c.shake, new Vector3(size.x * 0.22f, size.y * (0.18f + k * 0.08f), fz),
                 new Vector3(size.x * 0.4f, 0.035f, 0.02f), dark);
        // Ön solda kontrol paneli bulunur. Panel koyu çerçeve, durum ekranı ve sıcaklık göstergesinden oluşur.
        var panelPos = new Vector3(-size.x * 0.27f, size.y * 0.5f, fz);
        Prim(PrimitiveType.Cube, "Panel", c.shake, panelPos, new Vector3(size.x * 0.32f, size.y * 0.42f, 0.03f), dark);
        c.screen = Look.Glow(OffColor, 0f);
        Prim(PrimitiveType.Quad, "Ekran", c.shake, panelPos + new Vector3(-size.x * 0.03f, size.y * 0.08f, -0.02f),
             new Vector3(size.x * 0.18f, size.y * 0.14f, 1f), c.screen);
        c.tempH = size.y * 0.3f;
        c.tempBottom = panelPos.y - c.tempH / 2;
        float tx = panelPos.x + size.x * 0.11f;
        Prim(PrimitiveType.Quad, "SıcaklıkYuva", c.shake, new Vector3(tx, panelPos.y, fz - 0.02f), new Vector3(0.06f, c.tempH, 1f),
             Look.Lit(new Color(0.05f, 0.05f, 0.06f), 0.2f));
        c.tempMat = Look.Glow(Color.blue, 1.5f);
        c.tempFill = Prim(PrimitiveType.Quad, "Sıcaklık", c.shake, new Vector3(tx, panelPos.y, fz - 0.025f), new Vector3(0.04f, 0.01f, 1f), c.tempMat).transform;

        // Kırmızı, turuncu ve yeşil lambalı sanayi tipi ışık kulesi kurulur.
        var towerBase = new Vector3(-size.x / 2 + 0.18f, size.y + 0.06f, size.z / 2 - 0.2f);
        Prim(PrimitiveType.Cylinder, "Direk", c.shake, towerBase + Vector3.up * 0.15f, new Vector3(0.05f, 0.15f, 0.05f), dark);
        c.tower = new Material[3];
        Color[] tc = { FailColor, UnloadColor, LoadColor };
        for (int k = 0; k < 3; k++)
        {
            c.tower[k] = Look.Glow(tc[k], 0f);
            Prim(PrimitiveType.Cylinder, "Kule", c.shake, towerBase + Vector3.up * (0.62f - k * 0.13f), new Vector3(0.16f, 0.06f, 0.16f), c.tower[k]);
        }

        // Üstteki fanın koyu gövdesi, dönen kanatları ve göbeği kurulur.
        var fanPos = new Vector3(size.x * 0.12f, size.y + 0.07f, 0.05f);
        float fr = Mathf.Min(size.x, size.z) * 0.62f;  // fan çapı [m]
        Prim(PrimitiveType.Cylinder, "FanGövde", c.shake, fanPos, new Vector3(fr, 0.04f, fr), dark);
        c.rotor = new GameObject("Rotor").transform;
        c.rotor.SetParent(c.shake, false);
        c.rotor.localPosition = fanPos + Vector3.up * 0.05f;
        var blade = Look.Lit(new Color(0.7f, 0.72f, 0.75f), 0.6f, 0.7f);
        for (int k = 0; k < 4; k++)
            Prim(PrimitiveType.Cube, "Kanat", c.rotor, Vector3.zero, new Vector3(fr * 0.92f, 0.015f, 0.09f), blade).transform.localRotation = Quaternion.Euler(0, k * 45f, 8f);
        Prim(PrimitiveType.Cylinder, "Göbek", c.rotor, Vector3.up * 0.01f, new Vector3(0.14f, 0.03f, 0.14f), dark);

        // VSD'nin yanında mavi ekranlı frekans çevirici dolabı durur.
        if (vsd)
        {
            var cab = new Vector3(size.x / 2 + 0.32f, 0.62f, -size.z * 0.15f);
            Prim(PrimitiveType.Cube, "Sürücü", c.shake, cab, new Vector3(0.42f, 1.24f, 0.55f), Look.Lit(new Color(0.2f, 0.3f, 0.45f), 0.4f, 0.2f));
            Prim(PrimitiveType.Quad, "SürücüEkran", c.shake, cab + new Vector3(0, 0.3f, -0.28f), new Vector3(0.26f, 0.14f, 1f), Look.Glow(new Color(0.3f, 0.65f, 1f), 1.5f));
        }

        c.mist = Particles("Tahliye", c.shake, new Vector3(size.x * 0.35f, size.y + 0.2f, 0f), new Color(0.92f, 0.95f, 1f, 0.55f),
                           size: 0.35f, life: 1.4f, speed: 1.6f, grow: 3.5f, rate: 0f, max: 120);
        c.haze = Particles("Sıcak hava", c.shake, fanPos + Vector3.up * 0.15f, new Color(1f, 0.95f, 0.9f, 0.08f),
                           size: 0.6f, life: 2.2f, speed: 0.7f, grow: 2f, rate: 0f, max: 80);
        c.labelPos = root.TransformPoint(basePos + Vector3.up * (size.y + 1.25f));
        return c;
    }

    void BuildTank(Vector3 p)
    {
        var paint = Look.Lit(new Color(0.42f, 0.5f, 0.6f), 0.55f, 0.35f);
        var dark = Look.Lit(new Color(0.13f, 0.14f, 0.16f), 0.4f, 0.4f);
        float r = 1.5f, h = 2.8f, y0 = 0.6f;  // tank çapı, gövde boyu, gövdenin başladığı yükseklik [m]
        Prim(PrimitiveType.Cylinder, "Tank", root, p + Vector3.up * (y0 + h / 2), new Vector3(r, h / 2, r), paint);
        Prim(PrimitiveType.Sphere, "Kubbe", root, p + Vector3.up * (y0 + h), new Vector3(r, r * 0.5f, r), paint);
        Prim(PrimitiveType.Sphere, "Taban", root, p + Vector3.up * y0, new Vector3(r, r * 0.5f, r), paint);
        for (int k = 0; k < 3; k++)  // ayaklar
        {
            var q = Quaternion.Euler(0, k * 120f + 30f, 0) * Vector3.forward * r * 0.38f;
            Prim(PrimitiveType.Cylinder, "Ayak", root, p + q + Vector3.up * 0.3f, new Vector3(0.1f, 0.3f, 0.1f), dark);
        }
        // Cam gözetleme borusu ve içinde basınçla yükselen ışıklı sütun kurulur.
        var gp = new Vector3(p.x - 0.35f, 0f, p.z - r / 2 - 0.18f);
        Prim(PrimitiveType.Cylinder, "Cam", root, gp + Vector3.up * (GlassBottom + GlassHeight / 2), new Vector3(0.22f, GlassHeight / 2, 0.22f),
             Look.Glass(new Color(0.8f, 0.9f, 1f, 0.18f)));
        foreach (float yy in new[] { GlassBottom - 0.05f, GlassBottom + GlassHeight + 0.05f })
            Prim(PrimitiveType.Cylinder, "Bağlantı", root, gp + Vector3.up * yy, new Vector3(0.3f, 0.05f, 0.3f), dark);
        glassFillMat = Look.Glow(LoadColor, 1.4f);
        glassFill = Prim(PrimitiveType.Cylinder, "Basınç", root, gp + Vector3.up * GlassBottom, new Vector3(0.15f, 0.5f, 0.15f), glassFillMat).transform;
        Prim(PrimitiveType.Cylinder, "6bar", root, gp + Vector3.up * (GlassBottom + TankY(data.pMin)), new Vector3(0.3f, 0.012f, 0.3f), Look.Glow(FailColor, 2.5f));
        Prim(PrimitiveType.Cylinder, "6.2bar", root, gp + Vector3.up * (GlassBottom + TankY(data.pMargin)), new Vector3(0.27f, 0.008f, 0.27f), Look.Glow(UnloadColor, 2f));

        // Manometrenin kadranı ve dönen ibresi kurulur.
        var gpos = new Vector3(p.x + 0.35f, y0 + h - 0.35f, p.z - r / 2 - 0.04f);
        Prim(PrimitiveType.Cylinder, "Manometre", root, gpos + new Vector3(0, 0, 0.04f), new Vector3(0.62f, 0.04f, 0.62f), dark)
            .transform.localRotation = Quaternion.Euler(90, 0, 0);
        Prim(PrimitiveType.Quad, "Kadran", root, gpos - new Vector3(0, 0, 0.005f), new Vector3(0.56f, 0.56f, 1f), Look.Dial());
        needle = new GameObject("İbre").transform;
        needle.SetParent(root, false);
        needle.localPosition = gpos - new Vector3(0, 0, 0.015f);
        Prim(PrimitiveType.Cube, "İğne", needle, new Vector3(0, 0.1f, 0), new Vector3(0.02f, 0.22f, 0.01f), Look.Glow(new Color(1f, 0.2f, 0.1f), 1.2f));
        Prim(PrimitiveType.Cylinder, "Mil", needle, Vector3.zero, new Vector3(0.05f, 0.01f, 0.05f), dark).transform.localRotation = Quaternion.Euler(90, 0, 0);

        // Emniyet ventili basınç üst sınırı (pMax, 8 bar) geçince üfler.
        var top = p + Vector3.up * (y0 + h + r * 0.5f);
        Prim(PrimitiveType.Cylinder, "Ventil", root, top + Vector3.up * 0.1f, new Vector3(0.12f, 0.12f, 0.12f), Look.Lit(new Color(0.8f, 0.65f, 0.2f), 0.6f, 0.8f));
        safetyValve = Particles("Ventil buharı", root, top + Vector3.up * 0.25f, new Color(0.95f, 0.97f, 1f, 0.5f),
                                size: 0.25f, life: 1.2f, speed: 2.5f, grow: 3f, rate: 0f, max: 100);
        tankLabelPos = root.TransformPoint(top + Vector3.up * 0.75f);
    }

    void BuildFactory(Vector3 p)
    {
        var wall = Look.Concrete(new Color(0.78f, 0.72f, 0.64f), new Vector2(1.5f, 1f));
        var roof = Look.Lit(new Color(0.3f, 0.33f, 0.38f), 0.5f, 0.6f);
        var dark = Look.Lit(new Color(0.13f, 0.14f, 0.16f), 0.4f, 0.4f);
        var size = new Vector3(2.6f, 2.1f, 2.8f);
        Prim(PrimitiveType.Cube, "Fabrika", root, p + Vector3.up * size.y / 2, size, wall);
        for (int k = 0; k < 3; k++)  // testere dişi çatı
            Prim(PrimitiveType.Cube, "Çatı", root, p + new Vector3(0f, size.y + 0.25f, -size.z / 2 + 0.47f + k * 0.93f), new Vector3(size.x + 0.15f, 0.08f, 1.05f), roof)
                .transform.localRotation = Quaternion.Euler(-28f, 0f, 0f);
        factoryWindows = Look.Glow(new Color(1f, 0.82f, 0.55f), 2f);
        for (int k = 0; k < 3; k++)
            Prim(PrimitiveType.Quad, "Pencere", root, p + new Vector3(-0.85f + k * 0.85f, size.y * 0.68f, -size.z / 2 - 0.01f), new Vector3(0.6f, 0.4f, 1f), factoryWindows);
        Prim(PrimitiveType.Quad, "Kapı", root, p + new Vector3(0.75f, 0.6f, -size.z / 2 - 0.012f), new Vector3(0.8f, 1.2f, 1f), Look.Stripes(new Vector2(2f, 3f)));
        Prim(PrimitiveType.Cylinder, "Baca", root, p + new Vector3(0.8f, size.y + 0.9f, 0.7f), new Vector3(0.3f, 0.7f, 0.3f), dark);
        chimney = Particles("Duman", root, p + new Vector3(0.8f, size.y + 1.7f, 0.7f), new Color(0.75f, 0.76f, 0.78f, 0.25f),
                            size: 0.4f, life: 3.5f, speed: 0.8f, grow: 3f, rate: 0f, max: 150);

        // Döner uyarı lambası basınç 6 bar altına inince kırmızı yanar ve döner.
        var bpos = p + new Vector3(-0.8f, size.y + 0.45f, -0.6f);
        beaconMat = Look.Glow(FailColor, 0f);
        Prim(PrimitiveType.Sphere, "İkaz", root, bpos, new Vector3(0.22f, 0.22f, 0.22f), beaconMat);
        beaconLight = new GameObject("İkaz ışığı").AddComponent<Light>();
        beaconLight.type = LightType.Spot;
        beaconLight.color = FailColor;
        beaconLight.spotAngle = 50f;
        beaconLight.range = 9f;
        beaconLight.intensity = 0f;
        beaconLight.transform.SetParent(root, false);
        beaconLight.transform.localPosition = bpos;

        // Konveyör basınçlı havayla çalışan üretim hattını temsil eder. Hızı talebe bağlıdır, basınç düşerse durur.
        float bz = p.z - size.z / 2 - 0.8f;
        beltStart = new Vector3(p.x - beltLen / 2, 0.7f, bz);
        Prim(PrimitiveType.Cube, "Bant", root, new Vector3(p.x, 0.55f, bz), new Vector3(beltLen, 0.08f, 0.6f), Look.Lit(new Color(0.08f, 0.08f, 0.09f), 0.2f));
        foreach (float dx in new[] { -1.4f, 0f, 1.4f })
            Prim(PrimitiveType.Cube, "BantAyak", root, new Vector3(p.x + dx, 0.26f, bz), new Vector3(0.08f, 0.52f, 0.5f), dark);
        var box = Look.Lit(new Color(0.72f, 0.55f, 0.35f), 0.2f);
        for (int k = 0; k < 5; k++)
            boxes.Add(Prim(PrimitiveType.Cube, "Koli", root, Vector3.zero, new Vector3(0.28f, 0.22f, 0.3f), box).transform);
        // Bandın üstüne havayla çalışan (pnömatik), inip kalkan bir pres konur.
        pistonX = p.x - 0.5f;
        pistonZ = bz;
        Prim(PrimitiveType.Cube, "PresGövde", root, new Vector3(pistonX, 1.55f, bz), new Vector3(0.5f, 0.35f, 0.5f), Look.Lit(new Color(0.85f, 0.5f, 0.15f), 0.5f, 0.3f));
        Prim(PrimitiveType.Cube, "PresAyak", root, new Vector3(pistonX, 1.15f, bz + 0.32f), new Vector3(0.12f, 1.2f, 0.12f), dark);
        piston = Prim(PrimitiveType.Cylinder, "Piston", root, new Vector3(pistonX, 1.2f, bz), new Vector3(0.12f, 0.25f, 0.12f),
                      Look.Lit(new Color(0.8f, 0.82f, 0.85f), 0.85f, 0.9f)).transform;
        factoryLabelPos = root.TransformPoint(p + Vector3.up * 3.6f);
    }

    // Arayüzden başka model ya da senaryo seçilince yeni kaydı bağlar. Sahnedeki objeler aynı kalır.
    public void SetRun(ReplayData d, ReplayRun r)
    {
        data = d;
        run = r;
        lastStep = -1;
    }

    // t adımında arızalı bir kompresör olup olmadığına bakar. Arıza aralığı bölüm başından saat olarak verilir.
    public bool Failed(float t)
    {
        float h = t * data.dt / 3600f;
        return data.failComp >= 0 && h >= data.shadeStart && h < data.shadeEnd;
    }

    public Vector3 Origin => root.position;  // istasyonun dünyadaki merkezi
    public Vector3 CompLabel(int k) => comps[k].labelPos;  // kompresör etiketinin konumu (dünya)

    static float TankY(float p) => Mathf.Clamp01((p - TankLow) / (TankHigh - TankLow)) * GlassHeight;

    // Gece ve gündüz ışığını ayarlar. night 0 ise gündüz, 1 ise gece. Arka duvardaki pencereler buna göre parlar.
    public void SetLighting(float night)
    {
        Look.SetGlow(wallWindows, Color.Lerp(new Color(0.7f, 0.85f, 1f), new Color(0.12f, 0.16f, 0.3f), night), Mathf.Lerp(1.4f, 0.6f, night));
        // Floresanlar hep yanar. Gündüz güneş baskın olduğu için geride kalırlar, gece salonu onlar aydınlatır.
        foreach (var l in ceilingLights) l.intensity = Mathf.Lerp(10f, 20f, night);
    }

    // Sahneyi t anına getirir. t adım cinsinden zamandır ve ondalıklı olabilir. dtReal son kareden beri geçen
    // gerçek süre [s], speed oynatma hızıdır.
    public void UpdateView(float t, float dtReal, float speed)
    {
        int n = run.pressure.Length;
        int i = Mathf.Clamp(Mathf.FloorToInt(t), 0, n - 1);
        int j = Mathf.Min(i + 1, n - 1);
        float p = Mathf.Lerp(run.pressure[i], run.pressure[j], Mathf.Clamp01(t - i));
        bool failed = Failed(t);
        bool blink = Mathf.Repeat(Time.time * 2f, 1f) < 0.5f;

        DetectEvents(i, speed);

        int[] states = { run.k1[i], run.k2[i], run.vsdSpeed[i] > 0f ? 2 : 0 };
        float[] temps = { run.temp1[i], run.temp2[i], run.temp3[i] };
        float flowTotal = 0f;
        for (int k = 0; k < 3; k++)
        {
            var c = comps[k];
            bool vsd = k == 2, isFail = failed && k == data.failComp;
            float sp = vsd ? run.vsdSpeed[i] : (states[k] == 0 ? 0f : 1f);  // motor devri (0-1)
            Color col = isFail ? FailColor : states[k] == 2 ? LoadColor : states[k] == 1 ? UnloadColor : OffColor;
            float glow = isFail ? (blink ? 3f : 0.3f) : states[k] == 0 ? 0f : vsd ? 1.2f + 1.6f * sp : 2.6f;
            Look.SetGlow(c.strip, col, glow);
            Look.SetGlow(c.screen, col, glow * 0.6f + 0.15f);
            Look.SetGlow(c.tower[0], FailColor, isFail && blink ? 4f : 0f);
            Look.SetGlow(c.tower[1], UnloadColor, !isFail && states[k] == 1 ? 3.5f : 0f);
            Look.SetGlow(c.tower[2], LoadColor, !isFail && states[k] == 2 ? 3.5f : 0f);

            c.fanAngle += sp * 1100f * dtReal;  // fan devirle orantılı döner
            c.rotor.localRotation = Quaternion.Euler(0f, c.fanAngle, 0f);
            // Çalışan motor hafifçe titrer (yükteyken daha çok)
            float amp = states[k] == 2 ? 0.006f : states[k] == 1 ? 0.003f : 0f;
            var bp = c.shake.localPosition;
            c.shake.localPosition = new Vector3(bp.x, 0.14f + amp * Mathf.Sin(Time.time * 83f + k), bp.z);

            // Sıcaklık göstergesi ortam sıcaklığında kısa ve mavi, trip sıcaklığında dolu ve kırmızı olur
            float h = Mathf.Clamp01((temps[k] - data.ambientC) / (data.tripC - data.ambientC));
            float fillH = Mathf.Max(0.01f, h * c.tempH);
            var tp = c.tempFill.localPosition;
            c.tempFill.localScale = new Vector3(0.04f, fillH, 1f);
            c.tempFill.localPosition = new Vector3(tp.x, c.tempBottom + fillH / 2, tp.z);
            Look.SetGlow(c.tempMat, Color.Lerp(new Color(0.2f, 0.55f, 1f), FailColor, h), 1.6f);

            var em = c.haze.emission;
            em.rateOverTime = sp * (states[k] == 2 ? 6f : 2f);

            float flow = isFail ? 0f : vsd ? data.compFlow[k] * sp : (states[k] == 2 ? data.compFlow[k] : 0f);
            flowTotal += flow;
            Animate(c.path, flow, dtReal);
        }

        // Tankta gözetleme borusundaki sütun, rengi ve manometre ibresi güncellenir.
        float y = Mathf.Max(TankY(p), 0.02f);
        glassFill.localScale = new Vector3(0.15f, y / 2, 0.15f);
        var gp = glassFill.localPosition;
        glassFill.localPosition = new Vector3(gp.x, GlassBottom + y / 2, gp.z);
        Look.SetGlow(glassFillMat, PressureColor(p), 1.6f);
        needle.localRotation = Quaternion.Euler(0f, 0f, -(-135f + 270f * Mathf.Clamp(p, 0f, Look.DialMaxBar) / Look.DialMaxBar));
        var sv = safetyValve.emission;
        sv.rateOverTime = p > data.pMax ? 40f : 0f;

        // Fabrikada konveyör ve pres talep kadar hızlı çalışır. Basınç 6 bar altına inince üretim durur.
        bool low = p < data.pMin;
        float d = run.demand[i];
        float prod = low ? 0f : Mathf.Clamp01(d / 25f);  // üretim hızı (0-1)
        Look.SetGlow(factoryWindows, low ? FailColor : new Color(1f, 0.82f, 0.55f), low ? (blink ? 2.5f : 0.4f) : 0.8f + 1.6f * prod);
        Look.SetGlow(beaconMat, FailColor, low ? 4f : 0f);
        beaconLight.intensity = low ? 12f : 0f;
        beaconLight.transform.localRotation = Quaternion.Euler(20f, Time.time * 360f, 0f);
        beltPos += prod * 0.8f * dtReal;
        for (int k = 0; k < boxes.Count; k++)
            boxes[k].localPosition = beltStart + Vector3.right * Mathf.Repeat(beltPos + k * beltLen / boxes.Count, beltLen);
        pistonPhase += prod * 2.2f * dtReal;
        piston.localPosition = new Vector3(pistonX, 1.15f - 0.22f * Mathf.Max(0f, Mathf.Sin(pistonPhase * Mathf.PI * 2f)), pistonZ);
        var ch = chimney.emission;
        ch.rateOverTime = 2f + 10f * prod;
        Animate(demandPath, d, dtReal);

        sfx.UpdateLoops(states, failed && data.failComp == 2 ? 0f : run.vsdSpeed[i], flowTotal, dtReal);
    }

    // Adım ilerledikçe durum değişikliklerini yakalar. Boşa almada tahliye buharı çıkar ve ses çalar. Kalkışta ve
    // basınç 6 bar altına inince de ses çalar. İleri ya da geri sarmadaki büyük sıçramalar olay sayılmaz.
    void DetectEvents(int i, float speed)
    {
        if (lastStep < 0 || i < lastStep || i - lastStep > 60) { lastStep = i; return; }
        for (int s = lastStep + 1; s <= i; s++)
        {
            int[] prev = { run.k1[s - 1], run.k2[s - 1], run.vsdSpeed[s - 1] > 0 ? 2 : 0 };
            int[] now = { run.k1[s], run.k2[s], run.vsdSpeed[s] > 0 ? 2 : 0 };
            for (int k = 0; k < 3; k++)
            {
                if (k < 2 && prev[k] == 2 && now[k] != 2)  // yükten çıkan kompresörün tahliye vanası üfler
                {
                    comps[k].mist.Emit(35);
                    sfx.Blowoff(speed);
                }
                if (prev[k] == 0 && now[k] != 0) sfx.MotorStart(speed);
            }
            if (run.pressure[s - 1] >= data.pMin && run.pressure[s] < data.pMin) sfx.Alarm();
        }
        lastStep = i;
    }

    public Color PressureColor(float p) =>
        p < data.pMin ? new Color(1f, 0.25f, 0.2f) :
        p < data.pMargin ? new Color(1f, 0.72f, 0.15f) : new Color(0.3f, 0.7f, 1f);

    void Animate(FlowPath path, float flow, float dtReal)
    {
        bool on = flow > 0.05f;
        path.phase += flow / 13f * 2.2f * dtReal;  // 13 m³/dk akışta halkalar saniyede 2,2 m ilerler
        float spacing = path.length / path.rings.Length;
        for (int b = 0; b < path.rings.Length; b++)
        {
            var ring = path.rings[b];
            ring.gameObject.SetActive(on);
            if (!on) continue;
            PointAt(path, Mathf.Repeat(path.phase + b * spacing, path.length), out var pos, out var dir);
            ring.localPosition = pos;
            ring.localRotation = Quaternion.FromToRotation(Vector3.up, dir);
        }
    }

    static void PointAt(FlowPath path, float d, out Vector3 pos, out Vector3 dir)
    {
        for (int k = 0; k < path.pts.Length - 1; k++)
        {
            float seg = Vector3.Distance(path.pts[k], path.pts[k + 1]);
            dir = (path.pts[k + 1] - path.pts[k]).normalized;
            if (d <= seg) { pos = path.pts[k] + dir * d; return; }
            d -= seg;
        }
        pos = path.pts[path.pts.Length - 1];
        dir = Vector3.up;
    }

    FlowPath MakePath(Vector3[] pts, int count)
    {
        var path = new FlowPath { pts = pts, rings = new Transform[count] };
        for (int k = 0; k < pts.Length - 1; k++) path.length += Vector3.Distance(pts[k], pts[k + 1]);
        var mat = Look.Glow(AirColor, 3f);
        for (int b = 0; b < count; b++)
            path.rings[b] = Prim(PrimitiveType.Cylinder, "Hava", root, Vector3.zero, new Vector3(0.24f, 0.035f, 0.24f), mat).transform;
        return path;
    }

    // Aşağıdaki şekil yardımcılarında konumlar parent'a göre yerel verilir.
    void Pipe(Vector3 a, Vector3 b)
    {
        var mat = Look.Lit(PipeColor, 0.65f, 0.55f);
        var go = Prim(PrimitiveType.Cylinder, "Boru", root, (a + b) / 2, new Vector3(0.17f, Vector3.Distance(a, b) / 2, 0.17f), mat);
        go.transform.localRotation = Quaternion.FromToRotation(Vector3.up, b - a);
        var flange = Look.Lit(new Color(0.25f, 0.27f, 0.3f), 0.5f, 0.7f);
        foreach (var e in new[] { a, b })
        {
            Prim(PrimitiveType.Sphere, "Dirsek", root, e, Vector3.one * 0.19f, mat);
            var f = Prim(PrimitiveType.Cylinder, "Flanş", root, Vector3.MoveTowards(e, (a + b) / 2, 0.12f), new Vector3(0.26f, 0.025f, 0.26f), flange);
            f.transform.localRotation = Quaternion.FromToRotation(Vector3.up, b - a);
        }
    }

    static GameObject Prim(PrimitiveType type, string name, Transform parent, Vector3 pos, Vector3 scale, Material mat)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = name;
        Object.Destroy(go.GetComponent<Collider>());
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        go.transform.localScale = scale;
        go.GetComponent<Renderer>().sharedMaterial = mat;
        return go;
    }

    // Parçacık sistemi kurar. rate 0'dan büyükse parçacık sürekli çıkar, 0 ise yalnızca Emit() çağrılınca bir kerede çıkar.
    // grow, parçacığın ömrü bitince ulaştığı boyut çarpanıdır.
    static ParticleSystem Particles(string name, Transform parent, Vector3 pos, Color color, float size, float life,
                                    float speed, float grow, float rate, int max)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        go.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);  // koni yukarı baksın diye döndürülür
        var ps = go.AddComponent<ParticleSystem>();
        ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        var main = ps.main;
        main.duration = 1f;
        main.loop = true;
        main.startLifetime = new ParticleSystem.MinMaxCurve(life * 0.7f, life);
        main.startSpeed = new ParticleSystem.MinMaxCurve(speed * 0.6f, speed);
        main.startSize = new ParticleSystem.MinMaxCurve(size * 0.7f, size);
        main.startColor = color;
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.maxParticles = max;
        main.gravityModifier = -0.05f;
        var em = ps.emission;
        em.rateOverTime = rate;
        var shape = ps.shape;
        shape.shapeType = ParticleSystemShapeType.Cone;
        shape.angle = 18f;
        shape.radius = 0.1f;
        var col = ps.colorOverLifetime;
        col.enabled = true;
        var g = new Gradient();
        g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                  new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(1f, 0.15f), new GradientAlphaKey(0f, 1f) });
        col.color = g;
        var sz = ps.sizeOverLifetime;
        sz.enabled = true;
        sz.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, grow));
        var lim = ps.limitVelocityOverLifetime;
        lim.enabled = true;
        lim.dampen = 0.08f;
        lim.limit = 0.4f;
        var r = go.GetComponent<ParticleSystemRenderer>();
        r.sharedMaterial = Look.Particle();
        r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        ps.Play();
        return ps;
    }
}
