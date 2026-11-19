using UnityEngine;

// Bu sınıf materyal ve doku üreten yardımcı fonksiyonları içerir. Beton zemin, manometre kadranı, duman parçacığı ve
// uyarı şeridi dokuları görsel dosyası kullanılmadan kodla çizilir.
public static class Look
{
    static Material lit, emissive, glass, particle;  // Resources klasöründeki temel materyaller (BuildTools üretir)
    static Texture2D concreteTex, softTex, dialTex, stripeTex;

    public static void Init()
    {
        lit = Resources.Load<Material>("Lit");
        emissive = Resources.Load<Material>("Emissive");
        glass = Resources.Load<Material>("Glass");
        particle = Resources.Load<Material>("Particle");
        concreteTex = MakeConcrete();
        softTex = MakeSoft();
        dialTex = MakeDial();
        stripeTex = MakeStripes();
    }

    // Normal bir yüzey materyali üretir. smooth parlaklığı (0 mat, 1 ayna), metal metalikliği (0-1) belirler.
    public static Material Lit(Color c, float smooth = 0.45f, float metal = 0f)
    {
        var m = new Material(lit) { color = c };
        m.SetFloat("_Smoothness", smooth);
        m.SetFloat("_Metallic", metal);
        return m;
    }

    // Lamba ve ekranlar için ışık saçan bir yüzey üretir. intensity 1'den büyükse parlama oluşur.
    public static Material Glow(Color c, float intensity)
    {
        var m = new Material(emissive);
        SetGlow(m, c, intensity);
        return m;
    }

    public static void SetGlow(Material m, Color c, float intensity)
    {
        m.color = new Color(c.r * 0.3f, c.g * 0.3f, c.b * 0.3f);
        m.SetColor("_EmissionColor", new Color(c.r, c.g, c.b) * intensity);
    }

    public static Material Glass(Color c)
    {
        var m = new Material(glass) { color = c };
        return m;
    }

    public static Material Particle()
    {
        var m = new Material(particle);
        m.SetTexture("_BaseMap", softTex);
        return m;
    }

    public static Material Concrete(Color tint, Vector2 tiling)
    {
        var m = Lit(tint, 0.18f);
        m.SetTexture("_BaseMap", concreteTex);
        m.mainTextureScale = tiling;
        return m;
    }

    public static Material Stripes(Vector2 tiling)
    {
        var m = Lit(Color.white, 0.3f);
        m.SetTexture("_BaseMap", stripeTex);
        m.mainTextureScale = tiling;
        return m;
    }

    public static Material Dial()
    {
        var m = Lit(Color.white, 0.7f);
        m.SetTexture("_BaseMap", dialTex);
        return m;
    }

    // Gürültü ve derz çizgileriyle beton dokusu çizer
    static Texture2D MakeConcrete()
    {
        const int N = 256;
        var tex = new Texture2D(N, N, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat };
        var px = new Color[N * N];
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
            {
                float n = 0f, amp = 0.5f, f = 4f / N;
                for (int o = 0; o < 4; o++, amp *= 0.5f, f *= 2f)  // doku kesintisiz döşensin diye gürültü periyodik alınır
                    n += amp * Mathf.PerlinNoise(Mathf.Repeat(x * f, 64f) + o * 17f, Mathf.Repeat(y * f, 64f) + o * 31f);
                float speck = Mathf.PerlinNoise(x * 0.9f, y * 0.9f) > 0.82f ? -0.06f : 0f;  // küçük lekeler
                float joint = (x % 128 < 2 || y % 128 < 2) ? -0.18f : 0f;  // derz çizgisi
                float v = 0.78f + (n - 0.5f) * 0.22f + speck + joint;
                px[y * N + x] = new Color(v, v, v * 1.01f, 1f);
            }
        tex.SetPixels(px);
        tex.Apply();
        return tex;
    }

    // Parçacıklar için ortası opak, kenarları saydam bir yuvarlak çizer
    static Texture2D MakeSoft()
    {
        const int N = 64;
        var tex = new Texture2D(N, N, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(N / 2f, N / 2f)) / (N / 2f);
                float a = Mathf.Clamp01(1f - d);
                tex.SetPixel(x, y, new Color(1, 1, 1, a * a * (3 - 2 * a)));
            }
        tex.Apply();
        return tex;
    }

    // Sarı-siyah çapraz uyarı şeridi çizer
    static Texture2D MakeStripes()
    {
        const int N = 64;
        var tex = new Texture2D(N, N, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat };
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
                tex.SetPixel(x, y, ((x + y) / 16) % 2 == 0 ? new Color(0.95f, 0.75f, 0.1f) : new Color(0.08f, 0.08f, 0.08f));
        tex.Apply();
        return tex;
    }

    // MakeDial 0-10 bar arasını gösteren manometre kadranını çizer. Renkli yay 6 bar altında kırmızı, 6-6,2 bar arasında sarı,
    // 6,2-8 bar arasında yeşil, 8,5 bar üstünde yine kırmızıdır. 8-8,5 bar arası renksiz kalır.
    public const float DialMaxBar = 10f;
    static Texture2D MakeDial()
    {
        const int N = 256;
        var tex = new Texture2D(N, N, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp };
        var c = new Vector2(N / 2f, N / 2f);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++)
            {
                var d = new Vector2(x + 0.5f, y + 0.5f) - c;
                float r = d.magnitude / (N / 2f);                       // merkezden uzaklık (0-1)
                float ang = Mathf.Atan2(d.x, d.y) * Mathf.Rad2Deg;      // yukarıdan saat yönünde açı
                float bar = (ang + 135f) / 270f * DialMaxBar;            // bu açının karşılık geldiği basınç
                Color col = new Color(0.93f, 0.93f, 0.9f);
                if (r > 0.97f) col = new Color(0, 0, 0, 0);              // kadran dışı
                else if (r > 0.88f) col = new Color(0.15f, 0.15f, 0.17f); // çerçeve
                else if (bar >= 0 && bar <= DialMaxBar)
                {
                    if (r > 0.72f && r < 0.84f)                           // renkli bölge yayı
                        col = bar < 6f ? new Color(0.85f, 0.2f, 0.2f) : bar < 6.2f ? new Color(0.95f, 0.75f, 0.15f)
                            : bar < 8f ? new Color(0.25f, 0.7f, 0.35f) : bar > 8.5f ? new Color(0.85f, 0.2f, 0.2f) : col;
                    float frac = bar - Mathf.Round(bar);                  // her 1 bar'da bir çentik
                    if (r > 0.6f && r < 0.86f && Mathf.Abs(frac) < 0.04f) col = new Color(0.1f, 0.1f, 0.1f);
                }
                tex.SetPixel(x, y, col);
            }
        tex.Apply();
        return tex;
    }
}
