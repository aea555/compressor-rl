using UnityEngine;

// Bu sınıf arayüzün yazı tipini, renklerini, panel ve buton stillerini tutar. Yuvarlak köşeler küçük dokularla
// çizildiği için panel büyüse de köşeler bozulmaz.
public static class Theme
{
    public static readonly Color Bg = new Color(0.055f, 0.066f, 0.086f);            // en koyu zemin
    public static readonly Color PanelFill = new Color(0.05f, 0.06f, 0.08f, 0.9f); // panel dolgusu
    public static readonly Color Line = new Color(1f, 1f, 1f, 0.08f);               // ince kenarlık
    public static readonly Color Text = new Color(0.93f, 0.95f, 0.98f);             // ana yazı
    public static readonly Color Muted = new Color(0.58f, 0.63f, 0.7f);             // ikincil yazı
    public static readonly Color Accent = new Color(0.33f, 0.62f, 1f);              // vurgu (seçili buton, ilerleme)
    public static readonly Color Good = new Color(0.36f, 0.86f, 0.53f);             // tasarruf ya da iyi sonuç
    public static readonly Color Bad = new Color(1f, 0.42f, 0.36f);                 // kayıp ya da kötü sonuç
    public static readonly Color Warn = new Color(1f, 0.76f, 0.25f);                // uyarı

    public static Font Regular, Medium, SemiBold, Bold;  // Barlow yazı tipinin kalınlıkları
    public static GUIStyle Panel, Card, Pill, Btn, BtnOn, H1, H2, Body, Small, Tiny, Caps, Value, Big, Center;
    public static Texture2D White, Round;  // düz dikdörtgen ve yuvarlak doku, yuvarlak olan nokta ve oynatma başı için
    static Texture2D panelTex, cardTex, pillTex, btnTex, btnHoverTex, btnOnTex, btnOnHoverTex, barTex;
    static GUIStyle bar;  // yuvarlak uçlu dolgu (çubuklar, etiket kutuları)
    static float builtFor = -1f;           // stillerin hangi ölçek için üretildiği

    public static void Ensure(float ui)
    {
        if (Regular == null)
        {
            Regular = Resources.Load<Font>("Fonts/Barlow-Regular");
            Medium = Resources.Load<Font>("Fonts/Barlow-Medium");
            SemiBold = Resources.Load<Font>("Fonts/Barlow-SemiBold");
            Bold = Resources.Load<Font>("Fonts/Barlow-Bold");
            White = Texture2D.whiteTexture;
            Round = RoundTex(32, 16, Color.white, Color.clear);
            panelTex = RoundTex(32, 10, PanelFill, Line);
            cardTex = RoundTex(32, 8, new Color(1f, 1f, 1f, 0.045f), new Color(1f, 1f, 1f, 0.06f));
            pillTex = RoundTex(32, 15, new Color(0.04f, 0.05f, 0.07f, 0.85f), new Color(1f, 1f, 1f, 0.1f));
            btnTex = RoundTex(32, 8, new Color(1f, 1f, 1f, 0.06f), new Color(1f, 1f, 1f, 0.09f));
            btnHoverTex = RoundTex(32, 8, new Color(1f, 1f, 1f, 0.12f), new Color(1f, 1f, 1f, 0.16f));
            btnOnTex = RoundTex(32, 8, Accent * new Color(1, 1, 1, 0.9f), new Color(0.6f, 0.8f, 1f, 0.6f));
            btnOnHoverTex = RoundTex(32, 8, Accent, new Color(0.7f, 0.85f, 1f, 0.8f));
            barTex = RoundTex(16, 6, Color.white, Color.clear);
            bar = Box(barTex, 6);
        }
        if (Mathf.Approximately(ui, builtFor)) return;
        builtFor = ui;

        Panel = Box(panelTex, 10);
        Card = Box(cardTex, 8);
        Pill = Box(pillTex, 15);
        Btn = new GUIStyle { font = Medium, fontSize = Px(15, ui), alignment = TextAnchor.MiddleCenter, richText = true,
                             border = new RectOffset(8, 8, 8, 8), padding = new RectOffset(Px(12, ui), Px(12, ui), 0, 0) };
        Btn.normal.background = btnTex; Btn.normal.textColor = new Color(0.82f, 0.86f, 0.92f);
        Btn.hover.background = btnHoverTex; Btn.hover.textColor = Color.white;
        Btn.active.background = btnOnTex; Btn.active.textColor = Color.white;
        BtnOn = new GUIStyle(Btn) { font = SemiBold };
        BtnOn.normal.background = btnOnTex; BtnOn.normal.textColor = Color.white;
        BtnOn.hover.background = btnOnHoverTex; BtnOn.hover.textColor = Color.white;

        H1 = Label(SemiBold, 26, ui, Text);
        H2 = Label(SemiBold, 19, ui, Text);
        Body = Label(Regular, 17, ui, Text);
        Small = Label(Regular, 15, ui, Text);
        Tiny = Label(Regular, 13, ui, Muted);
        Caps = Label(SemiBold, 11, ui, Muted);  // küçük başlıklar (BÜYÜK HARF)
        Value = Label(SemiBold, 22, ui, Text);   // gösterge değerleri
        Big = Label(Bold, 34, ui, Text);
        Center = new GUIStyle(Big) { alignment = TextAnchor.MiddleCenter };
    }

    static int Px(float size, float ui) => Mathf.Max(8, Mathf.RoundToInt(size * ui));

    static GUIStyle Label(Font f, float size, float ui, Color c)
    {
        var s = new GUIStyle { font = f, fontSize = Px(size, ui), richText = true, wordWrap = false, clipping = TextClipping.Overflow };
        s.normal.textColor = c;
        return s;
    }

    static GUIStyle Box(Texture2D tex, int r)
    {
        var s = new GUIStyle { border = new RectOffset(r, r, r, r) };
        s.normal.background = tex;
        return s;
    }

    // Kenarı yumuşatılmış, ince kenarlıklı ve yuvarlak köşeli bir dikdörtgen dokusu üretir
    static Texture2D RoundTex(int n, int r, Color fill, Color border)
    {
        var tex = new Texture2D(n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                // Piksel köşedeyse köşe merkezine, değilse kenara olan uzaklığı hesaplanır.
                float cx = Mathf.Clamp(x + 0.5f, r, n - r), cy = Mathf.Clamp(y + 0.5f, r, n - r);
                float d = r - Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(cx, cy));  // içe doğru mesafe
                float a = Mathf.Clamp01(d + 0.5f);                    // kenar yumuşatma
                float b = Mathf.Clamp01(1.5f - d);                    // 1 piksellik kenarlık
                Color c = Color.Lerp(fill, new Color(border.r, border.g, border.b, Mathf.Max(border.a, fill.a)), border.a > 0 ? b : 0f);
                c.a *= a;
                tex.SetPixel(x, y, Tint(c));  // Tint'teki sebeple dokunun rengi de lineere çevrilir
            }
        tex.Apply();
        return tex;
    }

    // Proje lineer renk uzayında olduğu için IMGUI renkleri olduğundan açık gösterir.
    // Renk önce lineere çevrilince ekranda doğru ton görünür.
    static Color Tint(Color c) => QualitySettings.activeColorSpace == ColorSpace.Linear ? new Color(c.linear.r, c.linear.g, c.linear.b, c.a) : c;

    // Düz renkli dikdörtgen çizer
    public static void Rect(Rect r, Color c)
    {
        var old = GUI.color;
        GUI.color = Tint(c);
        GUI.DrawTexture(r, White, ScaleMode.StretchToFill, true);
        GUI.color = old;
    }

    // Yuvarlak köşeli renkli dolgu çizer. Köşeler her boyutta düzgün kalır.
    public static void RoundRect(Rect r, Color c)
    {
        if (Event.current.type != EventType.Repaint) return;
        var old = GUI.backgroundColor;
        GUI.backgroundColor = Tint(c);
        bar.Draw(r, false, false, false, false);
        GUI.backgroundColor = old;
    }

    // Renkli yuvarlak nokta çizer
    public static void Dot(Rect r, Color c)
    {
        var old = GUI.color;
        GUI.color = Tint(c);
        GUI.DrawTexture(r, Round, ScaleMode.StretchToFill, true);
        GUI.color = old;
    }

    public static string Hex(Color c) => ColorUtility.ToHtmlStringRGB(c);
}
