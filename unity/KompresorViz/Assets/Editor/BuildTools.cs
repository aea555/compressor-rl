using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// Bu sınıf editörü açmadan, komut satırından projeyi hazırlayıp derler.
//   Unity -batchmode -quit -projectPath unity/KompresorViz -executeMethod BuildTools.BuildMac
// BuildMac URP ayarlarını yapar, Resources klasörüne temel materyalleri koyar, sahneyi kurar ve macOS uygulamasını derler.
public static class BuildTools
{
    const string ScenePath = "Assets/Scenes/Main.unity";
    const string RenderDir = "Assets/Settings";

    static void Setup()
    {
        PlayerSettings.colorSpace = ColorSpace.Linear;  // ışık hesabı fiziksel olarak doğru olsun diye seçilir, URP de bunu önerir
        var pipeline = SetupPipeline();
        var profile = SetupVolumeProfile();
        SetupMaterials();

        // Sahnede yalnızca kamera, ReplayApp ve post-processing hacmi bulunur. Geri kalan her şeyi ReplayApp çalışırken kurar.
        Directory.CreateDirectory("Assets/Scenes");
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene);
        var cam = new GameObject("Main Camera", typeof(Camera), typeof(AudioListener));
        cam.tag = "MainCamera";
        new GameObject("ReplayApp", typeof(ReplayApp));
        var vol = new GameObject("PostProcess", typeof(Volume)).GetComponent<Volume>();
        vol.isGlobal = true;
        vol.sharedProfile = profile;
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };

        PlayerSettings.productName = "Kompresor Siralama";
        PlayerSettings.companyName = "DeepRL";
        PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
        PlayerSettings.defaultScreenWidth = 1600;
        PlayerSettings.defaultScreenHeight = 900;
        PlayerSettings.resizableWindow = true;
        PlayerSettings.runInBackground = true;  // pencere arka plandayken de oynatma devam eder
        AssetDatabase.SaveAssets();
    }

    // URP varlıklarını oluşturur ve projenin render hattı olarak atar
    static UniversalRenderPipelineAsset SetupPipeline()
    {
        Directory.CreateDirectory(RenderDir);
        string rendererPath = $"{RenderDir}/Renderer.asset", pipePath = $"{RenderDir}/URP.asset";
        AssetDatabase.DeleteAsset(rendererPath);
        AssetDatabase.DeleteAsset(pipePath);

        var renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
        renderer.postProcessData = AssetDatabase.LoadAssetAtPath<PostProcessData>(
            "Packages/com.unity.render-pipelines.universal/Runtime/Data/PostProcessData.asset");
        AssetDatabase.CreateAsset(renderer, rendererPath);

        // SSAO köşelerde ve temas noktalarında yumuşak gölge çizdiği için objeler zemine oturmuş görünür.
        var ssao = ScriptableObject.CreateInstance<ScreenSpaceAmbientOcclusion>();
        ssao.name = "SSAO";
        AssetDatabase.AddObjectToAsset(ssao, renderer);
        renderer.rendererFeatures.Add(ssao);
        var so = new SerializedObject(ssao);
        Set(so, "m_Settings.Intensity", 1.2f);
        Set(so, "m_Settings.Radius", 0.4f);
        Set(so, "m_Settings.DirectLightingStrength", 0.3f);
        so.ApplyModifiedPropertiesWithoutUndo();
        var rso = new SerializedObject(renderer);
        var map = rso.FindProperty("m_RendererFeatureMap");  // editörün özellik listesiyle eşleştirdiği kimlikler
        if (map != null)
        {
            AssetDatabase.TryGetGUIDAndLocalFileIdentifier(ssao, out _, out long id);
            map.arraySize = 1;
            map.GetArrayElementAtIndex(0).longValue = id;
            rso.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorUtility.SetDirty(renderer);

        var pipe = UniversalRenderPipelineAsset.Create(renderer);
        AssetDatabase.CreateAsset(pipe, pipePath);
        var ps = new SerializedObject(pipe);
        Set(ps, "m_MSAA", 4);                          // kenar yumuşatma (4x MSAA)
        Set(ps, "m_SupportsHDR", true);                // parlayan ışıklar (bloom) için
        Set(ps, "m_ShadowDistance", 45f);              // gölgelerin çizildiği mesafe [m]
        Set(ps, "m_SoftShadowsSupported", true);       // yumuşak gölge kenarları
        Set(ps, "m_MainLightShadowmapResolution", 4096);
        Set(ps, "m_AdditionalLightsPerObjectLimit", 8);
        Set(ps, "m_RenderScale", 1f);
        ps.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(pipe);

        GraphicsSettings.defaultRenderPipeline = pipe;
        for (int q = 0; q < QualitySettings.names.Length; q++)
        {
            QualitySettings.SetQualityLevel(q, false);
            QualitySettings.renderPipeline = pipe;
        }
        AssetDatabase.SaveAssets();
        return pipe;
    }

    // Post-processing ayarlarını yapar. Lambaların parlaması (bloom), ton eşleme, hafif vinyet ve renk ayarı eklenir.
    static VolumeProfile SetupVolumeProfile()
    {
        string path = $"{RenderDir}/PostProcess.asset";
        AssetDatabase.DeleteAsset(path);
        var profile = ScriptableObject.CreateInstance<VolumeProfile>();
        AssetDatabase.CreateAsset(profile, path);

        var bloom = profile.Add<Bloom>(true);
        bloom.threshold.Override(1.0f);
        bloom.intensity.Override(0.9f);
        bloom.scatter.Override(0.65f);
        var tone = profile.Add<Tonemapping>(true);
        tone.mode.Override(TonemappingMode.ACES);
        var vig = profile.Add<Vignette>(true);
        vig.intensity.Override(0.28f);
        vig.smoothness.Override(0.45f);
        var col = profile.Add<ColorAdjustments>(true);
        col.postExposure.Override(0.35f);
        col.contrast.Override(12f);
        col.saturation.Override(8f);
        foreach (var c in profile.components) AssetDatabase.AddObjectToAsset(c, profile);
        EditorUtility.SetDirty(profile);
        AssetDatabase.SaveAssets();
        return profile;
    }

    // Çalışma anında renk verilecek temel materyalleri oluşturur. Materyaller Resources klasöründe olduğu için
    // shader'ları ve kullanılan seçenekleri (emisyon, saydamlık) derlemeye otomatik dahil olur.
    static void SetupMaterials()
    {
        Directory.CreateDirectory("Assets/Resources");
        foreach (var old in new[] { "Lit", "Unlit", "Emissive", "Glass", "Particle" })
            AssetDatabase.DeleteAsset($"Assets/Resources/{old}.mat");

        var lit = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        lit.SetFloat("_Smoothness", 0.45f);
        AssetDatabase.CreateAsset(lit, "Assets/Resources/Lit.mat");

        var emissive = new Material(lit);  // kendi ışığını yayan yüzeyler (lambalar, ekranlar)
        emissive.EnableKeyword("_EMISSION");
        emissive.globalIlluminationFlags = MaterialGlobalIlluminationFlags.None;
        emissive.SetColor("_EmissionColor", Color.white);
        AssetDatabase.CreateAsset(emissive, "Assets/Resources/Emissive.mat");

        var unlit = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        AssetDatabase.CreateAsset(unlit, "Assets/Resources/Unlit.mat");

        var glass = new Material(lit);  // yarı saydam cam (tank gözetleme borusu)
        MakeTransparent(glass, additive: false);
        glass.SetFloat("_Smoothness", 0.95f);
        AssetDatabase.CreateAsset(glass, "Assets/Resources/Glass.mat");

        var particle = new Material(Shader.Find("Universal Render Pipeline/Particles/Unlit"));  // buhar, duman
        MakeTransparent(particle, additive: false);
        AssetDatabase.CreateAsset(particle, "Assets/Resources/Particle.mat");
    }

    static void MakeTransparent(Material m, bool additive)
    {
        m.SetFloat("_Surface", 1f);
        m.SetFloat("_Blend", additive ? 2f : 0f);
        m.SetFloat("_SrcBlend", (float)(additive ? BlendMode.One : BlendMode.SrcAlpha));
        m.SetFloat("_DstBlend", (float)(additive ? BlendMode.One : BlendMode.OneMinusSrcAlpha));
        m.SetFloat("_SrcBlendAlpha", 1f);
        m.SetFloat("_DstBlendAlpha", (float)BlendMode.OneMinusSrcAlpha);
        m.SetFloat("_ZWrite", 0f);
        m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        m.SetOverrideTag("RenderType", "Transparent");
        m.renderQueue = (int)RenderQueue.Transparent;
        m.SetShaderPassEnabled("DepthOnly", false);
        m.SetShaderPassEnabled("ShadowCaster", false);
    }

    // URP'nin dışarıya açmadığı ayarları SerializedObject üzerinden atar
    static void Set(SerializedObject so, string name, object value)
    {
        var p = so.FindProperty(name);
        if (p == null) { Debug.LogWarning($"Ayar bulunamadı: {name}"); return; }
        switch (value)
        {
            case int i: p.intValue = i; break;
            case float f: p.floatValue = f; break;
            case bool b: p.boolValue = b; break;
        }
    }

    public static void BuildMac() => Build(BuildTarget.StandaloneOSX, "../Build/KompresorViz.app");

    public static void BuildWeb() => Build(BuildTarget.WebGL, "../Build/Web");

    static void Build(BuildTarget target, string output)
    {
        Setup();
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = output,
            target = target,
            options = BuildOptions.None,
        });
        Debug.Log($"BUILD RESULT: {report.summary.result}, {report.summary.totalErrors} hata");
        if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
            EditorApplication.Exit(1);
    }
}
