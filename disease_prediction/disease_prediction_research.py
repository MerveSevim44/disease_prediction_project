from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import numpy as np
import pandas as pd


df = pd.read_csv('C:\\Users\\merve\\Desktop\\disease_prediction\\data\\Training.csv') 
df.head()


def check_missing_values(df):
    """
    Veri setindeki eksik değerleri kontrol eder ve görselleştirir.
    """

    print("=" * 70)
    print("🔍 EKSİK DEĞER ANALİZİ")
    print("=" * 70)

    # 1. Genel Özet
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    missing_percentage = (total_missing / total_cells) * 100

    print(f"\n📊 GENEL ÖZET:")
    print(f"   Toplam satır: {df.shape[0]}")
    print(f"   Toplam kolon: {df.shape[1]}")
    print(f"   Toplam hücre: {total_cells:,}")
    print(f"   Eksik değer: {total_missing:,}")
    print(f"   Eksik oran: %{missing_percentage:.2f}")

    # 2. Kolonlara göre eksik değer
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100)
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )

    if len(missing_data) > 0:
        print(f"\n⚠️  EKSİK DEĞER OLAN KOLONLAR ({len(missing_data)} adet):")
        print("-" * 70)
        for idx, row in missing_data.iterrows():
            print(f"   {row['Column']:40s} | "
                  f"Eksik: {int(row['Missing_Count']):5d} | "
                  f"%{row['Missing_Percentage']:5.2f}")
    else:
        print(f"\n✅ HİÇBİR KOLONĐA EKSİK DEĞER YOK!")
        print("   Veri seti temiz ve kullanıma hazır 🎉")

    # 5. Veri tipi kontrolü
    print(f"\n📌 VERİ TİPLERİ:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {str(dtype):15s} : {count:3d} kolon")

    return missing_data if len(missing_data) > 0 else None
missing_report = check_missing_values(df)


print("number of unique disease: ", df.nunique())
print("disease " , df["prognosis"])

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
check_df(df)

def grab_col_names(dataframe, cat_th= 42, car_th=50):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

# Hedef değişkeni çıkar
feature_cols = [col for col in df.columns if col != 'prognosis']

# Her semptom için analiz
for symptom in feature_cols:
    print(f"\n{'=' * 70}")
    print(f"SEMPTOM: {symptom.upper()}")
    print(f"{'=' * 70}")

    # Crosstab oluştur
    crosstab = pd.crosstab(df[symptom], df['prognosis'])

    # Eğer semptom binary (0-1) ise
    if df[symptom].nunique() == 2 and set(df[symptom].unique()) == {0, 1}:

        # Semptom OLAN hastalar (1)
        if 1 in crosstab.index:
            print(f"\n✓ {symptom.replace('_', ' ').title()} OLAN hastalık sayıları:")
            symptom_present = crosstab.loc[1][crosstab.loc[1] > 0].sort_values(ascending=False)
            if len(symptom_present) > 0:
                print(symptom_present.head(10))  # İlk 10 hastalık
                print(f"Toplam {len(symptom_present)} farklı hastalıkta görülüyor")
            else:
                print("Hiçbir hastalıkta görülmüyor")

        # Semptom OLMAYAN hastalar (0)
        if 0 in crosstab.index:
            print(f"\n✗ {symptom.replace('_', ' ').title()} OLMAYAN hastalık sayıları:")
            symptom_absent = crosstab.loc[0][crosstab.loc[0] > 0].sort_values(ascending=False)
            if len(symptom_absent) > 0:
                print(symptom_absent.head(10))  # İlk 10 hastalık
                print(f"Toplam {len(symptom_absent)} farklı hastalıkta bu semptom YOK")

    else:
        # Binary olmayan değişkenler için (eğer varsa)
        print(f"\nBu semptom {df[symptom].nunique()} farklı değer alıyor")
        print(crosstab)

    print("\n" + "-" * 70)

# Her semptom için kaç hastalıkta göründüğünü hesapla
symptom_specificity = {}

for symptom in feature_cols:
    if df[symptom].nunique() == 2:
        crosstab = pd.crosstab(df[symptom], df['prognosis'])
        if 1 in crosstab.index:
            num_diseases = (crosstab.loc[1] > 0).sum()
            symptom_specificity[symptom] = num_diseases

# En spesifik semptomlar (az sayıda hastalıkta görülenler)
print("\n" + "=" * 70)
print("EN SPESİFİK SEMPTOMLAR (Az sayıda hastalıkta görülenler)")
print("=" * 70)

sorted_specific = sorted(symptom_specificity.items(), key=lambda x: x[1])

for symptom, num_diseases in sorted_specific[:15]:  # İlk 15
    print(f"{symptom.replace('_', ' ').title()}: {num_diseases} hastalıkta")

    # Bu semptomu olan hastalıkları göster
    crosstab = pd.crosstab(df[symptom], df['prognosis'])
    diseases = crosstab.loc[1][crosstab.loc[1] > 0].index.tolist()
    print(f"  → {', '.join(diseases)}\n")


def analyze_correlation_matrix(dataframe, target_col='prognosis', figsize=(20, 18)):
    """
    Veri setindeki semptomlar arası korelasyon analizini yapar.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Analiz edilecek veri seti
    target_col : str
        Hedef değişken adı (analizden çıkarılacak)
    figsize : tuple
        Grafik boyutu

    Returns:
    --------
    dict : Korelasyon analizi sonuçları
    """

    # Sadece sayısal kolonları al (hedef değişken hariç)
    numerical_features = dataframe.drop(target_col, axis=1)

    # Korelasyon matrisini hesapla
    correlation_matrix = numerical_features.corr()

    print("=" * 70)
    print("KORELASYON MATRİSİ ANALİZİ BAŞLATILDI")
    print("=" * 70)
    print(f"Toplam özellik sayısı: {len(numerical_features.columns)}")
    print(f"Korelasyon matrisi boyutu: {correlation_matrix.shape}")

    # Sonuçları sakla
    results = {
        'correlation_matrix': correlation_matrix,
        'high_positive': None,
        'high_negative': None,
        'mean_correlations': None,
        'multicollinearity': None
    }

    # Korelasyon çiftlerini oluştur
    corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_pairs.append({
                'semptom_1': correlation_matrix.columns[i],
                'semptom_2': correlation_matrix.columns[j],
                'korelasyon': correlation_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)

    # Yüksek pozitif korelasyonlar
    high_positive = corr_df[corr_df['korelasyon'] > 0.5].sort_values('korelasyon', ascending=False)
    results['high_positive'] = high_positive

    # Yüksek negatif korelasyonlar
    high_negative = corr_df[corr_df['korelasyon'] < -0.3].sort_values('korelasyon')
    results['high_negative'] = high_negative

    # Ortalama korelasyonlar
    mean_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
    results['mean_correlations'] = mean_correlations

    # Multicollinearity kontrolü
    multicollinearity = corr_df[corr_df['korelasyon'] > 0.9].sort_values('korelasyon', ascending=False)
    results['multicollinearity'] = multicollinearity

    return results


def print_correlation_summary(results, top_n=20):
    """
    Korelasyon analizi sonuçlarını özetler ve yazdırır.

    Parameters:
    -----------
    results : dict
        analyze_correlation_matrix fonksiyonundan dönen sonuçlar
    top_n : int
        Gösterilecek üst sıra sayısı
    """

    print("\n" + "=" * 70)
    print("🔴 GÜÇLÜ POZİTİF KORELASYONLAR (>0.5)")
    print("=" * 70)

    high_positive = results['high_positive']
    if len(high_positive) > 0:
        print(f"\nToplam {len(high_positive)} adet güçlü pozitif korelasyon bulundu.\n")
        for idx, row in high_positive.head(top_n).iterrows():
            print(f"{row['semptom_1']:35} <--> {row['semptom_2']:35} : {row['korelasyon']:.3f}")
    else:
        print("Güçlü pozitif korelasyon bulunamadı.")

    print("\n" + "=" * 70)
    print("🔵 GÜÇLÜ NEGATİF KORELASYONLAR (<-0.3)")
    print("=" * 70)

    high_negative = results['high_negative']
    if len(high_negative) > 0:
        print(f"\nToplam {len(high_negative)} adet güçlü negatif korelasyon bulundu.\n")
        for idx, row in high_negative.head(top_n).iterrows():
            print(f"{row['semptom_1']:35} <--> {row['semptom_2']:35} : {row['korelasyon']:.3f}")
    else:
        print("Güçlü negatif korelasyon bulunamadı.")

    print("\n" + "=" * 70)
    print("📊 EN ÇOK DİĞER SEMPTOMLARLA İLİŞKİLİ OLANLAR")
    print("=" * 70)

    mean_correlations = results['mean_correlations']
    print(f"\nEn yüksek ortalama korelasyona sahip {top_n} semptom:\n")
    for symptom, corr in mean_correlations.head(top_n).items():
        print(f"{symptom:45} : {corr:.3f}")

    print("\n" + "=" * 70)
    print("⚠️  MULTICOLLINEARİTY KONTROLÜ (>0.9)")
    print("=" * 70)

    multicollinearity = results['multicollinearity']
    if len(multicollinearity) > 0:
        print(f"\n⚠️  UYARI: {len(multicollinearity)} adet çok yüksek korelasyon bulundu!")
        print("Bu semptom çiftleri neredeyse aynı bilgiyi taşıyor.")
        print("Model için birini çıkarmayı düşünebilirsiniz:\n")
        for idx, row in multicollinearity.iterrows():
            print(f"{row['semptom_1']:35} <--> {row['semptom_2']:35} : {row['korelasyon']:.3f}")
    else:
        print("\n✅ Çok yüksek korelasyon bulunamadı (multicollinearity riski düşük)")


def plot_full_correlation_heatmap(correlation_matrix, figsize=(20, 18)):
    """
    Tüm özelliklerin korelasyon matrisini görselleştirir.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Korelasyon matrisi
    figsize : tuple
        Grafik boyutu
    """

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix,
                annot=False,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Korelasyon Katsayısı'})
    plt.title('Semptomlar Arası Korelasyon Matrisi (Tüm Özellikler)',
              fontsize=16,
              pad=20,
              fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_top_correlations_heatmap(correlation_matrix, top_n=30, figsize=(16, 14)):
    """
    En yüksek ortalama korelasyona sahip N semptomu görselleştirir.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Korelasyon matrisi
    top_n : int
        Gösterilecek semptom sayısı
    figsize : tuple
        Grafik boyutu
    """

    mean_correlations = correlation_matrix.abs().mean().sort_values(ascending=False)
    top_symptoms = mean_correlations.head(top_n).index
    corr_top = correlation_matrix.loc[top_symptoms, top_symptoms]

    plt.figure(figsize=figsize)
    sns.heatmap(corr_top,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Korelasyon Katsayısı'})
    plt.title(f'En İlişkili {top_n} Semptom Arası Korelasyon Matrisi',
              fontsize=14,
              pad=20,
              fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

"""""
def plot_correlation_dendrogram(correlation_matrix, figsize=(20, 10)):
 
    Semptomların hiyerarşik kümeleme dendrogramını oluşturur.

    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Korelasyon matrisi
    figsize : tuple
        Grafik boyutu
   
    from scipy.cluster.hierarchy import dendrogram, linkage

    try:
        # Transpose edilmiş veriyi kullan (her semptom bir gözlem gibi)
        # Böylece özellikler arası mesafe hesaplanır
        data_for_clustering = correlation_matrix.T

        # Linkage hesapla (correlation metric ile)
        linkage_matrix = linkage(data_for_clustering, method='average', metric='correlation')

        plt.figure(figsize=figsize)
        dendrogram_plot = dendrogram(
            linkage_matrix,
            labels=correlation_matrix.columns.tolist(),
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0.7
        )
        plt.title('Semptom Kümeleme Dendrogramı (Hiyerarşik Kümeleme)',
                  fontsize=14,
                  pad=20,
                  fontweight='bold')
        plt.xlabel('Semptomlar', fontsize=12)
        plt.ylabel('Mesafe (Korelasyon Uzaklığı)', fontsize=12)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Kesim Noktası = 0.5', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Dendrogram oluşturulamadı: {e}")
        print("Alternatif görselleştirme yöntemi kullanılıyor...")

        # Alternatif: Clustermap kullan
        plt.figure(figsize=figsize)
        sns.clustermap(correlation_matrix,
                       cmap='coolwarm',
                       center=0,
                       figsize=(15, 15),
                       dendrogram_ratio=0.1,
                       cbar_pos=(0.02, 0.8, 0.03, 0.18))
        plt.suptitle('Semptom Kümeleme (Clustermap)',
                     fontsize=14,
                     y=0.98,
                     fontweight='bold')
        plt.show()
"""""

def plot_correlation_distribution(results, figsize=(12, 5)):
    """
    Korelasyon değerlerinin dağılımını görselleştirir.

    Parameters:
    -----------
    results : dict
        analyze_correlation_matrix fonksiyonundan dönen sonuçlar
    figsize : tuple
        Grafik boyutu
    """

    correlation_matrix = results['correlation_matrix']

    # Üst üçgen korelasyon değerlerini al (kendisi ile korelasyonları çıkar)
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    correlations = upper_triangle.stack().values

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(correlations, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Korelasyon = 0')
    axes[0].axvline(x=0.5, color='orange', linestyle='--', linewidth=1.5, label='Korelasyon = 0.5')
    axes[0].axvline(x=-0.5, color='orange', linestyle='--', linewidth=1.5, label='Korelasyon = -0.5')
    axes[0].set_xlabel('Korelasyon Değeri', fontsize=11)
    axes[0].set_ylabel('Frekans', fontsize=11)
    axes[0].set_title('Korelasyon Değerlerinin Dağılımı', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(correlations, vert=True)
    axes[1].set_ylabel('Korelasyon Değeri', fontsize=11)
    axes[1].set_title('Korelasyon Değerleri Box Plot', fontsize=12, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=-0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # İstatistikler
    print("\n" + "=" * 70)
    print("📈 KORELASYON DEĞERLERİ İSTATİSTİKLERİ")
    print("=" * 70)
    print(f"Ortalama korelasyon: {correlations.mean():.4f}")
    print(f"Medyan korelasyon: {np.median(correlations):.4f}")
    print(f"Standart sapma: {correlations.std():.4f}")
    print(f"Minimum korelasyon: {correlations.min():.4f}")
    print(f"Maksimum korelasyon: {correlations.max():.4f}")
    print(f"\nPozitif korelasyon sayısı: {(correlations > 0).sum()}")
    print(f"Negatif korelasyon sayısı: {(correlations < 0).sum()}")
    print(f"Güçlü pozitif (>0.5) sayısı: {(correlations > 0.5).sum()}")
    print(f"Güçlü negatif (<-0.5) sayısı: {(correlations < -0.5).sum()}")


def full_correlation_analysis(dataframe, target_col='prognosis', top_n=20, show_plots=True):
    """
    Tüm korelasyon analizini tek bir fonksiyonda çalıştırır.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        Analiz edilecek veri seti
    target_col : str
        Hedef değişken adı
    top_n : int
        Gösterilecek üst sıra sayısı
    show_plots : bool
        Grafiklerin gösterilip gösterilmeyeceği

    Returns:
    --------
    dict : Analiz sonuçları
    """

    print("\n" + "🔍 " * 35)
    print("KAPSAMLI KORELASYON ANALİZİ")
    print("🔍 " * 35 + "\n")

    # 1. Korelasyon matrisini hesapla
    results = analyze_correlation_matrix(dataframe, target_col)

    # 2. Özet bilgileri yazdır
    print_correlation_summary(results, top_n)

    if show_plots:
        # 3. Tüm korelasyon matrisi
        plot_full_correlation_heatmap(results['correlation_matrix'])

        # 4. En ilişkili semptomlar
        plot_top_correlations_heatmap(results['correlation_matrix'], top_n=30)

        # 5. Dendrogram
        #plot_correlation_dendrogram(results['correlation_matrix'])

        # 6. Dağılım grafikleri
        plot_correlation_distribution(results)

    print("\n" + "✅ " * 35)
    print("ANALİZ TAMAMLANDI")
    print("✅ " * 35 + "\n")

    return results


# KULLANIM ÖRNEĞİ:

# Tam analiz
# 1. TEK SATIRDA TÜM ANALİZ
#results = full_correlation_analysis(df, target_col='prognosis', top_n=20, show_plots=True)

# 2. SADECE HESAPLAMALAR (GRAFİKSİZ)
#results = full_correlation_analysis(df, target_col='prognosis', show_plots=False)

# 3. ADIM ADIM ÖZELLEŞMTIRILMŞ ANALİZ
#results = analyze_correlation_matrix(df, target_col='prognosis')
#print_correlation_summary(results, top_n=15)
#plot_top_correlations_heatmap(results['correlation_matrix'], top_n=25)
#plot_correlation_distribution(results)

# 4. SADECE BİR GÖRSEL
#results = analyze_correlation_matrix(df, target_col='prognosis')
#plot_full_correlation_heatmap(results['correlation_matrix'], figsize=(22, 20))


def perform_feature_engineering(df):
    """
    Yüksek korelasyonlu semptomlardan yeni özellikler oluşturur ve
    multicollinearity'i temizler.

    Parameters:
    -----------
    df : pd.DataFrame
        Orijinal veri seti (prognosis hedef değişkeni içermeli)

    Returns:
    --------
    df_engineered : pd.DataFrame
        Feature engineering uygulanmış veri seti
    """

    df_new = df.copy()

    print("\n" + "🔧 " * 35)
    print("FEATURE ENGINEERING BAŞLATILDI")
    print("🔧 " * 35 + "\n")

    original_shape = df_new.shape

    # 1. Respiratory System Score
    print("1️  Respiratory system features oluşturuluyor...")
    df_new['NEW_respiratory_symptom_score'] = df_new[['congestion', 'loss_of_smell',
                                                      'runny_nose', 'sinus_pressure',
                                                      'throat_irritation', 'redness_of_eyes']].sum(axis=1)
    df_new = df_new.drop(['congestion', 'loss_of_smell', 'runny_nose',
                          'sinus_pressure', 'throat_irritation', 'redness_of_eyes'], axis=1)

    # 2. Metabolic/Diabetes Score
    print("2️  Metabolic/Diabetes features oluşturuluyor...")
    df_new['NEW_metabolic_symptom_score'] = df_new[['irregular_sugar_level',
                                                    'polyuria',
                                                    'increased_appetite']].sum(axis=1)
    df_new = df_new.drop(['irregular_sugar_level', 'polyuria', 'increased_appetite'], axis=1)

    # 3. Hepatic Complication Score
    print("3️  Hepatic complication features oluşturuluyor...")
    df_new['NEW_hepatic_complication_score'] = df_new[['acute_liver_failure',
                                                       'stomach_bleeding',
                                                       'coma']].sum(axis=1)
    df_new = df_new.drop(['acute_liver_failure', 'stomach_bleeding', 'coma'], axis=1)

    # 4. Hepatitis Related Score
    print("4️  Hepatitis related features oluşturuluyor...")
    df_new['NEW_hepatitis_related_score'] = df_new[['yellow_urine',
                                                    'receiving_unsterile_injections',
                                                    'receiving_blood_transfusion']].sum(axis=1)
    df_new = df_new.drop(['yellow_urine', 'receiving_unsterile_injections',
                          'receiving_blood_transfusion'], axis=1)

    # 5. Thyroid Related Score
    print("5️ Thyroid related features oluşturuluyor...")
    df_new['NEW_thyroid_related_score'] = df_new[['brittle_nails', 'swollen_extremeties',
                                                  'cold_hands_and_feets', 'enlarged_thyroid',
                                                  'weight_gain', 'puffy_face_and_eyes']].sum(axis=1)
    df_new = df_new.drop(['brittle_nails', 'swollen_extremeties', 'cold_hands_and_feets',
                          'enlarged_thyroid', 'weight_gain', 'puffy_face_and_eyes'], axis=1)

    # 6. Digestive/Anal Symptoms Score
    print("6️  Digestive/Anal symptom features oluşturuluyor...")
    df_new['NEW_digestive_anal_symptom_score'] = df_new[['pain_in_anal_region',
                                                         'irritation_in_anus',
                                                         'bloody_stool',
                                                         'pain_during_bowel_movements']].sum(axis=1)
    df_new = df_new.drop(['pain_in_anal_region', 'irritation_in_anus', 'bloody_stool',
                          'pain_during_bowel_movements'], axis=1)

    # 7. Neurocardiac Symptoms Score
    print("7️  Neurocardiac symptom features oluşturuluyor...")
    df_new['NEW_neurocardiac_symptom_score'] = df_new[['slurred_speech',
                                                       'palpitations',
                                                       'anxiety',
                                                       'drying_and_tingling_lips']].sum(axis=1)
    df_new = df_new.drop(['slurred_speech', 'palpitations', 'anxiety',
                          'drying_and_tingling_lips'], axis=1)

    # 8. Skin Infection Score
    print("8️  Skin infection features oluşturuluyor...")
    df_new['NEW_skin_infection_score'] = df_new[['blister',
                                                 'yellow_crust_ooze',
                                                 'red_sore_around_nose']].sum(axis=1)
    df_new = df_new.drop(['blister', 'yellow_crust_ooze', 'red_sore_around_nose'], axis=1)

    # 9. Venous Disorder Score
    print("9️  Venous disorder features oluşturuluyor...")
    df_new['NEW_venous_disorder_score'] = df_new[['bruising',
                                                  'swollen_legs',
                                                  'prominent_veins_on_calf',
                                                  'cramps',
                                                  'swollen_blood_vessels']].sum(axis=1)
    df_new = df_new.drop(['bruising', 'swollen_legs', 'prominent_veins_on_calf',
                          'cramps', 'swollen_blood_vessels'], axis=1)

    # 10. Psoriasis Related Score
    print(" Psoriasis related features oluşturuluyor...")
    df_new['NEW_psoriasis_related_score'] = df_new[['skin_peeling',
                                                    'inflammatory_nails',
                                                    'silver_like_dusting',
                                                    'small_dents_in_nails']].sum(axis=1)
    df_new = df_new.drop(['skin_peeling', 'inflammatory_nails',
                          'silver_like_dusting', 'small_dents_in_nails'], axis=1)

    # 11. Hepatic Failure Score
    print("11 Hepatic failure features oluşturuluyor...")
    df_new['NEW_hepatic_failure_score'] = df_new[['swelling_of_stomach',
                                                  'history_of_alcohol_consumption',
                                                  'distention_of_abdomen',
                                                  'fluid_overload.1']].sum(axis=1)
    df_new = df_new.drop(['swelling_of_stomach', 'history_of_alcohol_consumption',
                          'distention_of_abdomen', 'fluid_overload.1'], axis=1)

    # 12. Hormonal Imbalance Score
    print("12 Hormonal imbalance features oluşturuluyor...")
    df_new['NEW_hormonal_imbalance_score'] = df_new[['mood_swings',
                                                     'abnormal_menstruation']].sum(axis=1)
    df_new = df_new.drop(['mood_swings', 'abnormal_menstruation'], axis=1)

    # 13. Gastrointestinal Distress Score
    print("13 Gastrointestinal distress features oluşturuluyor...")
    df_new['NEW_gastrointestinal_distress_score'] = df_new[['passage_of_gases',
                                                            'internal_itching']].sum(axis=1)
    df_new = df_new.drop(['passage_of_gases', 'internal_itching'], axis=1)

    # 14. Typhoid Symptoms Score
    print("14 Typhoid symptom features oluşturuluyor...")
    df_new['NEW_typhoid_symptom_score'] = df_new[['toxic_look_(typhos)',
                                                  'belly_pain']].sum(axis=1)
    df_new = df_new.drop(['toxic_look_(typhos)', 'belly_pain'], axis=1)

    # 15. Joint Pain Score
    print("15 Joint pain features oluşturuluyor...")
    df_new['NEW_joint_pain_score'] = df_new[['knee_pain',
                                             'hip_joint_pain']].sum(axis=1)
    df_new = df_new.drop(['knee_pain', 'hip_joint_pain'], axis=1)

    # 16. Urinary Discomfort Score
    print("16 Urinary discomfort features oluşturuluyor...")
    df_new['NEW_urinary_discomfort_score'] = df_new[['bladder_discomfort',
                                                     'continuous_feel_of_urine','foul_smell_of urine']].sum(axis=1)
    df_new = df_new.drop(['bladder_discomfort', 'continuous_feel_of_urine','foul_smell_of urine'], axis=1)

    # 17. Mobility Issue Score
    print("17 Mobility issue features oluşturuluyor...")
    df_new['NEW_mobility_issue_score'] = df_new[['swelling_joints',
                                                 'painful_walking']].sum(axis=1)
    df_new = df_new.drop(['swelling_joints', 'painful_walking'], axis=1)

    # 18. Neurological Impairment Score
    print("18 Neurological impairment features oluşturuluyor...")
    df_new['NEW_neurological_impairment_score'] = df_new[['weakness_of_one_body_side',
                                                          'altered_sensorium']].sum(axis=1)
    df_new = df_new.drop(['weakness_of_one_body_side', 'altered_sensorium'], axis=1)

    # 19. Balance Disorder Score
    print("19 Balance disorder features oluşturuluyor...")
    df_new['NEW_balance_disorder_score'] = df_new[['spinning_movements',
                                                   'unsteadiness']].sum(axis=1)
    df_new = df_new.drop(['spinning_movements', 'unsteadiness'], axis=1)

    # RAPOR
    print("\n" + "=" * 70)
    print("📊 FEATURE ENGINEERING RAPORU")
    print("=" * 70)

    new_features = sorted([c for c in df_new.columns if 'NEW_' in c])

    print(f"\n🔢 Özellik Sayısı:")
    print(f"   Orijinal: {original_shape[1]} özellik")
    print(f"   Yeni:     {df_new.shape[1]} özellik")
    print(f"   Çıkarılan: {original_shape[1] - df_new.shape[1]} özellik")
    print(f"   Oluşturulan: {len(new_features)} yeni özellik")
    print(f"   İndirgeme oranı: %{((original_shape[1] - df_new.shape[1]) / original_shape[1] * 100):.1f}")

    print(f"\n✅ Yeni Oluşturulan Özellikler ({len(new_features)} adet):")
    for i, feat in enumerate(new_features, 1):
        count = (df_new[feat] > 0).sum()
        percentage = (count / len(df_new)) * 100
        print(f"   {i:2d}. {feat:50s} | Pozitif: {count:4d} ({percentage:5.1f}%)")

    # Korelasyon Kontrolü
    print(f"\n🔍 Multicollinearity Kontrolü:")
    X_new = df_new.drop('prognosis', axis=1)
    corr_new = X_new.corr().abs()

    # Üst üçgen matrisini al (diagonal hariç)
    upper_triangle = corr_new.where(
        np.triu(np.ones(corr_new.shape), k=1).astype(bool)
    )

    # Yüksek korelasyonları bul
    high_corr_pairs = []
    for column in upper_triangle.columns:
        high_corr_features = upper_triangle[column][upper_triangle[column] > 0.9]
        for idx, value in high_corr_features.items():
            high_corr_pairs.append((column, idx, value))

    print(f"   Yüksek korelasyon (>0.9) çift sayısı: {len(high_corr_pairs)}")

    if len(high_corr_pairs) == 0:
        print("   ✅ Multicollinearity temizlendi!")
    else:
        print(f"   ⚠️  Hala {len(high_corr_pairs)} yüksek korelasyon var:")
        for feat1, feat2, corr_val in high_corr_pairs[:5]:
            print(f"      • {feat1} <-> {feat2}: {corr_val:.3f}")
        if len(high_corr_pairs) > 5:
            print(f"      ... ve {len(high_corr_pairs) - 5} tane daha")

    print("\n" + "✅ " * 35)
    print("FEATURE ENGINEERING TAMAMLANDI")
    print("✅ " * 35 + "\n")

    return df_new


#yeni geliştirlmiş dataframe
df_engineered = perform_feature_engineering(df)

# Veri setini kaydet (opsiyonel)
# df_engineered.to_csv('disease_prediction_engineered.csv', index=False),

def evaluate_models(X, y):
    """
    Verilen classification modellerini kullanarak çapraz doğrulama yapar ve
    her model için Accuracy, Precision, Recall, F1-Score hesaplar.

    Parametreler:
    -----------
    X : pandas.DataFrame veya numpy.ndarray
        Özellikler (input)
    y : pandas.Series veya numpy.ndarray
        Hedef değişken (output) - Label encoded olmalı

    Dönüş:
    ------
    pd.DataFrame : Her modelin performans metrikleri
    """


    # Classification modelleri
    models = [
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=42)),
        ('RF', RandomForestClassifier(random_state=42))
        #('GBM', GradientBoostingClassifier(random_state=42)),
        #('XGBoost', XGBClassifier(random_state=42, eval_metric='mlogloss')),
        #('LightGBM', LGBMClassifier(random_state=42, verbose=-1))
    ]

    results = []

    print("=" * 70)
    print("MODEL PERFORMANS DEĞERLENDİRMESİ")
    print("=" * 70)

    for name, classifier in models:
        # Accuracy
        accuracy = cross_val_score(classifier, X, y, cv=10,
                                   scoring='accuracy').mean()

        # Precision (weighted - multi-class için)
        precision = cross_val_score(classifier, X, y, cv=10,
                                    scoring='precision_weighted').mean()

        # Recall (weighted - multi-class için)
        recall = cross_val_score(classifier, X, y, cv=10,
                                 scoring='recall_weighted').mean()

        # F1-Score (weighted - multi-class için)
        f1 = cross_val_score(classifier, X, y, cv=10,
                             scoring='f1_weighted').mean()

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        print(f"{name:10s} | Accuracy: {accuracy:.4f} | "
              f"Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | "
              f"F1: {f1:.4f}")

    print("=" * 70)

    # DataFrame'e çevir
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)

    # En iyi modeli göster
    best_model = results_df.iloc[0]
    print(f"\n🏆 EN İYİ MODEL: {best_model['Model']}")
    print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall:    {best_model['Recall']:.4f}")
    print(f"   F1-Score:  {best_model['F1-Score']:.4f}")

    return results_df

# Hedef değişkeni encode et
le = LabelEncoder()
y_encoded = le.fit_transform(df_engineered['prognosis'])

# X ve y hazırla
X = df_engineered.drop('prognosis', axis=1)
y = y_encoded

# Modelleri değerlendir
results = evaluate_models(X, y)

# Sonuçları görselleştir
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(results))
width = 0.2

ax.bar(x_pos - 1.5 * width, results['Accuracy'], width, label='Accuracy', alpha=0.8)
ax.bar(x_pos - 0.5 * width, results['Precision'], width, label='Precision', alpha=0.8)
ax.bar(x_pos + 0.5 * width, results['Recall'], width, label='Recall', alpha=0.8)
ax.bar(x_pos + 1.5 * width, results['F1-Score'], width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Karşılaştırması - Classification Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results['Model'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def check_overfitting_with_cv(X, y, models_list):
    """
    Cross-validation ile birlikte overfitting kontrolü yapar.

    Parameters:
    -----------
    X : pandas.DataFrame veya numpy.ndarray
        Özellikler (features)
    y : pandas.Series veya numpy.ndarray
        Hedef değişken (encoded)
    models_list : list of tuples
        [(name, model), ...] formatında model listesi

    Returns:
    --------
    pd.DataFrame : Overfitting metrikleri ile birlikte sonuçlar
    """

    results = []

    print("\n" + "=" * 80)
    print("OVERFITTING ANALİZİ İLE MODEL PERFORMANS DEĞERLENDİRMESİ")
    print("=" * 80)

    # Her model için analiz
    for name, model in models_list:
        print(f"\n{'─' * 80}")
        print(f"📊 Model: {name}")
        print(f"{'─' * 80}")

        # 1. Cross-validation skorları (CV ile train-test farkı)
        cv_results = cross_validate(
            model, X, y, cv=10,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
            return_train_score=True
        )

        # Train ve test skorlarının ortalaması
        train_acc = cv_results['train_accuracy'].mean()
        test_acc = cv_results['test_accuracy'].mean()
        train_f1 = cv_results['train_f1_weighted'].mean()
        test_f1 = cv_results['test_f1_weighted'].mean()

        # Standart sapma (model stabilitesi)
        test_acc_std = cv_results['test_accuracy'].std()

        # Overfitting skoru (train-test farkı)
        acc_diff = train_acc - test_acc
        f1_diff = train_f1 - test_f1

        # 2. Ayrı bir train-test split ile de kontrol
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_score_split = accuracy_score(y_train, y_train_pred)
        test_score_split = accuracy_score(y_test, y_test_pred)
        split_diff = train_score_split - test_score_split

        # 3. Overfitting değerlendirmesi
        avg_diff = (acc_diff + split_diff) / 2

        if avg_diff < 0.02:
            status = "✅ İYİ"
            overfitting_level = "Yok"
            color = "🟢"
        elif avg_diff < 0.05:
            status = "⚠️  DİKKAT"
            overfitting_level = "Hafif"
            color = "🟡"
        elif avg_diff < 0.10:
            status = "🔴 SORUN"
            overfitting_level = "Orta"
            color = "🟠"
        else:
            status = "🚨 CİDDİ"
            overfitting_level = "Yüksek"
            color = "🔴"

        # Sonuçları yazdır
        print(f"\n{color} OVERFITTING DURUMU: {status} - {overfitting_level}")
        print(f"\n📈 Cross-Validation Skorları:")
        print(f"   Train Accuracy:      {train_acc:.4f}")
        print(f"   Test Accuracy (CV):  {test_acc:.4f}")
        print(f"   Fark (CV):           {acc_diff:.4f}")
        print(f"   Test Std Dev:        {test_acc_std:.4f}")

        print(f"\n📊 Train-Test Split Skorları:")
        print(f"   Train Accuracy:      {train_score_split:.4f}")
        print(f"   Test Accuracy:       {test_score_split:.4f}")
        print(f"   Fark (Split):        {split_diff:.4f}")

        print(f"\n🎯 Ortalama Overfitting Skoru: {avg_diff:.4f}")

        # Stabilite analizi
        if test_acc_std > 0.05:
            print(f"⚠️  Model kararsız - Yüksek varyans ({test_acc_std:.4f})")
        else:
            print(f"✅ Model stabil - Düşük varyans ({test_acc_std:.4f})")

        # Sonuçları kaydet
        results.append({
            'Model': name,
            'CV_Test_Accuracy': test_acc,
            'CV_Train_Accuracy': train_acc,
            'CV_Diff': acc_diff,
            'Split_Train_Acc': train_score_split,
            'Split_Test_Acc': test_score_split,
            'Split_Diff': split_diff,
            'Avg_Overfitting': avg_diff,
            'Test_Std': test_acc_std,
            'Overfitting_Level': overfitting_level,
            'Precision': cv_results['test_precision_weighted'].mean(),
            'Recall': cv_results['test_recall_weighted'].mean(),
            'F1_Score': test_f1
        })

    # DataFrame oluştur
    results_df = pd.DataFrame(results)

    # Görselleştirme
    plot_overfitting_analysis(results_df)

    # Özet
    print("\n" + "=" * 80)
    print("📋 OVERFITTING ÖZET TABLOSU")
    print("=" * 80)

    summary = results_df[['Model', 'CV_Test_Accuracy', 'Avg_Overfitting',
                          'Overfitting_Level', 'Test_Std']].copy()
    summary.columns = ['Model', 'Test Acc', 'Overfitting', 'Seviye', 'Std Dev']
    print(summary.to_string(index=False))

    # En iyi model
    best_idx = results_df['CV_Test_Accuracy'].idxmax()
    best_model = results_df.loc[best_idx]

    print("\n" + "=" * 80)
    print("🏆 EN İYİ MODEL")
    print("=" * 80)
    print(f"Model:              {best_model['Model']}")
    print(f"Test Accuracy:      {best_model['CV_Test_Accuracy']:.4f}")
    print(f"Overfitting Skoru:  {best_model['Avg_Overfitting']:.4f} ({best_model['Overfitting_Level']})")
    print(f"Precision:          {best_model['Precision']:.4f}")
    print(f"Recall:             {best_model['Recall']:.4f}")
    print(f"F1-Score:           {best_model['F1_Score']:.4f}")
    print("=" * 80 + "\n")

    # Öneriler
    print_recommendations(results_df)

    return results_df


def plot_overfitting_analysis(results_df):
    """Overfitting analizi için grafikler çizer."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Overfitting Analizi', fontsize=16, fontweight='bold')

    models = results_df['Model']

    # 1. Train vs Test Accuracy (CV)
    x = np.arange(len(models))
    width = 0.35

    axes[0, 0].bar(x - width / 2, results_df['CV_Train_Accuracy'], width,
                   label='Train', color='#3498db', alpha=0.8)
    axes[0, 0].bar(x + width / 2, results_df['CV_Test_Accuracy'], width,
                   label='Test', color='#e74c3c', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Train vs Test Accuracy (CV)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Overfitting Skorları
    colors = ['#2ecc71' if x < 0.02 else '#f39c12' if x < 0.05
    else '#e67e22' if x < 0.10 else '#c0392b'
              for x in results_df['Avg_Overfitting']]

    axes[0, 1].bar(models, results_df['Avg_Overfitting'], color=colors, alpha=0.8)
    axes[0, 1].axhline(y=0.02, color='g', linestyle='--', label='İyi (<0.02)')
    axes[0, 1].axhline(y=0.05, color='orange', linestyle='--', label='Dikkat (<0.05)')
    axes[0, 1].axhline(y=0.10, color='red', linestyle='--', label='Sorun (<0.10)')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Overfitting Skoru')
    axes[0, 1].set_title('Overfitting Seviyeleri')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Model Stabilitesi (Std Dev)
    axes[1, 0].bar(models, results_df['Test_Std'], color='#9b59b6', alpha=0.8)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='Kararsız (>0.05)')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Standart Sapma')
    axes[1, 0].set_title('Model Stabilitesi')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Genel Performans Matrisi
    metrics_df = results_df[['Model', 'CV_Test_Accuracy', 'Precision', 'Recall', 'F1_Score']].set_index('Model')

    im = axes[1, 1].imshow(metrics_df.values.T, cmap='YlGn', aspect='auto')
    axes[1, 1].set_xticks(np.arange(len(models)))
    axes[1, 1].set_yticks(np.arange(len(metrics_df.columns)))
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].set_yticklabels(metrics_df.columns)
    axes[1, 1].set_title('Performans Metrikleri Heatmap')

    # Değerleri göster
    for i in range(len(metrics_df.columns)):
        for j in range(len(models)):
            text = axes[1, 1].text(j, i, f'{metrics_df.values.T[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()


def print_recommendations(results_df):
    """Overfitting durumuna göre öneriler verir."""

    print("💡 ÖNERİLER VE ÇÖZÜMLER")
    print("=" * 80)

    for idx, row in results_df.iterrows():
        model = row['Model']
        level = row['Overfitting_Level']

        print(f"\n🔧 {model}:")

        if level == "Yok":
            print("   ✅ Model iyi durumda, herhangi bir aksiyona gerek yok.")
        elif level == "Hafif":
            print("   ⚠️  Hafif overfitting var:")
            print("   • Regularization parametrelerini artırın (alpha, C)")
            print("   • Feature selection yapın")
        elif level == "Orta":
            print("   🔴 Orta seviye overfitting:")
            print("   • Daha fazla eğitim verisi toplayın")
            print("   • Model karmaşıklığını azaltın (max_depth, n_estimators)")
            print("   • Regularization uygulayın")
            print("   • Feature engineering gözden geçirin")
        else:  # Yüksek
            print("   🚨 Ciddi overfitting!")
            print("   • Model çok karmaşık - daha basit model deneyin")
            print("   • Veri sayısını artırın")
            print("   • Cross-validation fold sayısını artırın")
            print("   • Feature sayısını azaltın")

    print("\n" + "=" * 80)


# ==================== KULLANIM ÖRNEĞİ ====================


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Modeller listesi
models = [
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=42)),
    ('RF', RandomForestClassifier(random_state=42))
]

# Overfitting analizi
results_df = check_overfitting_with_cv(X, y, models)

# Sonuçları görüntüle
print(results_df)
