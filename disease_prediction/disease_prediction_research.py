from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

#Data Frame
df_train = pd.read_csv('C:\\Users\\merve\\Desktop\\disease_prediction\\data\\Training.csv')
df_train_new = df_train.loc[:, ~df_train.columns.str.contains('^Unnamed')] #we removed unnamed columns
df_test = pd.read_csv('C:\\Users\\merve\\Desktop\\disease_prediction\\data\\Testing.csv')
df_test_new = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]

duplicates = df_train_new.duplicated().sum()
print(f"Tespit edilen duplicate sayısı: {duplicates}")

diversity = (df_train_new.nunique() / len(df_train_new)).sort_values(ascending=False)
print(diversity)

unique_ratio = df_train_new.nunique().mean() / len(df_train_new)
avg_corr = df_train_new.corr(numeric_only=True).abs().mean().mean()
diversity_score = unique_ratio * (1 - avg_corr)

print(f"Çeşitlilik skoru: {diversity_score:.3f}")

#check missing values
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
missing_report = check_missing_values(df_train_new)

#we are checking out the data
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

check_df(df_train_new)

#we are separating cat_cols,num_cols and cat_but_car ...
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

cat_cols, cat_but_car, num_cols = grab_col_names(df_train_new)
"""
# extract target variables
feature_cols = [col for col in df_train_new.columns if col != 'prognosis']

# analyze for each symptom
for symptom in feature_cols:
    print(f"\n{'=' * 70}")
    print(f"SEMPTOM: {symptom.upper()}")
    print(f"{'=' * 70}")

    # Crosstab oluştur
    crosstab = pd.crosstab(df_train_new[symptom], df_train_new['prognosis'])

    # Eğer semptom binary (0-1) ise
    if df_train_new[symptom].nunique() == 2 and set(df_train_new[symptom].unique()) == {0, 1}:

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
        print(f"\nBu semptom {df_train_new[symptom].nunique()} farklı değer alıyor")
        print(crosstab)

    print("\n" + "-" * 70)

# Her semptom için kaç hastalıkta göründüğünü hesapla
symptom_specificity = {}

for symptom in feature_cols:
    if df_train_new[symptom].nunique() == 2:
        crosstab = pd.crosstab(df_train_new[symptom], df_train_new['prognosis'])
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
    crosstab = pd.crosstab(df_train_new[symptom], df_train_new['prognosis'])
    diseases = crosstab.loc[1][crosstab.loc[1] > 0].index.tolist()
    print(f"  → {', '.join(diseases)}\n")
"""

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


#yeni geliştirlmiş dataframe (Train test )


# Veri setini kaydet (opsiyonel)
# df_engineered.to_csv('disease_prediction_engineered.csv', index=False),

def evaluate_with_separate_test(X_train, y_train, X_test, y_test,
                                class_names=None):
    """
    Ayrı train-test seti ile model değerlendirmesi ve overfitting analizi

    Parameters:
    -----------
    X_train, y_train : Train verisi
    X_test, y_test : Test verisi
    class_names : list
        Sınıf isimleri (opsiyonel)

    Returns:
    --------
    pd.DataFrame : Model performans sonuçları
    """

    models = [
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier(random_state=42)),
        ('RF', RandomForestClassifier(random_state=42, n_estimators=100))
    ]

    results = []
    trained_models = {}

    print("\n" + "=" * 70)
    print("MODEL DEĞERLENDİRMESİ (Ayrı Train-Test)")
    print("=" * 70)

    for name, model in models:
        print(f"\n{'─' * 70}")
        print(f"📊 {name}")
        print(f"{'─' * 70}")

        # Model eğitimi
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Tahminler
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrikler
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        # Overfitting analizi
        overfitting_score = train_acc - test_acc

        if overfitting_score < 0.02:
            status = "✅ İYİ"
            level = "Yok"
            color = "🟢"
        elif overfitting_score < 0.05:
            status = "⚠️  DİKKAT"
            level = "Hafif"
            color = "🟡"
        elif overfitting_score < 0.10:
            status = "🔴 SORUN"
            level = "Orta"
            color = "🟠"
        else:
            status = "🚨 CİDDİ"
            level = "Yüksek"
            color = "🔴"

        # Sonuçları yazdır
        print(f"\n{color} OVERFITTING DURUMU: {status} - {level}")
        print(f"\n📈 Performans:")
        print(f"   Train Accuracy:  {train_acc:.4f}")
        print(f"   Test Accuracy:   {test_acc:.4f}")
        print(f"   Test Precision:  {test_precision:.4f}")
        print(f"   Test Recall:     {test_recall:.4f}")
        print(f"   Test F1-Score:   {test_f1:.4f}")
        print(f"\n🎯 Overfitting Skoru: {overfitting_score:.4f}")

        results.append({
            'Model': name,
            'Train_Acc': train_acc,
            'Test_Acc': test_acc,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1_Score': test_f1,
            'Overfitting': overfitting_score,
            'Level': level,
            'Status': status
        })

    results_df = pd.DataFrame(results)

    # Görselleştirme
    plot_results(results_df, trained_models, X_test, y_test, class_names)

    # Özet
    print("\n" + "=" * 70)
    print("📋 SONUÇ TABLOSU")
    print("=" * 70)
    print(results_df[['Model', 'Train_Acc', 'Test_Acc', 'Overfitting', 'Level']].to_string(index=False))

    # En iyi model
    best_idx = results_df['Test_Acc'].idxmax()
    best = results_df.iloc[best_idx]

    print("\n" + "=" * 70)
    print("🏆 EN İYİ MODEL")
    print("=" * 70)
    print(f"Model:             {best['Model']}")
    print(f"Test Accuracy:     {best['Test_Acc']:.4f}")
    print(f"Overfitting:       {best['Overfitting']:.4f} ({best['Level']})")
    print(f"Precision:         {best['Precision']:.4f}")
    print(f"Recall:            {best['Recall']:.4f}")
    print(f"F1-Score:          {best['F1_Score']:.4f}")
    print("=" * 70)

    return results_df, trained_models


def plot_results(results_df, trained_models, X_test, y_test, class_names=None):
    """Sonuçları görselleştirir"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Train vs Test Accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(results_df))
    width = 0.35

    ax1.bar(x - width / 2, results_df['Train_Acc'], width,
            label='Train', color='#3498db', alpha=0.8)
    ax1.bar(x + width / 2, results_df['Test_Acc'], width,
            label='Test', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Train vs Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 2. Overfitting Skorları
    ax2 = fig.add_subplot(gs[0, 2])
    colors = ['#2ecc71' if x < 0.02 else '#f39c12' if x < 0.05
    else '#e67e22' if x < 0.10 else '#c0392b'
              for x in results_df['Overfitting']]

    ax2.barh(results_df['Model'], results_df['Overfitting'], color=colors, alpha=0.8)
    ax2.axvline(x=0.02, color='g', linestyle='--', linewidth=1, label='İyi')
    ax2.axvline(x=0.05, color='orange', linestyle='--', linewidth=1, label='Dikkat')
    ax2.axvline(x=0.10, color='red', linestyle='--', linewidth=1, label='Sorun')
    ax2.set_xlabel('Overfitting Skoru', fontsize=10)
    ax2.set_title('Overfitting Seviyeleri', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Performans Metrikleri
    ax3 = fig.add_subplot(gs[1, :])
    metrics_data = results_df[['Model', 'Test_Acc', 'Precision', 'Recall', 'F1_Score']].set_index('Model')

    x = np.arange(len(results_df))
    width = 0.2

    ax3.bar(x - 1.5 * width, metrics_data['Test_Acc'], width, label='Accuracy', color='#3498db', alpha=0.8)
    ax3.bar(x - 0.5 * width, metrics_data['Precision'], width, label='Precision', color='#2ecc71', alpha=0.8)
    ax3.bar(x + 0.5 * width, metrics_data['Recall'], width, label='Recall', color='#f39c12', alpha=0.8)
    ax3.bar(x + 1.5 * width, metrics_data['F1_Score'], width, label='F1-Score', color='#9b59b6', alpha=0.8)

    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('Skor', fontsize=11)
    ax3.set_title('Tüm Performans Metrikleri (Test Set)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['Model'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])

    # 4. En iyi modelin confusion matrix'i
    best_model_name = results_df.loc[results_df['Test_Acc'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]

    ax4 = fig.add_subplot(gs[2, :])
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Küçük confusion matrix için normalize et
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                ax=ax4, cbar_kws={'label': 'Normalized Count'})
    ax4.set_title(f'Confusion Matrix - {best_model_name} (Test Set)',
                  fontsize=13, fontweight='bold')
    ax4.set_ylabel('Gerçek', fontsize=11)
    ax4.set_xlabel('Tahmin', fontsize=11)

    plt.suptitle('Overfitting Analizi - Ayrı Train-Test Değerlendirmesi',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.show()


##############################################################################
#TEST VERİ SETİ ÇOK AZZZZZZZ  (Genellenebilir değil)
##############################################################################

df_engineered_train = perform_feature_engineering(df_train_new)
df_engineered_test = perform_feature_engineering(df_test_new)

"""""
X_train = df_engineered_train.drop('prognosis', axis=1)
X_test = df_engineered_test.drop('prognosis', axis=1)

# Hedef değişkeni encode et
le = LabelEncoder()
y_train_encoded = le.fit_transform(df_engineered_train['prognosis'])
y_test_encoded = le.transform(df_engineered_test['prognosis'])

results_df, trained_models = evaluate_with_separate_test(
    X_train, y_train_encoded, X_test, y_test_encoded,
    class_names=le.classes_
)
"""""
#FEATURE IMPORTANCE
def analyze_feature_importance(X_train, y_train, X_test, y_test,
                               feature_names=None, top_n=20):
    """
    Detaylı feature importance analizi

    Parameters:
    -----------
    X_train, y_train : Train verisi
    X_test, y_test : Test verisi
    feature_names : list
        Feature isimleri (None ise X_train.columns kullanılır)
    top_n : int
        Gösterilecek en önemli feature sayısı

    Returns:
    --------
    dict : Her yöntem için importance değerleri
    """

    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

    results = {}

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALİZİ")
    print("=" * 80)

    # 1. Random Forest Feature Importance (Gini-based)
    print("\n📊 1. Random Forest - Gini Importance")
    print("─" * 80)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    rf_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    results['random_forest'] = rf_importance

    print(f"\n🔝 Top {min(top_n, len(rf_importance))} Önemli Feature:")
    print(rf_importance.head(top_n).to_string(index=False))

    # Test accuracy
    rf_test_score = rf_model.score(X_test, y_test)
    print(f"\n📈 Random Forest Test Accuracy: {rf_test_score:.4f}")

    # 2. Decision Tree Feature Importance
    print("\n" + "─" * 80)
    print("📊 2. Decision Tree - Feature Importance")
    print("─" * 80)

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)

    dt_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    results['decision_tree'] = dt_importance

    print(f"\n🔝 Top {min(top_n, len(dt_importance))} Önemli Feature:")
    print(dt_importance.head(top_n).to_string(index=False))

    dt_test_score = dt_model.score(X_test, y_test)
    print(f"\n📈 Decision Tree Test Accuracy: {dt_test_score:.4f}")

    # 3. Permutation Importance (Test seti üzerinde)
    print("\n" + "─" * 80)
    print("📊 3. Permutation Importance (Test Seti)")
    print("─" * 80)
    print("⏳ Hesaplanıyor... (biraz zaman alabilir)")

    perm_importance = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)

    results['permutation'] = perm_importance_df

    print(f"\n🔝 Top {min(top_n, len(perm_importance_df))} Önemli Feature:")
    print(perm_importance_df.head(top_n).to_string(index=False))

    # 4. Görselleştirme
    plot_feature_importance(results, top_n)

    # 5. Karşılaştırma analizi
    compare_importance_methods(results, top_n)

    # 6. Öneriler
    print_feature_recommendations(results, top_n)

    return results

def plot_feature_importance(results, top_n=20):
    """Feature importance görselleştirmeleri"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Importance Analizi', fontsize=16, fontweight='bold')

    # 1. Random Forest - Bar plot
    rf_top = results['random_forest'].head(top_n)
    axes[0, 0].barh(range(len(rf_top)), rf_top['Importance'],
                    color='#3498db', alpha=0.8)
    axes[0, 0].set_yticks(range(len(rf_top)))
    axes[0, 0].set_yticklabels(rf_top['Feature'], fontsize=9)
    axes[0, 0].set_xlabel('Importance (Gini)', fontsize=10)
    axes[0, 0].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # 2. Decision Tree - Bar plot
    dt_top = results['decision_tree'].head(top_n)
    axes[0, 1].barh(range(len(dt_top)), dt_top['Importance'],
                    color='#2ecc71', alpha=0.8)
    axes[0, 1].set_yticks(range(len(dt_top)))
    axes[0, 1].set_yticklabels(dt_top['Feature'], fontsize=9)
    axes[0, 1].set_xlabel('Importance (Gini)', fontsize=10)
    axes[0, 1].set_title('Decision Tree Feature Importance', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # 3. Permutation Importance - Bar plot with error bars
    perm_top = results['permutation'].head(top_n)
    axes[1, 0].barh(range(len(perm_top)), perm_top['Importance'],
                    xerr=perm_top['Std'], color='#e74c3c', alpha=0.8,
                    error_kw={'elinewidth': 1, 'alpha': 0.5})
    axes[1, 0].set_yticks(range(len(perm_top)))
    axes[1, 0].set_yticklabels(perm_top['Feature'], fontsize=9)
    axes[1, 0].set_xlabel('Importance (Decrease in Accuracy)', fontsize=10)
    axes[1, 0].set_title('Permutation Importance (Test Set)', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Karşılaştırma - Top 10 feature'ların 3 yöntemde karşılaştırması
    # En önemli 10 feature'ı al (RF'den)
    top_features = results['random_forest'].head(10)['Feature'].tolist()

    comparison_data = []
    for feature in top_features:
        rf_val = results['random_forest'][results['random_forest']['Feature'] == feature]['Importance'].values[0]
        dt_val = results['decision_tree'][results['decision_tree']['Feature'] == feature]['Importance'].values[0]
        perm_val = results['permutation'][results['permutation']['Feature'] == feature]['Importance'].values[0]

        comparison_data.append({
            'Feature': feature,
            'RF': rf_val,
            'DT': dt_val,
            'Perm': perm_val
        })

    comp_df = pd.DataFrame(comparison_data)

    x = np.arange(len(comp_df))
    width = 0.25

    axes[1, 1].bar(x - width, comp_df['RF'], width, label='Random Forest',
                   color='#3498db', alpha=0.8)
    axes[1, 1].bar(x, comp_df['DT'], width, label='Decision Tree',
                   color='#2ecc71', alpha=0.8)
    axes[1, 1].bar(x + width, comp_df['Perm'], width, label='Permutation',
                   color='#e74c3c', alpha=0.8)

    axes[1, 1].set_xlabel('Features', fontsize=10)
    axes[1, 1].set_ylabel('Normalized Importance', fontsize=10)
    axes[1, 1].set_title('Top 10 Features - Yöntem Karşılaştırması', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(comp_df['Feature'], rotation=45, ha='right', fontsize=8)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def compare_importance_methods(results, top_n=10):
    """Farklı importance yöntemlerini karşılaştırır"""

    print("\n" + "=" * 80)
    print("YÖNTEM KARŞILAŞTIRMASI")
    print("=" * 80)

    # Top N feature'ları al
    rf_top = set(results['random_forest'].head(top_n)['Feature'])
    dt_top = set(results['decision_tree'].head(top_n)['Feature'])
    perm_top = set(results['permutation'].head(top_n)['Feature'])

    # Kesişim analizi
    all_methods = rf_top & dt_top & perm_top
    two_methods = (rf_top & dt_top) | (rf_top & perm_top) | (dt_top & perm_top)
    two_methods = two_methods - all_methods

    print(f"\n📊 Top {top_n} Feature Analizi:")
    print(f"\n✅ 3 yöntemde de önemli ({len(all_methods)} feature):")
    if all_methods:
        for feat in sorted(all_methods):
            print(f"   • {feat}")
    else:
        print("   (Yok)")

    print(f"\n⚠️  2 yöntemde önemli ({len(two_methods)} feature):")
    if two_methods:
        for feat in sorted(two_methods):
            in_methods = []
            if feat in rf_top:
                in_methods.append("RF")
            if feat in dt_top:
                in_methods.append("DT")
            if feat in perm_top:
                in_methods.append("Perm")
            print(f"   • {feat} ({', '.join(in_methods)})")
    else:
        print("   (Yok)")

    # Tutarlılık skoru
    consistency = len(all_methods) / top_n
    print(f"\n📈 Tutarlılık Skoru: {consistency:.2%}")

    if consistency > 0.7:
        print("   ✅ Yüksek tutarlılık - Sonuçlar güvenilir")
    elif consistency > 0.4:
        print("   ⚠️  Orta tutarlılık - Sonuçları dikkatle yorumlayın")
    else:
        print("   🚨 Düşük tutarlılık - Feature'lar kararsız olabilir")


def print_feature_recommendations(results, top_n=20):
    """Feature importance'a göre öneriler"""

    print("\n" + "=" * 80)
    print("💡 ÖNERİLER VE YORUMLAR")
    print("=" * 80)

    rf_importance = results['random_forest']

    # En önemli feature'lar
    top_features = rf_importance.head(5)
    print("\n🎯 En Kritik 5 Feature (Random Forest):")
    for idx, row in top_features.iterrows():
        print(f"   {idx + 1}. {row['Feature']}: {row['Importance']:.4f}")

    # Düşük önemli feature'lar
    low_importance = rf_importance[rf_importance['Importance'] < 0.01]

    print(f"\n📉 Düşük Önemli Feature'lar ({len(low_importance)} adet):")
    if len(low_importance) > 0:
        print(f"   • Importance < 0.01 olan {len(low_importance)} feature var")
        print(f"   • Bu feature'ları çıkarmayı düşünebilirsiniz")
        print(f"   • Model daha basit ve hızlı olabilir")

        if len(low_importance) <= 10:
            print("\n   Düşük önemli feature'lar:")
            for idx, row in low_importance.iterrows():
                print(f"      • {row['Feature']}: {row['Importance']:.4f}")
    else:
        print("   • Tüm feature'lar önemli (>0.01)")

    # Öneriler
    print("\n" + "─" * 80)
    print("📋 AKSIYONLAR:")
    print("─" * 80)

    total_importance = rf_importance['Importance'].sum()
    cumsum_importance = rf_importance['Importance'].cumsum()

    # %90 importance için kaç feature yeterli?
    n_for_90 = (cumsum_importance <= 0.90 * total_importance).sum() + 1

    print(f"\n1. 📊 Feature Seçimi:")
    print(f"   • Toplam {len(rf_importance)} feature var")
    print(f"   • İlk {n_for_90} feature toplam önemin %90'ını sağlıyor")
    print(f"   • {len(rf_importance) - n_for_90} feature'ı çıkarabilirsiniz")

    print(f"\n2. 🔧 Model Optimizasyonu:")
    if len(low_importance) > 10:
        print(f"   • {len(low_importance)} düşük önemli feature'ı çıkarın")
        print(f"   • Model eğitim süresi azalır")
        print(f"   • Overfitting riski azalır")
    else:
        print(f"   • Tüm feature'lar önemli, hepsini kullanın")

    print(f"\n3. 📈 Feature Engineering:")
    print(f"   • Top 5 feature'ı temel alarak yeni feature'lar türetin")
    print(f"   • Bu feature'ların kombinasyonlarını deneyin")
    print(f"   • Domain knowledge ile zenginleştirin")

    print("\n" + "=" * 80)


def create_feature_importance_summary(results, output_file='feature_importance_summary.csv'):
    """Feature importance sonuçlarını CSV'ye kaydeder"""

    # Tüm sonuçları birleştir
    rf_imp = results['random_forest'][['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'})
    dt_imp = results['decision_tree'][['Feature', 'Importance']].rename(columns={'Importance': 'DT_Importance'})
    perm_imp = results['permutation'][['Feature', 'Importance']].rename(columns={'Importance': 'Perm_Importance'})

    summary = rf_imp.merge(dt_imp, on='Feature').merge(perm_imp, on='Feature')

    # Ortalama importance
    summary['Avg_Importance'] = summary[['RF_Importance', 'DT_Importance', 'Perm_Importance']].mean(axis=1)
    summary = summary.sort_values('Avg_Importance', ascending=False)

    # Kaydet
    summary.to_csv(output_file, index=False)
    print(f"\n✅ Feature importance summary kaydedildi: {output_file}")

    return summary


# ==================== KULLANIM ÖRNEĞİ ====================

"""""
# 1. Feature importance analizi
importance_results = analyze_feature_importance(
    X_train, y_train_encoded, X_test, y_test_encoded,
    feature_names=X_train.columns,
    top_n=20
)

# 2. Sonuçları kaydet
summary = create_feature_importance_summary(importance_results)

# 3. Düşük önemli feature'ları çıkar
low_importance_features = importance_results['random_forest'][
    importance_results['random_forest']['Importance'] < 0.01
]['Feature'].tolist()

print(f"\\n🗑️  Çıkarılabilecek {len(low_importance_features)} feature:")
print(low_importance_features)

# 4. Feature selection yapılmış yeni veri
X_train_selected = X_train.drop(columns=low_importance_features)
X_test_selected = X_test.drop(columns=low_importance_features)

print(f"\\n✅ Yeni feature sayısı: {X_train_selected.shape[1]} (Önceki: {X_train.shape[1]})")

# 5. Yeni veri ile model eğit
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_selected, y_train_encoded)

print(f"Feature selection sonrası test accuracy: {model.score(X_test_selected, y_test_encoded):.4f}")
"""


from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')


def proper_evaluation_with_small_test(df_train, df_test, target_col='prognosis'):
    """
    Küçük test seti için uygun değerlendirme stratejisi

    Strateji:
    1. Train'i temizle ve böl (internal validation)
    2. Cross-validation yap (güvenilir metrik)
    3. Orijinal test'i final validation olarak kullan

    Parameters:
    -----------
    df_train : pd.DataFrame
        Train verisi
    df_test : pd.DataFrame
        Test verisi (küçük)
    target_col : str
        Hedef değişken
    """

    print("=" * 80)
    print("KÜÇÜK TEST SETİ İÇİN UYGUN DEĞERLENDİRME")
    print("=" * 80)

    # 1. VERİ HAZIRLIĞI
    print("\n📊 1. VERİ HAZIRLIĞI")
    print("─" * 80)

    # Duplicate temizliği
    df_train_clean = df_train.drop_duplicates()
    df_test_clean = df_test.drop_duplicates()

    print(f"Train - Öncesi: {len(df_train):,} → Sonrası: {len(df_train_clean):,}")
    print(f"Test - Öncesi: {len(df_test):,} → Sonrası: {len(df_test_clean):,}")

    # Test seti analizi
    print(f"\n⚠️  UYARI: Test seti çok küçük!")
    print(f"   Test boyutu: {len(df_test_clean)} örnek")
    print(f"   Train/Test oranı: {len(df_train_clean) / len(df_test_clean):.1f}:1")

    test_disease_count = df_test_clean[target_col].value_counts()
    print(f"   Test'te hastalık başına ortalama: {test_disease_count.mean():.1f} örnek")

    if test_disease_count.min() == 0:
        print(f"   🚨 Bazı hastalıklar test'te yok!")

    # X, y hazırla
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    X_train_full = df_train_clean.drop(target_col, axis=1)
    y_train_full = le.fit_transform(df_train_clean[target_col])

    X_test_original = df_test_clean.drop(target_col, axis=1)
    y_test_original = le.transform(df_test_clean[target_col])

    # 2. STRATEJİ 1: TRAIN'İ BÖL (Internal Validation)
    print("\n" + "=" * 80)
    print("📊 2. STRATEJİ 1: TRAIN'İ BÖL (Internal Validation)")
    print("=" * 80)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    print(f"\nBölme sonrası:")
    print(f"  Internal Train: {len(X_train):,} örnek")
    print(f"  Internal Val:   {len(X_val):,} örnek")
    print(f"  Original Test:  {len(X_test_original):,} örnek")

    models = {
        'KNN': KNeighborsClassifier(),
        'CART': DecisionTreeClassifier(random_state=42, max_depth=10),
        'RF': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    print("\n" + "─" * 80)
    print("Internal Validation Sonuçları:")
    print("─" * 80)

    internal_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        internal_results.append({
            'Model': name,
            'Train_Acc': train_acc,
            'Val_Acc': val_acc,
            'Overfitting': train_acc - val_acc
        })

        status = "✅" if (train_acc - val_acc) < 0.05 else "⚠️"
        print(f"{name:6s} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Diff: {train_acc - val_acc:.4f} {status}")

    internal_df = pd.DataFrame(internal_results)

    # 3. STRATEJİ 2: CROSS-VALIDATION (En Güvenilir)
    print("\n" + "=" * 80)
    print("📊 3. STRATEJİ 2: CROSS-VALIDATION (5-Fold)")
    print("=" * 80)
    print("⏳ Hesaplanıyor...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    for name, model in models.items():
        results = cross_validate(
            model, X_train_full, y_train_full,
            cv=cv,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
            return_train_score=True,
            n_jobs=-1
        )

        train_mean = results['train_accuracy'].mean()
        test_mean = results['test_accuracy'].mean()
        test_std = results['test_accuracy'].std()

        cv_results.append({
            'Model': name,
            'CV_Train': train_mean,
            'CV_Test': test_mean,
            'CV_Std': test_std,
            'Overfitting': train_mean - test_mean,
            'Precision': results['test_precision_weighted'].mean(),
            'Recall': results['test_recall_weighted'].mean(),
            'F1': results['test_f1_weighted'].mean()
        })

    cv_df = pd.DataFrame(cv_results)

    print("\n" + "─" * 80)
    print("Cross-Validation Sonuçları (5-Fold):")
    print("─" * 80)
    print(cv_df[['Model', 'CV_Train', 'CV_Test', 'CV_Std', 'Overfitting']].to_string(index=False))

    # 4. STRATEJİ 3: ORİJİNAL TEST (Final Check)
    print("\n" + "=" * 80)
    print("📊 4. STRATEJİ 3: ORİJİNAL TEST SETİ (Final Validation)")
    print("=" * 80)
    print(f"⚠️  Not: {len(X_test_original)} örnek çok az, sadece referans amaçlı!\n")

    final_results = []

    for name, model in trained_models.items():
        # Full train ile yeniden eğit
        model.fit(X_train_full, y_train_full)

        final_acc = model.score(X_test_original, y_test_original)

        final_results.append({
            'Model': name,
            'Final_Test_Acc': final_acc
        })

        print(f"{name:6s} | Orijinal Test Accuracy: {final_acc:.4f}")

    final_df = pd.DataFrame(final_results)

    # 5. KARŞILAŞTIRMALı SONUÇLAR
    print("\n" + "=" * 80)
    print("📊 5. KARŞILAŞTIRMALI ÖZET")
    print("=" * 80)

    # Merge all results
    summary = internal_df[['Model', 'Val_Acc']].merge(
        cv_df[['Model', 'CV_Test', 'CV_Std']], on='Model'
    ).merge(
        final_df[['Model', 'Final_Test_Acc']], on='Model'
    )

    summary.columns = ['Model', 'Internal_Val', 'CV_Mean', 'CV_Std', 'Original_Test']

    print("\n" + summary.to_string(index=False))

    # Güvenilirlik analizi
    print("\n" + "─" * 80)
    print("🎯 GÜVENİLİRLİK ANALİZİ:")
    print("─" * 80)

    for idx, row in summary.iterrows():
        model_name = row['Model']

        # Confidence interval for original test (42 samples)
        n = len(X_test_original)
        acc = row['Original_Test']

        if n > 0 and acc < 1.0:
            # Wilson score interval
            z = 1.96  # 95% confidence
            phat = acc
            denominator = 1 + z ** 2 / n
            centre = (phat + z ** 2 / (2 * n)) / denominator
            adjustment = z * np.sqrt((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denominator
            ci_lower = max(0, centre - adjustment)
            ci_upper = min(1, centre + adjustment)

            print(f"\n{model_name}:")
            print(f"  ✅ En güvenilir: CV = {row['CV_Mean']:.4f} (±{row['CV_Std']:.4f})")
            print(f"  ⚠️  Original Test = {acc:.4f} (42 örnek)")
            print(f"     → %95 güven aralığı: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"     → Gerçek accuracy muhtemelen bu aralıkta")
        else:
            print(f"\n{model_name}:")
            print(f"  ✅ En güvenilir: CV = {row['CV_Mean']:.4f} (±{row['CV_Std']:.4f})")
            print(f"  ⚠️  Original Test = {acc:.4f} (42 örnek - çok az!)")

    # 6. GÖRSELLEŞTİRME
    plot_comparison(summary, cv_df)

    # 7. ÖNERİLER
    print("\n" + "=" * 80)
    print("💡 ÖNERİLER")
    print("=" * 80)

    best_model = cv_df.loc[cv_df['CV_Test'].idxmax(), 'Model']

    print(f"\n🏆 En iyi model: {best_model}")
    print(f"   CV Test Accuracy: {cv_df.loc[cv_df['Model'] == best_model, 'CV_Test'].values[0]:.4f}")
    print(f"   CV Std: {cv_df.loc[cv_df['Model'] == best_model, 'CV_Std'].values[0]:.4f}")

    print("\n📋 Aksiyonlar:")
    print("1. ✅ Cross-validation sonuçlarına güvenin (en güvenilir)")
    print(f"2. ⚠️  Orijinal {len(X_test_original)} test örneğine çok güvenmeyin")
    print("3. 💡 Eğer mümkünse daha fazla test verisi toplayın (min 100+)")
    print("4. 🎯 Final deployment öncesi yeni verilerle validate edin")

    print("\n" + "=" * 80)

    return {
        'internal_validation': internal_df,
        'cross_validation': cv_df,
        'final_test': final_df,
        'summary': summary,
        'trained_models': trained_models,
        'label_encoder': le
    }

def plot_comparison(summary_df, cv_df):
    """Sonuçları görselleştirir"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Farklı değerlendirme yöntemlerinin karşılaştırması
    x = np.arange(len(summary_df))
    width = 0.25

    axes[0].bar(x - width, summary_df['Internal_Val'], width,
                label='Internal Val', color='#3498db', alpha=0.8)
    axes[0].bar(x, summary_df['CV_Mean'], width,
                label='CV (5-fold)', color='#2ecc71', alpha=0.8)
    axes[0].bar(x + width, summary_df['Original_Test'], width,
                label='Original Test (42)', color='#e74c3c', alpha=0.8)

    # CV için error bars
    axes[0].errorbar(x, summary_df['CV_Mean'],
                     yerr=summary_df['CV_Std'],
                     fmt='none', color='black', capsize=3, linewidth=1)

    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Farklı Değerlendirme Yöntemleri Karşılaştırması')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df['Model'])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.05])

    # 2. CV sonuçları detaylı
    cv_metrics = cv_df[['Model', 'CV_Test', 'Precision', 'Recall', 'F1']].set_index('Model')

    cv_metrics.plot(kind='bar', ax=axes[1], width=0.8, alpha=0.8)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Cross-Validation Metrikleri (En Güvenilir)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1.05])
    axes[1].set_xticklabels(cv_metrics.index, rotation=0)

    plt.tight_layout()
    plt.show()


# ==================== KULLANIM ====================


# Kullanım:
results = proper_evaluation_with_small_test(
    df_train=df_engineered_train,  # Train veri setiniz
    df_test=df_engineered_test,         # 42 örneklik test setiniz
    target_col='prognosis'
)


def comprehensive_data_and_model_analysis(df_train, df_test, results, target_col='prognosis'):
    """
    Veri kalitesi ve model performansının kapsamlı analizi
    """

    print("=" * 80)
    print("KAPSAMLI VERİ VE MODEL ANALİZİ")
    print("=" * 80)

    # ============================================================================
    # 1. VERİ KALİTESİ ANALİZİ
    # ============================================================================
    print("\n📊 1. VERİ KALİTESİ ANALİZİ")
    print("─" * 80)

    # Duplicate analizi
    print("\n🔍 Duplicate Analizi:")
    print(f"Orijinal train boyutu: {len(df_train):,}")
    print(f"Temizlenmiş boyut: {len(df_train.drop_duplicates()):,}")
    print(f"Duplicate oranı: {(1 - len(df_train.drop_duplicates()) / len(df_train)) * 100:.1f}%")

    # Hastalık dağılımı
    disease_dist = df_train[target_col].value_counts()
    print(f"\n📈 Hastalık Dağılımı:")
    print(f"Toplam hastalık sayısı: {len(disease_dist)}")
    print(f"Örnek başına ortalama: {len(df_train) / len(disease_dist):.1f}")
    print(f"\nİlk 10 hastalık:")
    print(disease_dist.head(10))

    # Sınıf dengesizliği
    print(f"\n⚖️ Sınıf Dengesizliği:")
    print(f"En fazla örnek: {disease_dist.max()}")
    print(f"En az örnek: {disease_dist.min()}")
    print(f"İmbalance ratio: {disease_dist.max() / disease_dist.min():.1f}:1")

    # Duplicate'lerin özellikleri
    df_clean = df_train.drop_duplicates()
    duplicates = df_train[df_train.duplicated(keep=False)]

    if len(duplicates) > 0:
        print(f"\n🔄 Duplicate Özellikleri:")
        print(f"Duplicate örnek sayısı: {len(duplicates):,}")
        print(f"Duplicate içeren hastalık sayısı: {duplicates[target_col].nunique()}")
        print("\nEn çok duplicate olan hastalıklar:")
        dup_diseases = duplicates[target_col].value_counts().head(5)
        for disease, count in dup_diseases.items():
            original_count = df_train[df_train[target_col] == disease].shape[0]
            unique_count = df_clean[df_clean[target_col] == disease].shape[0]
            print(f"  {disease}: {original_count} → {unique_count} "
                  f"({original_count - unique_count} duplicate)")

    # ============================================================================
    # 2. FEATURE ANALİZİ
    # ============================================================================
    print("\n\n📊 2. FEATURE ANALİZİ")
    print("─" * 80)

    X = df_clean.drop(target_col, axis=1)

    # Feature istatistikleri
    print(f"\n📈 Feature İstatistikleri:")
    print(f"Toplam feature sayısı: {X.shape[1]}")
    print(f"Feature değer aralığı: [{X.values.min()}, {X.values.max()}]")

    # Sıfır olmayan feature'lar
    non_zero_counts = (X != 0).sum(axis=1)
    print(f"\nÖrnek başına aktif feature (sıfır olmayan):")
    print(f"  Ortalama: {non_zero_counts.mean():.1f}")
    print(f"  Min: {non_zero_counts.min()}")
    print(f"  Max: {non_zero_counts.max()}")

    # Feature variance
    feature_vars = X.var()
    zero_var_features = (feature_vars == 0).sum()
    print(f"\nSıfır varyans'lı feature sayısı: {zero_var_features}")

    # ============================================================================
    # 3. MODEL PERFORMANS ANALİZİ
    # ============================================================================
    print("\n\n📊 3. MODEL PERFORMANS ANALİZİ")
    print("─" * 80)

    rf_model = results['trained_models']['RF']
    knn_model = results['trained_models']['KNN']

    # RF Feature Importance
    print("\n🌲 Random Forest - Top 10 Önemli Feature:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10).to_string(index=False))

    # Importance dağılımı
    print(f"\nFeature Importance Dağılımı:")
    print(f"  Top 10 feature toplam importance: {feature_importance.head(10)['importance'].sum():.4f}")
    print(f"  Top 20 feature toplam importance: {feature_importance.head(20)['importance'].sum():.4f}")

    # ============================================================================
    # 4. OVERFITTING ANALİZİ
    # ============================================================================
    print("\n\n📊 4. OVERFITTING ANALİZİ")
    print("─" * 80)

    cv_results = results['cross_validation']

    print("\n🎯 Overfitting İndikatörleri:")
    for idx, row in cv_results.iterrows():
        model_name = row['Model']
        train_acc = row['CV_Train']
        test_acc = row['CV_Test']
        diff = train_acc - test_acc

        if diff < 0.01:
            status = "✅ İyi"
        elif diff < 0.05:
            status = "⚠️ Kabul edilebilir"
        else:
            status = "🚨 Overfit!"

        print(f"{model_name:6s} | Train: {train_acc:.4f} | CV: {test_acc:.4f} | "
              f"Diff: {diff:.4f} | {status}")

    # RF perfect score analizi
    print("\n🔍 Random Forest Perfect Score Analizi:")
    rf_row = cv_results[cv_results['Model'] == 'RF'].iloc[0]
    if rf_row['CV_Train'] == 1.0 and rf_row['CV_Test'] == 1.0:
        print("⚠️  UYARI: RF hem train hem test'te perfect score!")
        print("\nOlası nedenler:")
        print("1. Veri çok basit/kolay (277 örnek, duplicate'ler temizlenmiş)")
        print("2. Feature'lar hastalıkları mükemmel ayırt ediyor")
        print("3. Veri sızıntısı (data leakage) olabilir")
        print("4. Test fold'ları çok küçük (55 örnek/fold)")

        # Complexity check
        n_samples = len(df_clean)
        n_features = X.shape[1]
        n_classes = df_clean[target_col].nunique()

        print(f"\n📊 Model Complexity:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: {n_features}")
        print(f"  Classes: {n_classes}")
        print(f"  Samples/Class: {n_samples / n_classes:.1f}")
        print(f"  Features/Class: {n_features / n_classes:.1f}")

    # ============================================================================
    # 5. TEST SETİ ANALİZİ
    # ============================================================================
    print("\n\n📊 5. TEST SETİ ANALİZİ")
    print("─" * 80)

    test_diseases = df_test[target_col].value_counts()
    train_diseases = df_clean[target_col].value_counts()

    print(f"\n📈 Test Seti Kapsama:")
    print(f"Test'teki hastalık sayısı: {len(test_diseases)}")
    print(f"Train'deki hastalık sayısı: {len(train_diseases)}")

    # Overlap analizi
    test_disease_set = set(test_diseases.index)
    train_disease_set = set(train_diseases.index)

    overlap = test_disease_set & train_disease_set
    only_test = test_disease_set - train_disease_set

    print(f"\nOrtak hastalıklar: {len(overlap)}")
    if len(only_test) > 0:
        print(f"⚠️  Sadece test'te olan: {len(only_test)}")
        print(f"   → {list(only_test)}")

    # ============================================================================
    # 6. GÖRSELLEŞTİRME
    # ============================================================================
    plot_detailed_analysis(df_clean, feature_importance, cv_results, target_col)

    # ============================================================================
    # 7. ÖNERİLER
    # ============================================================================
    print("\n\n📊 7. ÖNERİLER VE SONUÇ")
    print("=" * 80)

    print("\n🎯 SONUÇ:")
    if rf_row['CV_Test'] == 1.0:
        print("RF'nin perfect score'u GEÇERLİ görünüyor çünkü:")
        print("✅ 1. Duplicate'ler temizlenmiş (277 unique örnek)")
        print("✅ 2. CV'de consistent (5-fold hepsi 1.0)")
        print("✅ 3. Test seti de benzer performans (0.976)")
        print("\nAncak:")
        print("⚠️  1. Veri seti çok küçük (277 örnek)")
        print("⚠️  2. Test seti yetersiz (42 örnek)")
        print("⚠️  3. Real-world validation gerekli")

    print("\n💡 TAVSİYELER:")
    print("1. ✅ RF modelini kullanabilirsiniz (en iyi performans)")
    print("2. 📊 Daha fazla veri toplayın (özellikle test için)")
    print("3. 🔍 Production'da performansı yakından izleyin")
    print("4. 🎯 Yeni hastalarla real-world validation yapın")
    print("5. 💾 Model versiyonlamayı unutmayın")

    print("\n📝 MODEL KAYDETME:")
    print("```python")
    print("import joblib")
    print("joblib.dump(results['trained_models']['RF'], 'rf_model.pkl')")
    print("joblib.dump(results['label_encoder'], 'label_encoder.pkl')")
    print("```")

    print("\n" + "=" * 80)


def plot_detailed_analysis(df_clean, feature_importance, cv_results, target_col):
    """Detaylı görselleştirmeler"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Hastalık dağılımı
    ax1 = fig.add_subplot(gs[0, :2])
    disease_counts = df_clean[target_col].value_counts()
    disease_counts.head(20).plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Örnek Sayısı')
    ax1.set_title('Top 20 Hastalık Dağılımı')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. Feature importance
    ax2 = fig.add_subplot(gs[0, 2])
    top_features = feature_importance.head(15)
    ax2.barh(range(len(top_features)), top_features['importance'], color='coral')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'], fontsize=8)
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 15 Features (RF)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. CV metrikleri
    ax3 = fig.add_subplot(gs[1, :])
    cv_metrics = cv_results[['Model', 'CV_Test', 'Precision', 'Recall', 'F1']].set_index('Model')
    cv_metrics.plot(kind='bar', ax=ax3, width=0.8, alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Cross-Validation Metrikleri')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.05])
    ax3.set_xticklabels(cv_metrics.index, rotation=0)

    # 4. Train vs Test accuracy
    ax4 = fig.add_subplot(gs[2, 0])
    models = cv_results['Model']
    x = np.arange(len(models))
    width = 0.35
    ax4.bar(x - width / 2, cv_results['CV_Train'], width, label='Train', alpha=0.8)
    ax4.bar(x + width / 2, cv_results['CV_Test'], width, label='CV Test', alpha=0.8)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Train vs Test Accuracy')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.05])

    # 5. Overfitting analizi
    ax5 = fig.add_subplot(gs[2, 1])
    overfitting = cv_results['CV_Train'] - cv_results['CV_Test']
    colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in overfitting]
    ax5.bar(models, overfitting, color=colors, alpha=0.7)
    ax5.axhline(y=0.05, color='orange', linestyle='--', label='Threshold (0.05)')
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Train - Test Accuracy')
    ax5.set_title('Overfitting Analizi')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. CV std (stability)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.bar(models, cv_results['CV_Std'], color='purple', alpha=0.7)
    ax6.set_xlabel('Model')
    ax6.set_ylabel('CV Std Dev')
    ax6.set_title('Model Stability (CV Std)')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Kapsamlı Model ve Veri Analizi', fontsize=14, fontweight='bold')
    plt.show()

# ==================== KULLANIM ====================

# Kullanım:
comprehensive_data_and_model_analysis(
    df_train=df_engineered_train,  # Orijinal train (duplicate'li)
    df_test=df_engineered_test,
    results=results,  # proper_evaluation_with_small_test sonucu
    target_col='prognosis'
)


