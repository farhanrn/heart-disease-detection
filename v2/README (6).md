
# Laporan Proyek Machine Learning - Farhan bRahman
-------------------------------------------------
## Domain Proyek
Penyakit kardiovaskular, termasuk penyakit jantung, adalah penyebab utama kematian di seluruh dunia. Menurut World Health Organization (WHO), penyakit kardiovaskular menyebabkan lebih dari 17 juta kematian setiap tahun, yang mencakup sekitar 31% dari total kematian global. Deteksi dini dan intervensi tepat waktu dapat secara signifikan mengurangi risiko kematian akibat penyakit ini.

Penyakit jantung sering kali tidak terdiagnosis hingga gejala serius muncul, sehingga kebutuhan akan sistem deteksi dini menjadi semakin mendesak. Teknologi Machine Learning menawarkan pendekatan berbasis data untuk memprediksi risiko penyakit jantung dengan memanfaatkan data kesehatan pasien. Sistem ini dapat membantu dokter dan penyedia layanan kesehatan dalam membuat keputusan lebih cepat dan lebih akurat.

Mengapa Masalah Ini Harus Diselesaikan?

- Tingginya angka kematian akibat penyakit jantung menciptakan beban besar pada sistem kesehatan global.
- Dengan memanfaatkan Machine Learning, pasien yang memiliki risiko tinggi dapat diidentifikasi lebih awal, memungkinkan pengobatan preventif atau perubahan gaya hidup sebelum komplikasi serius terjadi.
- Diagnosis dini dapat membantu mengurangi biaya perawatan jangka panjang yang terkait dengan pengobatan penyakit jantung stadium lanjut.

## Business Understanding

### Problem Statements
Pernyataan masalah berikut dirumuskan untuk memberikan klarifikasi dan fokus pada permasalahan yang ingin diselesaikan dalam proyek ini:

- Insight apa saja yang dapat dipelajari dari dataset?
- Algoritme-Algoritme apa yang sesuai untuk memprediksi seseorang mengalami penyakit jantung?

### Goals

Proyek ini memiliki tujuan sebagai berikut:
- Melakukan analisis yang komprehensif dari dataset penyakit jantung untuk dipelajari pola, tren dan pengetahuan dari data
- Mengembangkan model pembelajaran mesin untuk melakukan tugas prediksi

### Solution Statements
Proyek ini mengatasi masalah dengan mengembangkan model machine learning untuk mendeteksi penyakit jantung menggunakan data pasien. Dengan memanfaatkan algoritma machine learning tradisional  dan Neural Network (deep learning)

## Data Understanding
### Jumlah Baris dan Kolom
```
: Dataset Info :.
******************************
Total Rows: 1025
Total Columns: 14
******************************
.: Dataset Details :.
******************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1025 entries, 0 to 1024
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1025 non-null   int64  
 1   sex       1025 non-null   int64  
 2   cp        1025 non-null   int64  
 3   trestbps  1025 non-null   int64  
 4   chol      1025 non-null   int64  
 5   fbs       1025 non-null   int64  
 6   restecg   1025 non-null   int64  
 7   thalach   1025 non-null   int64  
 8   exang     1025 non-null   int64  
 9   oldpeak   1025 non-null   float64
 10  slope     1025 non-null   int64  
 11  ca        1025 non-null   int64  
 12  thal      1025 non-null   int64  
 13  target    1025 non-null   int64  
dtypes: float64(1), int64(13)
```

Jumlah Dataset ini terdiri dari 
**14 variabel/ Kolom** 

**1025 Baris**, yang terbagi menjadi:

- **9 variabel kategorikal**, dan
- **5 variabel kontinu**.


### Kondisi Data
#### Missing values
```
	    
age	          0
sex	          0
cp	          0
trestbps	  0
chol	      0
fbs	          0
restecg	      0
thalach	      0
exang	      0
oldpeak	      0
slope	      0
ca	          0
thal          0
target	      0
```
Dataset tidak memiliki nilai yang hilang / missing value, sehingga tidak perlu perlakuan khsuus dalam menangani hal tersebut

#### Data Duplicate
```python
print("Jumlah Data Duplikat")
data.duplicated().sum()
# Persentase Duplikasi Data
duplication_percentage = (data.duplicated().sum() / len(data)) * 100
print(f"Persentase Duplikasi Data : {duplication_percentage:.2f}%")
```
output :
```
Jumlah Data Duplikat
723
Persentase Duplikasi Data : 70.54%
```
Karena data memiliki rate duplicate data >70% maka data tetap dipertahankan untuk memperhatikan model performance. Selain itu dalam kasus pengumpulan data medis, duplikasi data bisa saja terjadi karena karakteristik tubuh pasien satu dengan yang lainnya memiliki kemungkinan besar sama / mirip

### Tautan Sumber Data
Dataset diperoleh pada tautan 
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Uraian Seluruh Fitur data
Uraian seluruh fitur pada dataset dapat dilihat pada tabel berikut

| **Nama Variabel** | **Deskripsi** | **Contoh Data** |
|--------------------|---------------|------------------|
| **Age**           | Usia pasien (dalam tahun) | 63; 37; ... |
| **Sex**           | Jenis kelamin pasien (0 = laki-laki; 1 = perempuan) | 1; 0; ... |
| **cp**            | Jenis nyeri dada (4 nilai: 0, 1, 2, 3) | 3; 1; 2; ... |
| **trestbps**      | Tekanan darah saat istirahat (dalam mm Hg) | 145; 130; ... |
| **chol**          | Kolesterol serum (dalam mg/dl) | 233; 250; ... |
| **fbs**           | Gula darah puasa > 120 mg/dl (1 = benar; 0 = salah) | 1; 0; ... |
| **restecg**       | Hasil elektrokardiografi saat istirahat (nilai: 0, 1, 2) | 0; 1; ... |
| **thalach**       | Detak jantung maksimum yang dicapai | 150; 187; ... |
| **exang**         | Angina akibat olahraga (1 = ya; 0 = tidak) | 1; 0; ... |
| **oldpeak**       | Depresi ST akibat olahraga dibandingkan saat istirahat | 2.3; 3.5; ... |
| **slope**         | Kemiringan segmen ST pada puncak olahraga (nilai: 0, 1, 2) | 0; 2; ... |
| **ca**            | Jumlah pembuluh darah utama (0-4) yang terlihat melalui fluoroskopi | 0; 3; ... |
| **thal**          | (3 = normal; 6 = cacat tetap; 7 = cacat reversibel) | 3; 7; ... |
| **Target**        | Kolom target (1 = Ya; 0 = Tidak) | 1; 0; ... |

## Exploratory Data  Analysis
### Univariate Analysis
a. Age

Distribusi kolom age menunjukkan bahwa data usia pasien cenderung terdistribusi normal. Hal ini terlihat dari bentuk histogram dan boxplot yang simetris, serta nilai skewness yang mendekati 0 (-0.2)

![Age_U](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_Age.png)
Meskipun mendekati normal, distribusinya sedikit platikurtik, yang berarti datanya lebih datar dan memiliki ekor yang lebih tipis dibandingkan dengan distribusi normal, ditunjukkan dengan nilai kurtosis -0.5. Q-Q plot juga menguatkan indikasi distribusi normal karena data cenderung mengikuti garis 45 derajat. Secara keseluruhan, data usia pasien terdistribusi secara merata di sekitar nilai rata-rata, dengan sedikit variasi, menunjukkan sebagian besar pasien berada di rentang usia tengah. Informasi ini penting untuk memahami karakteristik pasien dan bagaimana usia berkaitan dengan penyakit jantung.

b. Sex

Distribusi kolom Sex menunjukkan bahwa jumlah pasien perempuan dalam dataset lebih banyak dibandingkan dengan pasien laki-laki. Hal ini terlihat jelas pada pie chart dan histogram. Pie chart menunjukkan proporsi pasien perempuan sekitar 68% sedangkan pasien laki-laki sekitar 32%. Histogram juga memvisualisasikan hal yang sama, dengan jumlah pasien perempuan yang lebih tinggi dibandingkan laki-laki. Perbedaan jumlah ini mengindikasikan bahwa dataset mungkin memiliki bias atau mewakili populasi di mana perempuan lebih rentan atau lebih banyak terdiagnosis dengan penyakit jantung. Informasi ini penting untuk dipertimbangkan dalam analisis lebih lanjut, terutama saat membangun model prediktif, untuk memastikan model tidak bias terhadap salah satu jenis kelamin.

![Sex_U](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_Sex.png)

c. Chest Pain (CP)

Distribusi kolom Chest Pain (cp) menunjukkan bahwa tipe nyeri dada 0 adalah yang paling umum di antara pasien dalam dataset. Tipe nyeri dada ini diikuti oleh tipe 2, tipe 1, dan tipe 3 dengan jumlah pasien yang semakin sedikit. Visualisasi menggunakan pie chart dan histogram dengan jelas menggambarkan hal ini. Pie chart memperlihatkan proporsi setiap tipe nyeri dada, sedangkan histogram menunjukkan jumlah pasien untuk setiap tipe. Tipe nyeri dada 0 memiliki proporsi sekitar 47%, menjadikannya tipe yang paling dominan. Informasi ini penting untuk memahami jenis nyeri dada yang paling sering dialami pasien dan dapat menjadi faktor penting dalam mendiagnosis dan mengobati penyakit jantung.

![cp](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_CP.png)

d. trestbps

Distribusi kolom trestbps memberikan gambaran tentang tekanan darah saat istirahat pasien. Dari histogram, terlihat bahwa distribusi trestbps sedikit condong ke kanan (right-skewed), mengindikasikan adanya beberapa pasien dengan tekanan darah istirahat yang relatif tinggi. 

![trestbps](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_TrestBPS.png)

e. Cholestrol

Distribusi kolom chol (serum cholestoral) menunjukkan bahwa data kolesterol cenderung terdistribusi normal, tetapi dengan sedikit kemiringan ke kanan (right-skewed). Hal ini terlihat dari histogram yang menunjukkan sebagian besar data terpusat di sekitar nilai rata-rata, tetapi ada beberapa data dengan nilai kolesterol yang lebih tinggi, menyebabkan ekor distribusi memanjang ke kanan. Nilai skewness 1.129, yang berada di atas 1, menguatkan indikasi right-skewness yang cukup signifikan. 

![chol](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_Chol.png)

f. fbs 

Variabel fbs menunjukkan kadar gula darah puasa pasien. Pada dataset ini, mayoritas pasien memiliki kadar gula darah puasa di bawah 120 mg/dl, yang diindikasikan dengan nilai fbs 0. Meskipun lebih sedikit, terdapat juga pasien dengan kadar gula darah puasa di atas 120 mg/dl (nilai fbs 1).

Insight yang menarik adalah, pasien dengan kadar gula darah puasa rendah (fbs=0) cenderung memiliki penyakit jantung. Selain itu, distribusi penyakit jantung pada pasien dengan kadar gula darah puasa tinggi (fbs=1) cenderung seimbang. Hal ini menunjukkan bahwa kadar gula darah puasa dapat menjadi faktor risiko penyakit jantung, namun perlu dikaji lebih lanjut dengan variabel lain untuk melihat hubungan yang lebih kompleks.

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_FBS.png)

g. Thalach

Variabel thalach dalam dataset ini merepresentasikan detak jantung maksimum yang dicapai oleh pasien. Secara umum, distribusi thalach mendekati normal, meskipun sedikit miring ke kiri dan lebih datar dibandingkan distribusi normal ideal. Hal ini ditunjukkan oleh nilai skewness -0.4 dan kurtosis -0.5. Meskipun demikian, sebagian besar pasien memiliki detak jantung maksimum di sekitar nilai rata-rata.

Terdapat insight menarik terkait hubungan thalach dengan penyakit jantung. Pasien dengan penyakit jantung, terutama pada rentang usia 50-70 tahun, cenderung memiliki detak jantung maksimum yang lebih tinggi dibandingkan dengan mereka yang tidak memiliki penyakit jantung. Hal ini menunjukkan bahwa detak jantung maksimum dapat menjadi indikator penting dalam mendeteksi potensi masalah jantung. Secara keseluruhan, thalach merupakan variabel penting dalam dataset ini karena mencerminkan kemampuan jantung pasien. Dengan menganalisisnya bersama variabel lain seperti usia dan riwayat penyakit, kita dapat memperoleh pemahaman yang lebih komprehensif tentang kondisi kesehatan jantung pasien dan risiko penyakit jantung.

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_Thalach.png)

h. exang

Variabel exang dalam dataset ini menunjukkan apakah pasien mengalami angina yang dipicu oleh aktivitas fisik atau tidak. Angina sendiri merupakan nyeri dada yang muncul akibat kurangnya aliran darah ke jantung. Pada dataset ini, mayoritas pasien (sekitar 67%) tidak mengalami angina saat beraktivitas fisik (exang = 0). Namun, terdapat sekitar 33% pasien yang melaporkan mengalami angina saat beraktivitas fisik (exang = 1).

Angina yang dipicu oleh aktivitas fisik dapat menjadi indikasi adanya masalah pada jantung, seperti penyempitan pembuluh darah koroner. Oleh karena itu, variabel exang merupakan informasi penting dalam mendiagnosis dan memprediksi penyakit jantung. Dengan menggabungkan informasi exang dengan variabel lain, seperti usia, riwayat penyakit, dan hasil pemeriksaan lainnya, kita dapat memperoleh gambaran yang lebih komprehensif mengenai kondisi kesehatan jantung pasien.

Meskipun variabel exang hanya memiliki dua kategori, informasi yang diberikannya sangat berharga dalam konteks analisis penyakit jantung. Variabel ini dapat membantu mengidentifikasi pasien yang berisiko lebih tinggi dan memerlukan penanganan lebih lanjut.

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Exang.png)

i. Oldpeak

Variabel oldpeak dalam dataset ini merepresentasikan perubahan pada elektrokardiogram (ECG) pasien selama berolahraga, yang dikenal sebagai depresi segmen ST. Perubahan ini dapat mengindikasikan adanya masalah pada aliran darah ke jantung.

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Univariate_Oldpeak.png)

### Multivariate Analysis

a. Heart Disease Scatter Based on Age

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Multivariate_Heart%20Rate_Age.png)

Scatter plot ini menampilkan hubungan antara usia (age), detak jantung maksimum (thalach), dan status penyakit jantung (target). Berikut insight yang dapat diperoleh:

- Sebagian besar data terkonsentrasi pada rentang usia 50-70 tahun, baik untuk pasien dengan penyakit jantung maupun yang tidak. Hal ini menunjukkan bahwa kelompok usia tersebut merupakan kelompok yang paling banyak terwakili dalam dataset.
- Terdapat kecenderungan detak jantung maksimum menurun seiring bertambahnya usia, baik untuk pasien dengan penyakit jantung maupun yang tidak. Hal ini sesuai dengan pengetahuan medis bahwa detak jantung maksimum cenderung menurun secara alami seiring pertambahan usia.
- Pasien dengan penyakit jantung (target = 1) cenderung memiliki detak jantung maksimum yang lebih tinggi dibandingkan dengan pasien tanpa penyakit jantung (target = 0) pada rentang usia yang sama. Perbedaan ini terlihat lebih jelas pada rentang usia 50-70 tahun.
- Scatter plot ini mengindikasikan bahwa kombinasi usia dan detak jantung maksimum dapat menjadi faktor penting dalam menilai risiko penyakit jantung. Pasien dengan usia lebih tua dan detak jantung maksimum yang tinggi perlu diwaspadai karena berpotensi memiliki risiko penyakit jantung yang lebih tinggi.

b. Heart Disease Distribution based on Fasting Blood Sugar

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Multivariate_Fasting%20Blood%20Sugar.png)

Visualisasi ini berupa stacked bar chart yang menunjukkan distribusi penyakit jantung (target) berdasarkan kadar gula darah puasa (fbs). Berikut insight yang dapat diperoleh:

- Mayoritas Gula Darah Normal: Sebagian besar pasien dalam dataset memiliki kadar gula darah puasa di bawah 120 mg/dl (fbs = 0). Hal ini ditunjukkan oleh bar berwarna biru yang lebih tinggi pada kategori "< 120 mg/dl".

- Kecenderungan Penyakit Jantung pada Gula Darah Normal: Meskipun jumlah pasien dengan gula darah puasa normal lebih banyak, terdapat kecenderungan bahwa pasien dengan gula darah puasa normal justru lebih banyak yang menderita penyakit jantung (target = 1). Hal ini ditunjukkan oleh bagian bar berwarna oranye yang lebih tinggi pada kategori "< 120 mg/dl".

c. Correlation Matrix

![](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Multivariate_Correlation_Analysis.png)

Heatmap ini memvisualisasikan matriks korelasi antara variabel-variabel numerik dalam dataset. Berikut insight yang dapat diperoleh:

- Korelasi Positif: Warna merah pada heatmap menunjukkan korelasi positif antara dua variabel. Semakin merah warnanya, semakin kuat korelasi positifnya. Contohnya, terdapat korelasi positif yang cukup kuat antara age (usia) dan trestbps (tekanan darah saat istirahat), chol (kolesterol), dan oldpeak (depresi segmen ST).

- Korelasi Negatif: Warna biru pada heatmap menunjukkan korelasi negatif antara dua variabel. Semakin biru warnanya, semakin kuat korelasi negatifnya. Dalam heatmap ini, tidak terlihat korelasi negatif yang kuat antar variabel.

- Korelasi Lemah: Warna putih atau mendekati putih pada heatmap menunjukkan korelasi yang lemah atau tidak ada korelasi antara dua variabel.

- Hubungan Antar Variabel: Heatmap ini memberikan gambaran umum tentang hubungan antar variabel dalam dataset. Sebagai contoh, kita dapat melihat bahwa usia (age) cenderung memiliki korelasi positif dengan tekanan darah saat istirahat (trestbps), kolesterol (chol), dan depresi segmen ST (oldpeak). Hal ini menunjukkan bahwa faktor-faktor tersebut mungkin saling terkait dan perlu dipertimbangkan dalam analisis lebih lanjut.


## Data Preparation
### Feature Engineering
Proses Feature Engineering adalah sebagai berikut
- Identifikasi Variabel
Data dibagi menjadi dua tipe utama berdasarkan jenisnya:
```
# Sesuaikan Data Type
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=['int', 'float']).columns.tolist()

print("Variabel Kategorik :", categorical_cols)

print("Variables Numerik :", numerical_cols)

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
```
Variabel Kategorik: Kolom yang bertipe object (biasanya teks), yang disimpan dalam daftar categorical_cols. Variabel Numerik: Kolom bertipe int atau float, yang disimpan dalam daftar numerical_cols.

- Tranformasi Variabel Kategorik
Salinan dataset asli dibuat menggunakan `data.copy()` untuk menghindari perubahan pada data awal. Variabel kategorik kemudian diubah menjadi format numerik menggunakan `LabelEncoder`, yang mengonversi setiap nilai kategorik menjadi angka.

### Splitting Data
Proses Splitting Data Sebagai Berikut

**Pemisahan Fitur dan Target**  
   Dataset dibagi menjadi:
   - **`X` (Fitur):** Semua kolom kecuali kolom target (`target`).
   - **`y` (Target):** Kolom `target` yang berisi nilai yang ingin diprediksi.
```
X = df_processed.drop('target', axis=1)
y = df_processed['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(X_train.shape)
print(X_test.shape)
```
output :
```
(820, 13)
(205, 13)
```

**Pembagian Data**  
   Dataset dibagi menjadi data pelatihan dan data pengujian menggunakan **`train_test_split`** dari scikit-learn, dengan rasio 80:20
   - **Data Pelatihan:** `X_train` dan `y_train`.
   - **Data Pengujian:** `X_test` dan `y_test`.  
   Parameter `random_state=42` memastikan hasil pembagian data konsisten setiap kali kode dijalankan.

### Standarization

```
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Standarisasi memastikan bahwa semua fitur memiliki skala yang sama, dengan nilai rata-rata 0 dan standar deviasi 1, yang penting untuk algoritma yang sensitif terhadap skala fitur, seperti regresi logistik atau SVM. Fungsi `StandardScaler()` dari library **scikit-learn** digunakan untuk menghitung rata-rata dan standar deviasi berdasarkan data pelatihan (`X_train`). Selanjutnya, data pelatihan distandarisasi menggunakan metode `fit_transform()`, sedangkan data uji distandarisasi menggunakan metode `transform()` saja agar tidak "mengintip" data uji selama pelatihan. Hal ini memastikan standarisasi data konsisten di seluruh proses.

## Model Development

### Traditional Machine learning
### 1. **Stacking Classifier Base Models dan Meta-Model**
Eksperimen pertama adalah implementasi **Stacking Classifier** menggunakan beberapa algoritma ML *base model* dan model regresi logistik sebagai *meta-model*. Tujuan dari metode ini adalah menggabungkan kekuatan dari setiap model untuk meningkatkan akurasi prediksi secara keseluruhan.
- Melakukan standarisasi data menggunakan **StandardScaler**.
- Menerapkan beberapa **base model**, antara lain:
  - Random Forest
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Extra Trees Classifier
  - Decision Tree
- Menggabungkan prediksi dari *base models* menggunakan **meta-model Logistic Regression** dengan `StackingClassifier`.
- Mengevaluasi performa model menggunakan:
  - Skor Akurasi
  - Laporan Klasifikasi (*Classification Report*)
  - Skor Validasi Silang (*Cross-Validation Scores*)


- Cara kerja dan parameter dari setiap algoritma yang digunakan :

  **RandomForestClassifier**
  
  Cara Kerja:
  Model berbasis *ensemble* yang membangun beberapa *decision tree* pada subset data, kemudian menggabungkan hasilnya melalui rata-rata (untuk regresi) atau voting mayoritas (untuk klasifikasi).

   Parameter:
    - `n_estimators=200`: Jumlah pohon dalam hutan.
    - `max_depth=10`: Kedalaman maksimum setiap pohon.
    - `min_samples_split=5`: Minimum jumlah sampel untuk memisah simpul.
    - `min_samples_leaf=2`: Minimum sampel di setiap daun pohon.
    - `random_state=42`: Seed untuk hasil yang konsisten.

  **SVC (Support Vector Classifier)**

  Cara Kerja:

    Mencari hyperplane yang memisahkan data dengan margin maksimum. Kernel RBF digunakan untuk menangani data non-linear.
  - Parameter:
    - `probability=True`: Mengaktifkan prediksi probabilitas.
    - `kernel='rbf'`: Menggunakan kernel Radial Basis Function (RBF).
    - `C=1.0`: Parameter regulasi; mengontrol keseimbangan antara margin maksimum dan kesalahan klasifikasi.
    - `gamma='scale'`: Otomatis mengatur gamma berdasarkan data input.
    - `random_state=42`: Seed untuk hasil konsisten.


  **KNeighborsClassifier**
  
  Cara Kerja:
  
  Algoritma *instance-based learning* yang mengklasifikasikan berdasarkan mayoritas label dari tetangga terdekat.
  - Parameter:
    - `n_neighbors=5`: Jumlah tetangga terdekat yang dipertimbangkan.
    - `algorithm='auto'`: Memilih algoritma terbaik untuk mencari tetangga.
    - `p=2`: Menggunakan jarak Euclidean sebagai metrik.

  **LogisticRegression**
  
  Cara Kerja:

  Model probabilistik yang menggunakan fungsi sigmoid untuk memetakan data ke dalam probabilitas kelas.
  - **Parameter:**
    - `max_iter=2000`: Batas maksimum iterasi untuk konvergensi.
    - `C=0.5`: Parameter regulasi, mengontrol kekuatan penalti terhadap model yang kompleks.
    - `solver='lbfgs'`: Algoritma optimisasi untuk estimasi parameter.
    - `random_state=42`: Seed untuk hasil konsisten.

  **ExtraTreesClassifier**
  
  Cara Kerja:
  
  Model berbasis ensemble yang serupa dengan Random Forest tetapi menggunakan pembagian acak dalam pohon untuk mengurangi overfitting.

  - Parameter:
    - `n_estimators=200`: Jumlah pohon dalam ensemble.
    - `max_depth=10`: Kedalaman maksimum pohon.
    - `min_samples_split=5`: Minimum sampel untuk memisahkan simpul.
    - `min_samples_leaf=2`: Minimum sampel di setiap daun pohon.
    - `random_state=42`: Seed untuk hasil konsisten.

  **DecisionTreeClassifier**
  
  Cara Kerja:

    Algoritma pohon keputusan yang memisahkan data berdasarkan aturan if-else pada setiap simpul.
  - **Parameter:**
    - `max_depth=10`: Kedalaman maksimum pohon.
    - `min_samples_split=10`: Minimum sampel untuk memisahkan simpul.
    - `min_samples_leaf=2`: Minimum sampel di setiap daun pohon.
    - `random_state=42`: Seed untuk hasil konsisten.

  **StackingClassifier**
  
  Cara Kerja:
  
  Kombinasi beberapa model dasar (*base models*) menggunakan meta-model Logistic Regression untuk membuat prediksi yang lebih baik.
  - Parameter:
    - `estimators=base_models`: Daftar model dasar.
    - `final_estimator=meta_model`: Model meta untuk menggabungkan hasil dari model dasar.
    - `cv=5`: Menggunakan 5-fold cross-validation untuk melatih meta-model.

  ---

**Evaluasi Model**
- **Akurasi Model Stacking:**  
  Model Stacking diuji pada data pengujian dan menghasilkan metrik evaluasi seperti *accuracy* dan *classification report*.
- **Cross-Validation:**  
  Dilakukan untuk setiap model dasar dan model stacking untuk mengukur kinerja rata-rata (*mean accuracy*) pada data pelatihan.


### Deep Learning
- <<<<< Skenario 1 >>>>>

  **1. Arsitektur Model**
    Model menggunakan **Sequential API** dengan layer berikut:
    - **Input Layer:**  
      - `Dense(1024)`: Layer pertama dengan 1024 unit neuron dan fungsi aktivasi `relu`.  
      - `kernel_regularizer=regularizers.l2(0.5)`: Menambahkan regularisasi L2 untuk mencegah overfitting. Regulasi L2 memberikan penalti pada bobot besar.
      - `input_shape=(X_train.shape[1],)`: Menentukan jumlah fitur pada data input.
      - `Dropout(0.3)`: Mencegah overfitting dengan mengabaikan 30% neuron selama pelatihan.

    - **Hidden Layers:**  
      - `Dense(128)`: Layer dengan 128 neuron dan fungsi aktivasi `relu`.  
      - `kernel_regularizer=regularizers.l2(0.4)`: Regularisasi L2 diterapkan pada layer ini.  
      - `Dropout(0.2)`: Mengabaikan 20% neuron selama pelatihan.
      - `Dense(32)`: Layer tambahan dengan 32 neuron dan fungsi aktivasi `relu`.

    - **Output Layer:**  
      - `Dense(1)`: Layer output dengan 1 neuron dan fungsi aktivasi `sigmoid` untuk menghasilkan probabilitas kelas.

    **2. Kompilasi Model**
    Model dikompilasi menggunakan:
    - **Optimizer:** `adam`  
      - Algoritma optimasi adaptif yang efisien untuk data besar.
    - **Loss Function:** `binary_crossentropy`  
      - Digunakan karena ini adalah masalah klasifikasi biner.
    - **Metrics:** `accuracy`  
      - Untuk melacak akurasi selama pelatihan.
    **3. Pelatihan Model**
    Model dilatih menggunakan:
    - **`X_train` dan `y_train`** sebagai data pelatihan.  
    - **`validation_data=(X_test, y_test)`** untuk melacak kinerja model pada data validasi.
    - **Epochs:** 100 iterasi penuh melalui data pelatihan.
    - **Batch Size:** 16 sampel per batch.  
    - **Verbose:** 1 untuk mencetak log selama pelatihan.

    ---
  
-  <<<<<<< Skenario 2 >>>>>>>
    
      Penjelasan Cara Kerja dan Parameter algoritma pada Skenario 2

    **1. Dense Layer (Fully Connected Layer)**  

    Cara Kerja:  
    - Layer ini menghubungkan semua neuron di layer sebelumnya dengan semua neuron di layer ini.
    - Setiap koneksi memiliki bobot yang diperbarui selama pelatihan untuk meminimalkan error.

    Parameter Utama:
    - `units`: Jumlah neuron dalam layer.
    - `activation`: Fungsi aktivasi untuk memperkenalkan non-linearitas, seperti `relu`, `sigmoid`, atau `softmax`.
    - `input_shape`: Dimensi input untuk layer pertama.
    - `kernel_regularizer`: Penambahan regulasi L1/L2 untuk mengurangi overfitting.

    **2. Dropout Layer**  

    Cara Kerja:  
    - Secara acak "mematikan" (drop) beberapa neuron selama pelatihan untuk mencegah overfitting dengan membuat model lebih general.

    Parameter Utama:
    - `rate`: Proporsi neuron yang di-drop (contoh: `0.8` berarti 80% neuron dimatikan).

    **3. Adam Optimizer**  
    Cara Kerja:  
    - Optimizer berbasis gradient descent yang menggabungkan dua metode:
      - **Momentum**: Mempercepat konvergensi.
      - **Adaptive Learning Rate (AdaGrad)**: Menyesuaikan learning rate secara dinamis.
    - Adam sangat populer karena robust terhadap parameter learning rate dan bekerja baik pada data noisy.

    Parameter Utama:
    - `learning_rate`: Kecepatan pembelajaran.
    - `beta_1`: Parameter momentum pertama (default `0.9`).
    - `beta_2`: Parameter momentum kedua untuk penyesuaian learning rate (default `0.999`).


    **4. L2 Regularization (Kernel Regularizer)**  
    Cara Kerja:  
    - Menambahkan penalti terhadap bobot besar dalam model ke fungsi loss untuk mengurangi kompleksitas model dan mencegah overfitting.

    Parameter Utama:
    - `l2`: Parameter regulasi; semakin besar nilainya, semakin besar penalti.

    **5. Binary Crossentropy Loss**  
    Cara Kerja:  
    - Digunakan untuk masalah klasifikasi biner. Mengukur perbedaan antara probabilitas prediksi (`sigmoid output`) dan label sebenarnya.

    **6. Sigmoid Activation**  
    Cara Kerja:  
    - Mengubah output menjadi probabilitas dalam rentang \(0\) hingga \(1\).
---

-  <<<<<<< Skenario 3 >>>>>>>

    Penjelasan Cara Kerja dan Parameter algoritma pada Skenario 3

    **Early Stopping Callback**

    Cara Kerja:
    - **EarlyStopping** digunakan untuk menghentikan pelatihan model lebih awal jika tidak ada perbaikan pada fungsi loss dalam jumlah epoch tertentu. Ini membantu mencegah overfitting.
    - `patience=15`: Berarti pelatihan akan dihentikan jika tidak ada perbaikan pada loss dalam 15 epoch berturut-turut.
    - `monitor='loss'`: Memantau nilai loss untuk menghentikan pelatihan.
    - `mode='min'`: Menghentikan jika loss tidak lagi berkurang.
    **Arsitektur Model:**
    - **Layer Dense**: Setiap layer dihubungkan sepenuhnya dengan neuron-neuron pada layer sebelumnya.
    - **Dropout**: Digunakan untuk mencegah overfitting dengan "mematikan" beberapa neuron selama pelatihan.
    - **Regularisasi L2**: Menambahkan penalti terhadap bobot besar untuk mengurangi overfitting.
    - **Input Layer**: Menggunakan 13 fitur input dengan 64 neuron.
    - **Hidden Layers**: Terdapat tiga hidden layers dengan jumlah neuron yang berkurang (64 → 32 → 16 → 8). Setiap layer menggunakan fungsi aktivasi `ReLU` dan diikuti dengan layer `Dropout` untuk mencegah overfitting.
    - **Output Layer**: Menggunakan fungsi aktivasi `sigmoid` untuk klasifikasi biner (output 0 atau 1).

    **Compiler dan Optimizer**
    - Model dikompilasi dengan menggunakan **Adam optimizer** yang adaptif, yang mengatur learning rate secara otomatis.
    - **Loss Function**: Digunakan `binary_crossentropy` karena ini adalah masalah klasifikasi biner.
    - **Metrics**: Menggunakan akurasi untuk memantau performa model selama pelatihan.

---

-  <<<<<<< Skenario 4 >>>>>>>
  
    Penjelasan Cara Kerja dan Parameter algoritma pada Skenario 4

    **Early Stopping Callback**

    Cara Kerja:
    - **EarlyStopping** digunakan untuk menghentikan pelatihan model lebih awal jika tidak ada perbaikan pada fungsi loss selama sejumlah epoch tertentu.
    - **Patience** mengontrol berapa banyak epoch yang akan dilalui sebelum menghentikan pelatihan jika tidak ada perbaikan.

    - `monitor='loss'`: Memantau perubahan pada loss untuk menghentikan pelatihan.
    - `mode='min'`: Menghentikan pelatihan jika nilai loss tidak berkurang.

    **Arsitektur Model**

    Penjelasan Layer:
    - **Input Layer**: Menggunakan 13 fitur input dengan 64 neuron.
    - **Hidden Layers**: Terdapat beberapa layer tersembunyi dengan jumlah neuron yang berkurang (64 → 32 → 16 → 8), masing-masing menggunakan fungsi aktivasi `ReLU` dan diikuti dengan layer `Dropout` untuk mencegah overfitting.
    - **Output Layer**: Layer output dengan fungsi aktivasi `sigmoid` untuk menghasilkan output 0 atau 1 (klasifikasi biner).

    **Compiler dan Optimizer**

    Cara Kerja:
    - **Adam Optimizer**: Digunakan untuk optimasi model dengan learning rate 0.001.
    - **Loss Function**: Menggunakan `binary_crossentropy` karena ini adalah masalah klasifikasi biner.
    - **Metrics**: Menggunakan akurasi untuk memantau performa model selama pelatihan.
  ---

## Evaluation

### Insight apa saja yang dapat dipelajari dari dataset?
Berikut adalah penjelasan terkait dampak hasil 
**Exploratory Data Analysis (EDA)** terhadap **Business Understanding** dalam proyek prediksi penyakit jantung:

a. Peningkatan Pemahaman Data dan Populasi Target.

**EDA memberikan wawasan tentang distribusi data, tren, dan pola** seperti:
  - Sebaran usia pasien (mayoritas pada usia 50-70 tahun) mengarahkan perhatian pada populasi yang lebih berisiko.
  - Hubungan antara detak jantung maksimum (*thalach*) dan risiko penyakit jantung membantu identifikasi faktor risiko kunci.
  - Distribusi fitur seperti tekanan darah istirahat (*trestbps*) dan kolesterol menunjukkan kebutuhan pengelompokan pasien berdasarkan risiko.

  **Dampak:** Dengan memahami pola ini, model prediksi dapat disesuaikan untuk mendeteksi pasien dengan profil risiko tinggi, sehingga membantu dokter atau penyedia layanan kesehatan memprioritaskan intervensi.

b. Identifikasi Faktor Risiko Utama
**Insight dari EDA mengungkap hubungan antara variabel dan target**, seperti:
  - Variabel **oldpeak** (depresi segmen ST) dan **thalach** (detak jantung maksimum) memiliki hubungan signifikan dengan penyakit jantung.
  - Pasien dengan kadar gula darah puasa tinggi (*fbs = 1*) tidak selalu lebih rentan terhadap penyakit jantung, menunjukkan bahwa faktor lain seperti usia atau tekanan darah memiliki pengaruh yang lebih besar.  

  **Dampak:**  
  - Informasi ini membantu mempersempit faktor risiko utama yang relevan untuk intervensi klinis. Dokter dapat lebih fokus pada metrik yang relevan, seperti tekanan darah atau tingkat aktivitas fisik, untuk menilai kondisi pasien secara lebih efektif.


c. Optimalisasi Sumber Daya Medis
- Analisis hubungan multivariat, seperti hubungan antara usia dan detak jantung maksimum terhadap penyakit jantung, menunjukkan kelompok risiko tinggi.  Sebagai contoh, pasien pada rentang usia 50-70 tahun dengan detak jantung maksimum tinggi lebih rentan terhadap penyakit jantung.  

  **Dampak:**  
  - Informasi ini memungkinkan rumah sakit atau institusi medis untuk mengalokasikan sumber daya, seperti skrining lanjutan atau layanan diagnostik, kepada kelompok usia yang lebih berisiko, mengurangi biaya operasional secara keseluruhan.


### Algoritme-Algoritme apa yang sesuai untuk memprediksi seseorang mengalami penyakit jantung?

Untuk memperoleh jawaban dari algoritme yang dikembangkan, berikut merupakan hasilnya

======================================================
#### Hasil Evaluasi Stacking Classifier Base Models dan Meta-Model
======================================================

Akurasi Model Stacking
  - **Stacked Model Accuracy**: **98.54%**
  - Model stacking menunjukkan performa yang sangat baik dengan akurasi mendekati 99%.

  Classification Report
  | Kelas | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | **0** | 0.97      | 1.00   | 0.99     | 102     |
  | **1** | 1.00      | 0.97   | 0.99     | 103     |
  | **Rata-rata Akurasi** | 0.99 | 0.99 | 0.99 | 205 |

  - Model ini memiliki **Precision**, **Recall**, dan **F1-Score** yang hampir sempurna untuk kedua kelas, menunjukkan kemampuan prediksi yang sangat baik.

  Cross-Validation Performa Base Models

  | Model                | CV Scores                           | Mean CV Accuracy |
  |----------------------|--------------------------------------|------------------|
  | **Random Forest (rf)** | [0.9878, 0.9451, 0.9878, 0.9695, 0.9207] | **0.9622** |
  | **Support Vector Classifier (svc)** | [0.9451, 0.9085, 0.9085, 0.9451, 0.8902] | **0.9195** |
  | **K-Nearest Neighbors (knn)** | [0.8902, 0.8475, 0.8658, 0.8414, 0.7683] | **0.8427** |
  | **Logistic Regression (log)** | [0.8353, 0.8536, 0.8902, 0.8658, 0.7926] | **0.8476** |
  | **Extra Trees (et)** | [1.0000, 0.9390, 0.9573, 0.9634, 0.9329] | **0.9585** |
  | **Decision Tree (dt)** | [0.9085, 0.9329, 0.9329, 0.9390, 0.8841] | **0.9195** |

  Cross-Validation Performa Model Stacking
  - **Stacked Model Cross-Validation Mean Accuracy**: **96.10%**

  Model stacking berhasil menggabungkan kekuatan base models untuk meningkatkan performa keseluruhan, mendekati performa base model terbaik (Random Forest dan Extra Trees) dengan rata-rata akurasi 96.10% pada validasi silang.

  - **Stacked Model** memberikan akurasi tertinggi di data uji sebesar **98.54%**.
  - Model **Random Forest** dan **Extra Trees** menunjukkan performa konsisten dengan **Mean CV Accuracy** masing-masing sebesar **96.22%** dan **95.85%**, mendukung kontribusinya dalam model stacking.
  - Pendekatan stacking dapat menjadi pilihan unggul dalam masalah klasifikasi dengan data penyakit jantung yang kompleks.

**Dampak dari Stacking Classifier Base Models dan Meta-Model yang dievaluasi terhadap Business Understanding**

- Dengan tingkat akurasi uji sebesar 98.54%, precision, recall, dan F1-score yang hampir sempurna, model ini menunjukkan kemampuan untuk memberikan prediksi yang sangat akurat. Hal ini penting dalam pengambilan keputusan bisnis, karena model yang andal dapat meminimalkan risiko kesalahan prediksi, misalnya false negative dalam kasus penyakit jantung yang dapat berdampak fatal.

- Rata-rata akurasi cross-validation sebesar 96.10% mencerminkan bahwa model memiliki generalisasi yang baik pada berbagai subset data. Ini berarti model tidak hanya unggul dalam pengujian tetapi juga dapat diandalkan dalam dunia nyata, seperti saat diterapkan pada populasi pasien yang berbeda.

- Dengan performa yang hampir sempurna, model ini dapat meningkatkan kepercayaan stakeholders (dokter, rumah sakit, atau perusahaan asuransi kesehatan) karena mampu memberikan analisis yang akurat dan konsisten untuk membantu diagnosis atau manajemen risiko.

- Efektif. Kemampuan stacking untuk memanfaatkan kombinasi hasil terbaik dari base models (seperti Random Forest dan Extra Trees) membuatnya lebih unggul dalam menangani data kompleks dengan pola non-linear. Ini sangat relevan dalam data kesehatan, yang seringkali memiliki banyak fitur yang saling memengaruhi.

- Dampak pada Efisiensi Operasional: Dengan tingkat akurasi tinggi, model ini memungkinkan penghematan biaya dan waktu dalam proses diagnostik atau skrining awal. Sebagai contoh, pasien yang diprediksi tidak memiliki risiko penyakit dapat dikelompokkan untuk skrining tambahan yang lebih murah, sementara pasien dengan risiko tinggi dapat diberikan perhatian lebih segera.


======================================================
#### Hasil Evaluasi Deep Learning Model di Berbagai Skenario
======================================================

Hasil Model pada Skenario 1

![Model Visualization](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Model_DL_1.png)

Metrik Training dan Validation

| Metrik                 | Pelatihan    | Validasi    |
|------------------------|--------------|-------------|
| **Akurasi**            | 0.6671 (66.71%) | 0.6683 (66.83%) |
| **Loss**               | 0.6485       | 0.6654      |

- Akurasi pelatihan dan validasi hampir sama, yang menunjukkan bahwa model dapat mengeneralisasi dengan baik pada data validasi dan tidak mengalami overfitting yang signifikan.  
- Loss validasi sedikit lebih tinggi dari loss pelatihan, tetapi perbedaan ini kecil dan tidak menunjukkan masalah yang besar.

Metrik Pengujian

| Metrik               | Pengujian   |
|----------------------|-------------|
| **Akurasi**          | 0.6792 (67.92%) |
| **Loss**             | 0.6603      |

- Akurasi pengujian sedikit lebih tinggi daripada akurasi pelatihan dan validasi, yang menunjukkan bahwa model sedikit lebih baik dalam menangani data yang belum terlihat.


| Aspek                       | Penjelasan                                                                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Tingkat Performa**        | Akurasi ~67% menunjukkan model cukup berhasil dalam klasifikasi, tetapi masih bisa dioptimalkan.                                          |
| **Pelatihan vs Validasi**   | Akurasi pelatihan dan validasi yang sangat mirip menunjukkan model tidak overfit atau underfit secara signifikan.                          |
| **Loss**                     | Nilai loss wajar, tetapi model belum sepenuhnya konvergen.             |

Hasil pada Scenario 2 :
![Model Result Scenario 2](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Model_DL_2.png)

Metrik Pelatihan dan Validasi

| Metrik                 | Pelatihan    | Validasi    |
|------------------------|--------------|-------------|
| **Akurasi**            | 0.4993 (49.93%) | 0.5024 (50.24%) |
| **Loss**               | 0.6941       | 0.6944      |

- Akurasi pelatihan dan validasi sangat rendah, hanya sekitar 50%. Ini menunjukkan bahwa model hampir tidak lebih baik daripada prediksi acak.
- Loss pelatihan dan validasi yang hampir sama juga mengindikasikan bahwa model belum berhasil mempelajari pola yang signifikan dalam data.

| Metrik               | Pengujian   |
|----------------------|-------------|
| **Akurasi**          | 0.5024 (50.24%) |
| **Loss**             | 0.6953      |

- Akurasi pengujian mirip dengan akurasi validasi, menunjukkan bahwa model tidak dapat menangani data uji dengan baik.
- Loss pengujian sedikit lebih tinggi daripada loss pelatihan, yang menunjukkan bahwa model tidak cukup terlatih untuk melakukan prediksi yang akurat pada data baru.

| Aspek                       | Penjelasan                                                                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Tingkat Performa**        | Akurasi yang sangat rendah (~50%) menunjukkan bahwa model tidak berhasil dalam klasifikasi dan hampir tidak lebih baik dari tebak acak.  |
| **Pelatihan vs Validasi**   | Akurasi pelatihan dan validasi yang sangat mirip menunjukkan bahwa model belum belajar cukup banyak dari data dan mungkin underfitting.    |
| **Loss**                     | Nilai loss yang tinggi menunjukkan bahwa model tidak mampu mempelajari pola yang tepat, dan perlu perbaikan dalam arsitektur atau hyperparameter. |

Hasil pada skenario 3
![Model Skenario3](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Model_DL_3_92_%25.png)

Metrik Pelatihan dan Validasi

| Metrik                 | Pelatihan    | Validasi    |
|------------------------|--------------|-------------|
| **Akurasi**            | 0.9613 (96.13%) | 0.9268 (92.68%) |
| **Loss**               | 0.1695       | 0.2487      |

- **Akurasi Pelatihan**: Model berhasil mencapai **akurasi 96,13%** pada data pelatihan, yang menunjukkan kemampuan model dalam mempelajari data pelatihan dengan baik.
- **Akurasi Validasi**: Pada data validasi, model memperoleh **akurasi 92,68%**, yang menunjukkan bahwa model dapat menggeneralisasi dengan baik pada data yang tidak terlihat.
- **Loss Pelatihan**: Loss pada data pelatihan adalah **16,95%**, yang menunjukkan bahwa model telah mempelajari pola dengan baik, tetapi ada sedikit kesalahan.
- **Loss Validasi**: Loss pada data validasi adalah **24,87%**, yang sedikit lebih tinggi dibandingkan pelatihan, menunjukkan bahwa model mungkin sedikit overfitting pada data pelatihan.

Metrik Pengujian

| Metrik               | Pengujian   |
|----------------------|-------------|
| **Akurasi**          | 0.9254 (92.54%) |
| **Loss**             | 0.2496      |

- **Akurasi Pengujian**: Model memperoleh **akurasi 92,54%** pada data uji, yang menunjukkan bahwa model memiliki kemampuan generalisasi yang baik pada data dunia nyata.
- **Loss Pengujian**: Loss pada data uji adalah **24,96%**, sedikit lebih tinggi dibandingkan dengan loss pelatihan dan validasi, yang mungkin menunjukkan sedikit overfitting.

Analisis Kinerja Model

| Aspek                       | Penjelasan                                                                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Tingkat Performa**        | Akurasi pelatihan yang tinggi (96,13%) menunjukkan bahwa model belajar dengan baik pada data pelatihan, namun ada penurunan akurasi pada validasi dan pengujian (92,68% dan 92,54%), yang menunjukkan adanya sedikit overfitting. |
| **Pelatihan vs Validasi**   | Perbedaan antara akurasi pelatihan dan validasi menunjukkan bahwa model dapat menggeneralisasi dengan baik, meskipun ada sedikit kesalahan pada data validasi dan uji. |
| **Loss**                     | Loss yang lebih tinggi pada validasi dan pengujian menunjukkan bahwa meskipun model memiliki akurasi yang baik, masih ada ruang untuk perbaikan dalam hal generalisasi dan mengurangi overfitting. |

Hasil pada Scenario 4 :

![model_skenario_4](https://raw.githubusercontent.com/farhanrn/heart-disease-detection/refs/heads/main/src/Model_DL_3_92_%25.png)

Metrik Pelatihan dan Validasi

| Metrik                 | Pelatihan    | Validasi    |
|------------------------|--------------|-------------|
| **Akurasi**            | 0.9884 (98.84%) | 0.9695 (96.95%) |
| **Loss**               | 0.1601       | 0.2527      |

- **Akurasi Pelatihan**: Model berhasil mencapai **akurasi 98,84%** pada data pelatihan, yang menunjukkan model dapat mempelajari data pelatihan dengan sangat baik.
- **Akurasi Validasi**: Pada data validasi, model memperoleh **akurasi 96,95%**, yang menunjukkan kemampuan model untuk menggeneralisasi pada data yang tidak terlihat dengan baik.
- **Loss Pelatihan**: Loss pada data pelatihan adalah **16,01%**, yang menunjukkan bahwa model sangat baik dalam mempelajari pola data pelatihan.
- **Loss Validasi**: Loss pada data validasi adalah **25,27%**, yang sedikit lebih tinggi dibandingkan dengan pelatihan, namun masih dalam batas yang dapat diterima.

Metrik Pengujian

| Metrik               | Pengujian   |
|----------------------|-------------|
| **Akurasi**          | 0.9523 (95.23%) |
| **Loss**             | 0.3939      |

- **Akurasi Pengujian**: Model memperoleh **akurasi 95,23%** pada data uji, yang menunjukkan bahwa model juga berhasil dalam menggeneralisasi data baru dengan baik.
- **Loss Pengujian**: Loss pada data uji adalah **39,39%**, sedikit lebih tinggi dibandingkan dengan loss pada pelatihan dan validasi, yang dapat menunjukkan sedikit overfitting.

Analisis Kinerja Model

| Aspek                       | Penjelasan                                                                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| **Tingkat Performa**        | Akurasi pelatihan yang sangat tinggi (98,84%) menunjukkan bahwa model berhasil belajar dengan sangat baik pada data pelatihan. Namun, ada penurunan akurasi pada validasi dan pengujian, yang menunjukkan kemungkinan overfitting. |
| **Pelatihan vs Validasi**   | Meskipun akurasi pelatihan sangat tinggi, akurasi validasi yang sedikit lebih rendah (96,95%) menunjukkan bahwa model sedikit overfit pada data pelatihan. Namun, performa validasi masih sangat baik. |
| **Loss**                     | Loss pada data uji yang lebih tinggi menunjukkan bahwa model bisa mengalami sedikit kesulitan dalam menghadapi data baru. Ini adalah indikasi adanya sedikit overfitting yang bisa diperbaiki. |


**Dampak dari Deep Learning yang dievaluasi terhadap Business Understanding**

Model deep learning pada Scenario 4 menunjukkan performa yang sangat baik berdasarkan metrik evaluasi, menjadikannya pilihan yang layak untuk diimplementasikan dalam aplikasi prediksi penyakit jantung. Model deep learning pada **Scenario 4** menunjukkan performa yang sangat baik, ditandai dengan akurasi pelatihan sebesar **98.84%**, akurasi validasi **96.95%**, dan akurasi pengujian **95.23%**. Tingkat akurasi yang tinggi ini membuktikan bahwa model mampu menghasilkan prediksi yang andal, bahkan saat digunakan pada data baru. Performa validasi dan pengujian yang tetap tinggi menunjukkan bahwa model memiliki kemampuan generalisasi yang baik pada data yang tidak terlihat. Selain itu, nilai loss pada validasi dan pengujian masih berada dalam batas wajar, yang menunjukkan bahwa model dapat mempelajari pola dengan baik tanpa kehilangan konsistensi. Dirancang untuk menangani data dengan pola non-linear yang kompleks, model ini sangat efektif untuk menganalisis dataset kesehatan yang terdiri dari berbagai parameter, seperti tekanan darah, kadar gula, dan riwayat penyakit, sehingga menjadikannya pilihan yang sangat baik untuk aplikasi prediksi penyakit jantung.

Sama halnya pada penerapan traditional machine learning, Model deep learning memiliki dampak pada business understanding. Model deep learning pada **Scenario 4** memiliki dampak signifikan terhadap pemahaman bisnis, terutama dalam sektor kesehatan. Dengan akurasi yang tinggi, model ini dapat membantu dokter dan tenaga medis mendeteksi penyakit jantung lebih cepat dan akurat, sehingga mengurangi waktu diagnosis manual dan berpotensi menyelamatkan nyawa pasien. Keberhasilannya dalam menghasilkan prediksi yang andal juga meningkatkan kepercayaan dari berbagai pihak, seperti dokter, rumah sakit, perusahaan asuransi, dan pasien terhadap sistem berbasis AI. Dari segi operasional, model ini mampu meningkatkan efisiensi dengan mengelompokkan pasien berdasarkan hasil prediksi. Pasien dengan risiko rendah dapat difokuskan pada perawatan preventif, sedangkan pasien berisiko tinggi dapat menerima intervensi medis yang lebih cepat, yang secara signifikan mengurangi potensi komplikasi. Selain itu, model ini dapat membantu mengurangi biaya skrining dengan memungkinkan alokasi sumber daya yang lebih terfokus pada kasus-kasus prioritas.  

Pada tingkat strategis, model ini mendukung upaya kesehatan masyarakat dengan memprediksi risiko penyakit jantung pada populasi besar dan menghasilkan informasi yang dapat digunakan untuk membuat kebijakan kesehatan, seperti program edukasi preventif atau distribusi sumber daya yang lebih efektif. Implementasi model deep learning ini juga dapat meningkatkan reputasi rumah sakit atau perusahaan kesehatan, mencerminkan inovasi dan keunggulan teknologi mereka. Terakhir, model ini membantu mitigasi risiko dengan mendeteksi potensi penyakit jantung lebih dini, sehingga menurunkan kemungkinan terjadinya kejadian buruk, seperti serangan jantung mendadak, terutama pada pasien dengan gejala yang tidak kentara.

### Referensi Terkait
- World Health Organization (WHO). *Cardiovascular Diseases (CVDs)*. Diakses dari: [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)].
- Kementerian Kesehatan RI (2022). *Infodatin Penyakit Jantung*. Diakses dari: [https://pusdatin.kemkes.go.id].
- Benjamin, E. J., et al. (2019). *Heart Disease and Stroke Statistics—2019 Update: A Report From the American Heart Association*. Circulation, 139(10), e56–e528. DOI:10.1161/CIR.0000000000000659.
- Kumar, A., et al. (2020). *Prediction of Cardiovascular Disease Using Machine Learning Algorithms: A Comparative Study*. Procedia Computer Science, 167, 1388–1398. DOI:10.1016/j.procs.2020.03.308.
- Dutta, S., et al. (2020). *Impact of AI in Cardiology: Artificial Intelligence in Cardiovascular Risk Prediction*. Current Cardiovascular Risk Reports, 14(4), 15. DOI:10.1007/s12170-020-00624-6.
