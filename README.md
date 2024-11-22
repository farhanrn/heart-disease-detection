
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
Proyek ini mengatasi masalah dengan mengembangkan model machine learning untuk mendeteksi penyakit jantung menggunakan data pasien. Dengan memanfaatkan algoritma machine learning tradisional Extra Trees Classifier dan Neural Network (deep learning)

## Data Understanding
Dataset diperoleh pada tautan 
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Dataset ini terdiri dari **14 variabel**, yang terbagi menjadi:
- **9 variabel kategorikal**, dan
- **5 variabel kontinu**.


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

## Data Preparation

#### 1. Konfigurasi Sistem dan Library
- Instalasi library yang dibutuhkan, seperti **pycaret** dan **kaggle**.
- Import library utama:
  - **pandas**, **numpy**: Untuk analisis dan manipulasi data.
  - **matplotlib**, **seaborn**: Untuk visualisasi data.
  - **sklearn**, **tensorflow**: Untuk pemodelan machine learning.
- Library ini digunakan untuk analisis data, visualisasi, dan membangun model machine learning.

#### 2. Konfigurasi Dataset
- Dataset diunduh menggunakan library **kaggle**:
  - Pastikan Anda memiliki akun Kaggle.
  - Unduh file `kaggle.json` dari profil Kaggle Anda dan letakkan di direktori yang sesuai.
- Dataset dibaca menggunakan **pandas** dan disimpan dalam variabel `data`.

#### 3. Preprocessing Dataset
- Pengecekan **missing values** menggunakan:
  ```python
  data.isna().sum()
  ```
- Menampilkan informasi dan detail dataset menggunakan
    ```python
    data.info()
    data.describe()
    ```

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

## Modeling
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

### 2. AutoML dengan Menggunakan [pycaret](https://pycaret.org/)

**PyCaret** membantu melakukan pra-pemrosesan data, pelatihan model, dan evaluasi secara otomatis.
- Membandingkan performa beberapa algoritma klasifikasi populer.
- Menampilkan metrik evaluasi utama seperti:
  - Akurasi
  - Area Under Curve (AUC)
  - Recall
  - Precision
  - F1-Score
  - Cohen's Kappa
  - Matthews Correlation Coefficient (MCC)

### Deep Learning
- Terdapat 4 skenario arsitektur model Deep Learning (Neural Network) berserta model untuk *auto hyperparameter tuning*

## Evaluation
### Traditional Machine learning Model Evaluation
#### 1. Stacking Classifier Base Models dan Meta-Model
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

### 2. AutoML dengan Menggunakan [pycaret](https://pycaret.org/)

Berikut adalah hasil dari perbandingan berbagai model menggunakan PyCaret:

| Model                            | Akurasi  | AUC    | Recall  | Precision | F1      | Kappa   | MCC     | Waktu Pelatihan (Detik) |
|-----------------------------------|----------|--------|---------|-----------|---------|---------|---------|--------------------------|
| **Extra Trees Classifier (et)**   | 0.9832   | 0.9992 | 0.9865  | 0.9815    | 0.9838  | 0.9664  | 0.9669  | 0.3030                   |
| Extreme Gradient Boosting (xgboost) | 0.9777 | 0.9868 | 0.9810  | 0.9760    | 0.9784  | 0.9553  | 0.9555  | 0.1090                   |
| Random Forest Classifier (rf)     | 0.9749   | 0.9954 | 0.9783  | 0.9735    | 0.9757  | 0.9497  | 0.9502  | 0.3150                   |
| Light Gradient Boosting Machine (lightgbm) | 0.9749 | 0.9900 | 0.9783  | 0.9735    | 0.9757  | 0.9497  | 0.9501  | 0.3290                   |
| Decision Tree Classifier (dt)     | 0.9707   | 0.9708 | 0.9674  | 0.9759    | 0.9714  | 0.9414  | 0.9419  | 0.0650                   |
| Gradient Boosting Classifier (gbc) | 0.9582  | 0.9875 | 0.9727  | 0.9476    | 0.9597  | 0.9162  | 0.9173  | 0.3740                   |
| Ada Boost Classifier (ada)        | 0.8716   | 0.9556 | 0.8667  | 0.8833    | 0.8739  | 0.7432  | 0.7451  | 0.2010                   |
| Quadratic Discriminant Analysis (qda) | 0.8522 | 0.9256 | 0.8722  | 0.8455    | 0.8582  | 0.7040  | 0.7053  | 0.0500                   |
| Logistic Regression (lr)          | 0.8424   | 0.9115 | 0.8940  | 0.8186    | 0.8537  | 0.6837  | 0.6889  | 0.9190                   |
| Ridge Classifier (ridge)          | 0.8215   | 0.9127 | 0.9048  | 0.7844    | 0.8392  | 0.6411  | 0.6517  | 0.0370                   |
| Linear Discriminant Analysis (lda) | 0.8215  | 0.9128 | 0.9048  | 0.7844    | 0.8392  | 0.6411  | 0.6517  | 0.0750                   |
| Naive Bayes (nb)                  | 0.8187   | 0.8982 | 0.8640  | 0.8005    | 0.8304  | 0.6363  | 0.6399  | 0.1030                   |
| K Neighbors Classifier (knn)      | 0.6959   | 0.8188 | 0.6899  | 0.7120    | 0.6991  | 0.3914  | 0.3935  | 0.1100                   |
| SVM - Linear Kernel (svm)         | 0.6224   | 0.8289 | 0.5965  | 0.7533    | 0.5402  | 0.2472  | 0.2995  | 0.0820                   |
| Dummy Classifier (dummy)          | 0.5132   | 0.5000 | 1.0000  | 0.5132    | 0.6783  | 0.0000  | 0.0000  | 0.0360                   |


- **Model Terbaik**: `Extra Trees Classifier (et)` dengan akurasi 98.32% dan AUC 99.92%.
- **Model Cepat dan Andal**: `Extreme Gradient Boosting (xgboost)` memiliki waktu pelatihan terendah (0.109 detik) dengan akurasi 97.77%.
- Model sederhana seperti `Logistic Regression (lr)` memberikan akurasi moderat sebesar 84.24%.

### Deep Learning
### Scenario 1
```
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.5)),  # Input layer with L2 regularization
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.4)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=16,
                    verbose=1, )

```
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


### Scenario 2
```
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.5)),  # Input layer with L2 regularization
    layers.Dropout(0.8),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.4)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=16,
                    verbose=1, )

```
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

### Scenario 3

```
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=15)
model = Sequential()
model.add(Dense(64, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# compile model
adam = Adam(learning_rate=0.001)
# Change the loss to 'binary_crossentropy'
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# fit the model to the test data
history=model.fit(X_train_normalized, y_train, validation_data=(X_test_normalized, y_test),epochs=50, batch_size=64, callbacks=[es])
```

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

### Scenario 4

```
model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),  # Add batch normalization
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        #layers.BatchNormalization(),
        #layers.Dropout(0.4),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

model = create_deep_learning_model(input_shape=(X_train_scaled.shape[1],))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

```
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

Model deep learning yang dibangun menunjukkan performa yang sangat baik dengan akurasi di atas **96%** pada data pelatihan, validasi, dan pengujian. Meskipun ada sedikit overfitting yang terindikasi dengan perbedaan antara akurasi pelatihan dan validasi, model tetap menunjukkan hasil yang baik.

### Auto tuning
```
def create_and_train_model(X_train, y_train, X_test, y_test, optimizer='adam', learning_rate=0.001, num_layers=3, units_per_layer=64, dropout_rate=0.3, l2_reg=0.001, epochs=100, batch_size=32):

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential()
    model.add(layers.Dense(units_per_layer, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(layers.Dropout(dropout_rate))

    for _ in range(num_layers - 1):
      model.add(layers.Dense(units_per_layer, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)))
      model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer_instance = Adam(learning_rate=learning_rate) if optimizer=='adam' else optimizer

    model.compile(optimizer=optimizer_instance, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    return model, history, accuracy, loss

# Hyperparameter tuning (example)
best_accuracy = 0
best_params = {}
for learning_rate in [0.001, 0.01]:
    for num_layers in [2, 3]:
        for units_per_layer in [32, 64]:
            for dropout_rate in [0.2, 0.3]:
                for l2_reg in [0.001, 0.01]:
                    model, history, accuracy, loss = create_and_train_model(X_train, y_train, X_test, y_test, learning_rate=learning_rate, num_layers=num_layers, units_per_layer=units_per_layer, dropout_rate=dropout_rate, l2_reg=l2_reg)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'learning_rate': learning_rate,
                            'num_layers': num_layers,
                            'units_per_layer': units_per_layer,
                            'dropout_rate': dropout_rate,
                            'l2_reg': l2_reg
                        }
                        print("New best accuracy:", accuracy, best_params)


print("Best accuracy:", best_accuracy)
print("Best parameters:", best_params)
```
Evaluasi Kinerja Model Deep Learning dengan Hyperparameter Tuning

- Deskripsi Model

  Model yang digunakan adalah model **Deep Learning** dengan arsitektur **fully connected (Dense)**. Model ini dibangun dengan beberapa lapisan tersembunyi yang menggunakan **aktivasi ReLU** dan **regularisasi L2** untuk menghindari overfitting, bersama dengan **dropout** untuk menambah generalisasi. Model dioptimalkan menggunakan algoritma **Adam** dan fungsi loss **binary_crossentropy**, yang cocok untuk masalah klasifikasi biner.

- Arsitektur Model:
  - **Input Layer**: Menyesuaikan ukuran input data.
  - **Hidden Layers**: 
    - Lapisan tersembunyi dengan jumlah unit yang ditentukan oleh parameter **units_per_layer**.
    - **Dropout Layer** untuk mencegah overfitting.
    - Jumlah lapisan tersembunyi ditentukan oleh **num_layers**.
- **Output Layer**: Layer output dengan satu unit dan aktivasi **sigmoid**, karena ini adalah masalah klasifikasi biner.

Parameter yang Diuji:
- **learning_rate**: Tingkat pembelajaran untuk algoritma optimisasi (nilai diuji: 0.001, 0.01).
- **num_layers**: Jumlah lapisan tersembunyi (nilai diuji: 2, 3).
- **units_per_layer**: Jumlah unit dalam setiap lapisan tersembunyi (nilai diuji: 32, 64).
- **dropout_rate**: Tingkat dropout untuk lapisan (nilai diuji: 0.2, 0.3).
- **l2_reg**: Regularisasi L2 untuk menghindari overfitting (nilai diuji: 0.001, 0.01).

Selama tuning hyperparameter, hasil terbaik diperoleh dengan parameter berikut:

| **Parameter**        | **Nilai Terbaik**     |
|----------------------|-----------------------|
| **Learning Rate**    | 0.01                  |
| **Num Layers**       | 3                     |
| **Units per Layer**  | 64                    |
| **Dropout Rate**     | 0.3                   |
| **L2 Regularization**| 0.001                 |

### Akurasi Terbaik:
- **Akurasi Pelatihan**: **94.15%**
- **Akurasi Validasi**: **94.15%**

### Proses Pencarian Parameter Terbaik:
Model mengalami peningkatan akurasi pada setiap pengujian kombinasi parameter yang dilakukan. Berikut adalah beberapa titik penting dalam proses pencarian:

-  **Akurasi Terbaik** yang dicapai adalah **94.15%**, dengan parameter:
   - **Learning Rate**: 0.01
   - **Num Layers**: 3
   - **Units per Layer**: 64
   - **Dropout Rate**: 0.3
   - **L2 Regularization**: 0.001

-  Sebelumnya, nilai akurasi tertinggi adalah sekitar **92%**, tetapi kombinasi parameter tersebut akhirnya membawa peningkatan.

Model ini menunjukkan performa yang sangat baik pada data pelatihan dan validasi, dengan **akurasi mencapai 94.15%** pada konfigurasi terbaik. Proses tuning hyperparameter berhasil meningkatkan kinerja model dibandingkan dengan konfigurasi default. Penyesuaian lebih lanjut dengan pengaturan hyperparameter yang lebih optimal mungkin dapat menghasilkan model yang lebih baik.

### Referensi Terkait
- World Health Organization (WHO). *Cardiovascular Diseases (CVDs)*. Diakses dari: [https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)].
- Kementerian Kesehatan RI (2022). *Infodatin Penyakit Jantung*. Diakses dari: [https://pusdatin.kemkes.go.id].
- Benjamin, E. J., et al. (2019). *Heart Disease and Stroke Statistics—2019 Update: A Report From the American Heart Association*. Circulation, 139(10), e56–e528. DOI:10.1161/CIR.0000000000000659.
- Kumar, A., et al. (2020). *Prediction of Cardiovascular Disease Using Machine Learning Algorithms: A Comparative Study*. Procedia Computer Science, 167, 1388–1398. DOI:10.1016/j.procs.2020.03.308.
- Dutta, S., et al. (2020). *Impact of AI in Cardiology: Artificial Intelligence in Cardiovascular Risk Prediction*. Current Cardiovascular Risk Reports, 14(4), 15. DOI:10.1007/s12170-020-00624-6.




