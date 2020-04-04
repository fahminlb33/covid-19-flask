# COVID-19 VGG16

Repositori ini berisi kode yang digunakan pada esai dengan judul 
**Klasifikasi COVID-19 Menggunakan Citra Chest X-Ray**.

Sementara ini penulis tidak menyediakan file Jupyter Notebook :smile:

Written by: Fahmi Noor Fiqri (https://kodesiana.com)

## Setup Environment

Clone repository ini dan buat environment menggunakan Anaconda.
Anda juga bisa menggunakan virtual environment lain selain Anaconda.

```
git clone https://github.com/fahminlb33/covid-19-flask.git
cd covid-19-flask
conda create --name covid-19-flask --file requirements.txt
conda activate covid-19-flask
```

Untuk melakukan training, visualisai, dan menjalankan server Flask,
eksekusi perintah:

```
python train.py
python visualize.py
python web.py
```

Untuk penggunaan yang lebih spesifik, Anda dapat menggunakan command-line
seperti di bawah ini.

## Cara Penggunaan

Proyek ini berisi tiga file utama, yaitu **train.py, visualize.py**, dan
**web.py**. Ketiga file ini memiliki fungsi yang berbeda-beda seperti yang
di sebagai berikut.

### Training

```
python train.py [-d|--dataset] [-s|--stats] [-m|--model]
```

Argumen:

- `-d|--dataset`, lokasi input dataset.
- `-s|--stats`, lokasi output statistik *training* (accuracy-loss graph,
  confusion matrix, gambar augmentasi).
- `-m|--m`, lokasi output file model dari *neural network*.

Contoh:

```
python train.py --dataset dataset --stats model --model model\covid-vgg.h5
```

### Visualisasi *Intermediate Activation*

```
python visualize.py [-i|--input] [-s|--stats] [-m|--model]
```

Argumen:

- `-i|--input`, input file citra chest x-ray untuk divisualisasikan.
- `-s|--stats`, lokasi output visualisasi.
- `-m|--model`, lokasi input model yang akan divisualisasikan.

Contoh:

```
python visualize.py --input dataset\\normal\\IM-0115-0001.jpeg --stats model\layers --model model\covid-vgg.h5
```

### Running Server Flask

```
python web.py [-m|--model]
```

Argumen:

- `-m|--model`, lokasi input model yang akan digunakan.

Contoh:

```
python web.py --model model\covid-vgg.h5
```
