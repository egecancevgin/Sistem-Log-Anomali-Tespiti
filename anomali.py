import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

'''
    Yuklenecekler:
        $ apt-get update
        $ apt-get install python3-pip
        $ pip install pandas
        $ pip install numpy   (Gerekli olmayabilir)
        $ pip install nltk
        $ pip install scikit-learn
        $ pip install gensim
        $ python3 anomali.py
'''


def veri_okuma(dosya_ismi):
    '''
    Ismi girilen dosyayi okur
    :param dosya_ismi: Dosyanin string tipinde ismi
    :return: DataFrame türündeki okunmuş dosya
    '''
    veri = pd.read_csv(dosya_ismi)
    veri = veri.dropna()
    return veri

def ozellik_muhendisligi(veri):
    '''
    Veriyi makine ogrenimine sokmadan once isler
    :param veri: DataFrame turundeki veri seti
    :return: Islenmis veri
    '''
    # Timestamp'i 5 ayrı sutuna parcalayalım ve eski sutunlari kaldiralim
    veri['Month'] = pd.to_datetime(veri['Month'], format='%b').dt.month
    veri['Day'] = veri['Date']
    veri['Hour'] = pd.to_datetime(veri['Time'], format='%H:%M:%S').dt.hour
    veri['Minute'] = pd.to_datetime(veri['Time'], format='%H:%M:%S').dt.minute
    veri['Second'] = pd.to_datetime(veri['Time'], format='%H:%M:%S').dt.second
    veri = veri.drop(['Date', 'Time', 'Level'], axis=1)

    # 'word_tokenize' metodu için gerekli bir yükleme yapalım
    nltk.download('punkt')

    # 'Content' sütununu küçük harfe çevirelim ve kelimelerine ayıralım
    veri['Content'] = veri['Content'].apply(lambda x: word_tokenize(x.lower()))

    # Embedding representation'u oluşturalım
    word2vec_model = Word2Vec(
        sentences=veri['Content'], vector_size=100, window=5,
        min_count=1, workers=4
    )

    def cumle_embedding(cumle, model):
        # Her cumleyi temsil edecek bir feature vector oluşturur, ozetlemedir
        return sum([model.wv[kelime] for kelime in cumle])

    # Feature Vector olusturalim, toplamlarini alarak
    veri['Content_Vector'] = veri['Content'].apply(
        lambda x: cumle_embedding(x, word2vec_model)
    )

    # Component sutununun frekanslarini hesaplayalim, kac tane bulunduklarını
    component_frekanslari = veri['Component'].value_counts(normalize=True)

    # Frequency Encoding islemi yapalım, 'Component' sutunu artık ogrenime uygun
    veri['Component_Frequency_Encoded'] = veri['Component'].map(component_frekanslari)

    return veri

def model_egitimi(veri):
    '''
    Islenmis veriyi makine ogrenimine sokar
    :param veri: DataFrame turundeki islenmis veri seti
    :return: Egitilmis model
    '''
    # K-Means algoritmasi ile 10 centroid ile baslayan bir kumeleme yapalim
    kmeans = KMeans(n_clusters=10, random_state=42)
    veri['Cluster'] = kmeans.fit_predict(list(veri['Content_Vector']))

    # data['Cluster'] = kmeans.fit_predict(data[['Content_Vector','Component_Frequency_Encoded']])

    # Her veriyi en yakın kume merkezine olan uzakligina gore siralayalim
    cluster_merkezleri = kmeans.cluster_centers_
    veri['Distance_to_Center'] = veri.apply(
        lambda row: np.linalg.norm(row['Content_Vector']-cluster_merkezleri[row['Cluster']]),
        axis=1
    )

    # Esik degerini belirleyelim ve esik degeri asan verileri outlier olarak isaretleyelim
    esik_deger = veri['Distance_to_Center'].quantile(0.95)
    veri['Outlier'] = veri['Distance_to_Center'] > esik_deger

    # Incelemek icin Outlier sutununa bakabiliriz (opsiyonel)
    print(veri.loc[veri['Outlier'] == True].info())
    anomali_df = veri.loc[veri['Outlier'] == True]
    anomali_df = anomali_df.drop(['EventId', 'EventTemplate', 'Content_Vector', 'Component_Frequency_Encoded'], axis=1)

    return kmeans, anomali_df


def degerlendirme(model, veri):
    '''
    Makine ögreniminin evaluation asamasini gerceklestirir
    param model: Egitilmis model
    '''
    silh_ortalama = silhouette_score(veri['Content_Vector'].to_list(), veri['Cluster'])
    print("Silhouette Skoru:", silh_ortalama)

    inertia_degeri = model.inertia_
    print("Inertia Degeri:", inertia_degeri)

def optimizasyon(model, veri):
    '''
    Makine ogrenimi modelinin optimizasyonu yapilir
    '''
    parametre_grid = {
        'n_clusters': [8, 10],
        'init': ['k-means++', 'random'],
        'max_iter': [300, 500],
        'tol': [1e-4, 1e-5]
    }

    # Grid Search algoritması kullanarak hiperparametre optimizasyonu yapalım
    grid_search = GridSearchCV(
        estimator=model, param_grid=parametre_grid, cv=3
    )
    grid_search.fit(list(veri['Content_Vector']))

    # En iyi parametre kombinasyonunu alıp ekrana basalım ve en iyi modeli dönelim
    print("En iyi parametreler:", grid_search.best_params_)
    return grid_search.best_estimator_

def cikti(anomaliler):
    '''
    Anomalileri bir cikti dosyasina yazar
    param anomaliler: anomaliler
    '''
    dosya_ismi = 'anomaliler.csv'
    anomaliler.to_csv(dosya_ismi, index=False)
    return dosya_ismi

def main():
    '''
    Sürücü fonksiyon
    '''
    veri = veri_okuma('Linux_2k.log_structured.csv')
    islenmis_veri = ozellik_muhendisligi(veri)
    egitilmis_model, anomaliler = model_egitimi(islenmis_veri)
    cikti(anomaliler)
    degerlendirme(egitilmis_model, islenmis_veri)
    en_iyi_model = optimizasyon(egitilmis_model, islenmis_veri)
    degerlendirme(en_iyi_model, islenmis_veri)

main()
