# TTelekom-Bulut-Bilisim-Projesi
Turk Telekom Cloud Computing Camp Project

# Anomali Tespiti

Anomali tespiti, bir veri kümesindeki tipik örneklerden önemli ölçüde farklı olan, nadir değerleri bulmayı amaçlar.

Bu senaryoda bu işlemin matematiksel olarak nasıl gerçekleştiğini öğrenip, gerekli kütüphaneleri kullanarak kod yazacağız. Sonra da Docker ile bir ELK Stack ayağa kaldırıp verilerimizi 'curl' komutu ile göndereceğiz ve bunun analizlerini inceleyeceğiz.

Hazırsanız başlayalım.

## Birinci Adım: Makine Öğrenimi

Makine öğrenimi, bilgisayar sistemlerinin belirli bir yapıdaki verileri giriş olarak alan, sonrasında makine öğrenimi algoritmaları ile eğitilen ve değerlendirilen, bir tahmin yapmasını sağlayan yapay zeka dalıdır.

Makine öğrenimini representation (veri ve algoritma işleme), evaluation (değerlendirme) ve optimisation (optimizasyon) aşamalarına ayırabiliriz. Bu işlemleri PEP8 standartlarına uygun olmak üzere parçalayıp, fonksiyonlar aracılığıyla gerçekleştirip, ana yürütücü fonksiyonumuzda çağıralım.

Öncelikle makine öğreniminin gerçekleştiği dosyayı /root/workspace dizininde oluşturalım:
``` {.sh}
$ touch anomali.py
```

Sonrasında gerekli kütüphaneleri dosyanın en tepesinde 'import' edelim:
``` python
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
```

Bunu yaptıktan sonra dosyamızı indirelim, terminale şu komutu girin:
``` .sh
curl -O https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log_structured.csv
```

İşlem tamamlandıktan sonra dosyamızı okuma fonksiyonunu oluşturabiliriz, anomali.py dosyamıza şu fonksiyonu ekleyelim:
``` python
def veri_okuma(dosya_ismi):
    '''
    Ismi girilen dosyayi okur
    :param dosya_ismi: Dosyanin string tipinde ismi
    :return: DataFrame turundeki okunmus dosya
    '''
    veri = pd.read_csv(dosya_ismi).dropna()
    return veri
```

Şimdi de bir özellik mühendisliği yapalım ve veriyi işleyelim, verideki zaman pulunu parçalayıp, gerekli makine öğrenimi algoritmalarına hazır hale getirelim. Sonrasında da 'Content' sütununu küçük harfe getirip tokenize edelim. Veri metinsel olduğu için bu veriyi 'Embedding' formatında tutmamız gerekmektedir. Bu format metni parçalara ayırır ve her sözcüğü vektörleştirir. Her metin içine bakar ve vektörleri toplayıp benzer cümleleri temsil eder. Buna ek olarak bu kelime vektör değerlerini (0'dan 1'e) toplayarak uzun cümlelerin kısa cümlelerden uzak olduğunu belirtelim. 

Embedding formatı için Word2Vec algoritmasını kullanalım, metin işleme fonksiyonu içinde cumle_embedding() fonksiyonu da oluşturup özellik vektörümüzü oluşturalım. Son olarak da modele optimizasyon aşamasında ekleyebileceğimiz 'Component' sütununu da frekans bazlı bir temsil ile işleyelim. Bunun sebebi de normal temsillerin ekstra sütunlar ekleyip 'Segmentation Fault' hatası verecek olmasıdır.

Metin işleme ve özellik mühendisliği fonksiyonunumuzu da anomal.py dosyasına ekleyelim:
``` python
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
```

Word2Vec algoritması, 'Continuous Bag of Words' ve 'Skip-gram' modellerini içerir. CBoW modeli, bir kelimenin bağlam içindeki anlamını öğrenmek için tasarlanmıştır. Bir kelimenin bağlam içindeki olasılığı maksimize etmeye çalışır ve hedef kelimenin çevresindeki kelimelerden yola çıkarak, bu kelimenin orada olma olasılığını tahmin eder.
Ör: 'Ayıkla pirincin taşını' cümlesinde 'ayıkla' ve 'taşını' kelimelerinin arasında 'pirincin' olma olasılığı üzerinde çalışır.

Skip-gram da tam tersi olarak, hedef kelime kullanılarak çevresindeki kelimeleri tahmin etmeye çalışır. Kısaca bu iki modeli kullanarak öğretiyoruz.

Şimdi de modeli eğitelim, makine öğrenimi de öğrenme yolu bakımından 3'e ayrılabilir: Denetimli, Denetimsiz ve Pekiştirmeli öğrenim. Denetimli öğrenimde etiket sütunu bulunur. Bu etiket sütunu örnek olarak bir satırda bulunan verinin anomali olduğunu söyler, veya olmadığını. Makineye bunu veririz ve makine bunu öğrenir. Sonraki aşamada tahmin etmesi gerektiğinde bu öğrendiği neyin anomali neyin değil olduğu temsiline göre tahmin eder. 

Verinin etiketlenmesi genellikle insan eliyle gerçekleştirilir ve log projeleri özelinde etiketli veriye ulaşmak çok karşılaşan bir durum değildir. Elimizdeki veri de etiketsiz olduğu için biz de bu durumda denetimsiz makine öğrenimi yapmak durumundayız.

Temel denetimsiz makine öğrenimi algoritmaları K-Means, DBSCAN, Hiyerarşik Kümeleme, Isolation Forest, Gaussian Mixture Models olarak örneklendirilebilir. Biz bu modelde K-Means kullanacağız ancak optimizasyon aşamasında diğer modeller de denenebilir.

K-Means temelde ona verilen 'k' değeri kadar merkez oluşturulur. Örneğin 10 tane merkez oluşturup, verilerin birbirliğine benzerliğine göre gruplandırır. Her iterasyonda bu verileri doğru merkezlere yerleştirmeye çalışır. Sanki 10 farklı renk grubuna sahip bilyeleri 10 farklı keseye renklerine göre ayırarak doldurmak gibidir. Bu şekilde öğrenir ve sonrasında bunun değerlendirmesini de alışılagelmiş precision, recall gibi metriklerle yapamayız, Inertia ve Silhouette Skor değerlerine bakarak yapacağız.

Modeli oluşturup bu fonksiyonu anomali.py dosyamıza ekleyelim:
``` python
def model_egitimi(veri):
    '''
    Islenmis veriyi makine ogrenimine sokar
    :param veri: DataFrame turundeki islenmis veri seti
    :return: Egitilmis model
    '''
    # K-Means algoritmasi ile 10 centroid ile baslayan bir kumeleme yapalim
    kmeans = KMeans(n_clusters=10, random_state=42)
    veri['Cluster'] = kmeans.fit_predict(list(veri['Content_Vector']))

    # Bunu optimizasyon asamasinda deneyebiliriz, simdilik yorum satiri olarak kalsin
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
```

Şimdi de değerlendirme fonksiyonunu oluşturalım ve burada demin bahsettiğimiz iki metriği ekrana basalım, bu fonksiyona parametre olarak eğitilmiş modeli ve veriyi verelim:
``` python
def degerlendirme(model, veri):
    '''
    Makine ögreniminin evaluation asamasini gerceklestirir
    param model: Egitilmis model
    '''
    silh_ortalama = silhouette_score(veri['Content_Vector'].to_list(), veri['Cluster'])
    print("Silhouette Skoru:", silh_ortalama)

    inertia_degeri = model.inertia_
    print("Inertia Degeri:", inertia_degeri)
```

Şimdi de makine öğreniminin final ve en uzun, sürekli değişebilen aşamasının fonksiyonunu oluşturalım ve bunu da anomali.py dosyasına ekleyelim:
``` python
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
```

Makinenin anomali olarak karar verdiği veri noktalarını True/False şeklinde eklenmiş halleri ile anomaliler.csv çıktı dosyasına ekleme fonksiyonunu da oluşturalım ve anomali.py dosyasına ekleyelim:
``` python
def cikti(anomaliler):
    '''
    Anomalileri bir cikti dosyasina yazar
    param anomaliler: anomaliler
    '''
    dosya_ismi = 'anomaliler.csv'
    anomaliler.to_csv(dosya_ismi, index=False)
    return dosya_ismi
```

Son olarak sürücü fonksiyonumuzu oluşturup dosyamıza ekleyelim:
``` python
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
```

Tüm fonksiyonlar hazır, dosyanın sonunda sürücü fonksiyonumuzu çağıralım:
``` python
main()
```

Şimdi anomali.py dosyamız hazır, sırada terminalde yapacağımız bazı işlemler var, sırasıyla tek tek yapalım:
``` .sh
$ apt-get update
$ apt-get install python3-pip
$ pip install pandas
$ pip install numpy
$ pip install scikit-learn
$ pip install gensim
```

Gerekli dosyalar indirilmiş olmalı, şimdi son olarak dosyamızı çalıştırmak kaldı, /root/workspace dizininde olduğumuzdan emin olalım ve şunu terminale yazalım:
``` .sh
$ python3 anomali.py
```

Şimdi anomaliler.csv diye bir dosya oluşmuş olmalı, eğer bu oluşmuşsa ve maksimum 100 tane kayıt içeriyorsa işlem tamam diyebiliriz, sonraki aşamaya geçebiliriz.
``` .sh
ls -l
```
