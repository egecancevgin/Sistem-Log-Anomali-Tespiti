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




