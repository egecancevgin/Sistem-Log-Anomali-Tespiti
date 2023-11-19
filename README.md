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


## 2.Adım: Elasticsearch Bağlantısı

ELK Stack, açılımı Elasticsearch, Logstash ve Kibana olan üç açık kaynaklı yazılımın kombinasyonunu ifade eder.

- Elasticsearch, veri depolama ve sorgulama için kullanılan dağıtık bir arama/analiz motorudur. 
- Logstash, çeşitli kaynaklardan log verilerini alır, bunları işler, filtreler ve Elasticsearch gibi bir mekanizma için düzenler. 
- Kibana, Elasticsearch üzerinde depolanan log verilerini görselleştirmek ve analiz etmek için kullanılan bir arayüz sağlar.

Elimizdeki veriyi .json bulk formatına çevirip ELK'ya yollarsak önümüze güzel hizmetler çıkar, anomali tespiti ve görselleştirme de bunlara dahildir. Bu entegrasyon oldukça kolaydır ve verimlidir ancak bu anomali tespitinin eksikliği, kodun ve modelin hem değiştirilemez, hem görülemez hem de optimize edilemez olmasıdır. Biz kendi modellerimizi geliştirip, değerlendirip, optimize edip bunu ELK'ya yollamış olacağız.

Ayrıca bizim oluşturduğumuz model basit bir kümeleme işlemi yapıyor ve zaman serisi formatında değil. Kibanada bulunan gömülü anomali tespiti algoritmasında bu bulunmuyor, ve eski dönemde gerçekleştirilmiş ancak anomali olmayan verilere yakın veri noktaları ile karşılaştığımızda anomali tespiti yapıyor. Bizim hazırladığımız algoritmada ise zamandan bağımsız olarak tüm log çıktı mesajlarının içerikleri NLP temsilleri ile incelenip sadece bunlara göre bir kümeleme yapılmaktadır. Bu ELK'da olmayan bir bakış açısıdır ve aslında gizlenmiş çoğu anomaliyi yakalamaya yarayacaktır. Ancak ELK'da bulunan anomali tespit algoritmasında da bulunan çoğu özellikten de yoksundur. Bu yüzden ikisini de kullanmak oldukça verimli olacaktır.

Öncelikle yeni bir dosya açalım ve bu dosyanın amacı .csv uzantılı veri setimizi (anomaliler veya orijinal veri), Elasticsearch için uygun olan bulk .json formatına çevirmek olsun:
``` .sh
$ touch format_degistir.py
```

Şimdi dosyanın tepesinde ihtiyacımız olan kütüphaneleri çağıralım:
``` python
import pandas as pd
import json 
```

Bu dosya içerisine bir fonksiyon oluşturalım, anomaliler index ismiyle ve parametre ile girdiğimiz dosya path'leri ile dosya açalım:
``` python
def json_formatina_cevir(csv_dosya_path, json_dosya_path):
    '''
    Anomalilerin bulunduğu .csv uzantili dosyayi bulk formatina cevirir
    '''
    anomali_df = pd.read_csv(csv_dosya_path)
    json_veri = []
    
    # json_veri listesine satirlari dogru formatta ekleyelim
    for i, row in anomali_df.iterrows():
        json_veri.append({"index": {"_index": "anomaliler", "_id": i}})
        json_veri.append(row.to_dict())

    # .json uzantili dosya olusturup yazalim
    with open(json_dosya_path, 'w') as json_dosya:
        for bulk in json_veri:
            json.dump(bulk, json_dosya)
            json_dosya.write('\n')
```

Şimdi dosyanın en alt kısmında json_formatina_cevir() çağıralım:
``` python
json_formatina_cevir('anomaliler.csv', 'anomaliler.json')
```

Format değiştirme işlemini yapmak için terminalde, /root/workspace dizininde olduğumuzdan emin olup şunu çalıştıralım:
``` .sh
$ python3 format_degistir.py
```

Şimdi bir Docker Compose dosyası oluşturalım:
``` .sh
$ touch docker-compose.yml
```

Bu dosyanın içerisine şunları yazalım:
``` yaml
version: '3'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.10.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  kibana:
    image: docker.elastic.co/kibana/kibana:7.10.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  logstash:
    image: docker.elastic.co/logstash/logstash:7.10.0
    volumes:
      - ./logstash/config:/usr/share/logstash/config
    ports:
      - "5000:5000"
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch
```

Şimdi ELK'yı ayağa kaldıralım, bu komutu yazdıktan sonra bitmesini bekleyelim:
``` .sh
$ docker-compose up -d
```

Sistem çalışıyorsa verimizi 'curl' komutu ile ELK'ya yollayabiliriz:
``` .sh
$ curl -X POST "localhost:9200/_bulk" -H 'Content-Type: application/json' --data-binary @anomaliler.json
```

Eğer bir hata mesajı vermediyse bunu test etmek için 9200 port'una gidelim, ve link kısmına index anahtarını yazalım, örnek:
'https://ip10-244-17-209-user2877-9200.bulutbilisimciler.com' linkine 'anomaliler' index'ini ekleyelim:

'https://ip10-244-17-209-user2877-9200.bulutbilisimciler.com/anomaliler'

Şimdi '5601' portuna girelim ve çıkan ekranda 'Add data' varsa ona basalım. 
![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_0.png)

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_1.png)

Solda bulunan üç çizgili menü'ye basalım ve çıkan yerden Stack Management'a basalım.

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_2.png)

- Stack Management bölümünde 'Index Patterns' kısmı var, buna girelim ve çıkan ekranda 'Create index pattern' tuşuna basalım, ve buraya anomaliler yazıp devam edelim.

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_3.png)

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_4.png)

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_5.png)

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_6.png)

Burada sıkıntı yoksa yine üç çizgili sol menüden bu sefer Kibana başlığı altındaki Machine Learning kısmına tıklayalım.

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_7.png)

Bizi otomatik olarak attığı 'Data Visualizer' sayfasındaki 'Select an index pattern' başlığındaki 'Select index' butonuna basalım.

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_8.png)

Zaten burada 'anomaliler' index'i bulunuyor olmalı, bu index'e basalım.

![elo_0](https://github.com/egecancevgin/TTelekom-Bulut-Bilisim-Projesi/blob/49e2dc586a26aa34ab34a4392315a2b84575e2a5/elastic_9.png)

İşte bu kadar, tüm analizleri yapabiliriz, dashboard'a girmiş olduk.

Bu projenin ilerleyen kısımlarında RNN modelleri kullanımı, Isolation Forest ve zaman serisi modeline eklemeler de vardır, geliştirilmeye açık, ve hatta ayrı bir makine öğrenimi projesi olan Source Overcapacity Forecasting ile de birleştirilip, alerting mekanizmaları ile verimli kaynak ve enerji kullanımı sistemleri oluşturulabilir.
