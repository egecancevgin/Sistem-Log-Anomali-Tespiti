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
![elo_0]()

Solda bulunan üç çizgili menü'ye basalım ve çıkan yerden Stack Management'a basalım.

- Stack Management bölümünde 'Index Patterns' kısmı var, buna girelim ve çıkan ekranda 'Create index pattern' tuşuna basalım, ve buraya anomaliler yazıp devam edelim.

Burada sıkıntı yoksa yine üç çizgili sol menüden bu sefer Kibana başlığı altındaki Machine Learning kısmına tıklayalım.

Bizi otomatik olarak attığı 'Data Visualizer' sayfasındaki 'Select an index pattern' başlığındaki 'Select index' butonuna basalım.

Zaten burada 'anomaliler' index'i bulunuyor olmalı, bu index'e basalım.

İşte bu kadar, tüm analizleri yapabiliriz, dashboard'a girmiş olduk.

Bu projenin ilerleyen kısımlarında RNN modelleri kullanımı, Isolation Forest ve zaman serisi modeline eklemeler de vardır, geliştirilmeye açık, ve hatta ayrı bir makine öğrenimi projesi olan Source Overcapacity Forecasting ile de birleştirilip, alerting mekanizmaları ile verimli kaynak ve enerji kullanımı sistemleri oluşturulabilir.
