import pandas as pd
import json 


'''
    Yapilacaklar:
        $ python3 format_degistir.py
        $ docker-compose up -d
        $ curl -X POST "localhost:9200/_bulk" -H 'Content-Type: application/json' --data-binary @anomaliler.json
'''


def json_formatina_cevir(csv_dosya_path, json_dosya_path):
    '''
    Anomalilerin bulunduğu .csv uzantili dosyayi bulk formatina cevirir
    params:
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


json_formatina_cevir('anomaliler.csv', 'anomaliler.json')


"""
docker-compose.yml dosyasi icerigi:

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
      - "5000:5000"  # Logstash TCP input port
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch


logstash.conf dosyasi icerigi (opsiyonel):

input {
  file {
    path => "/usr/share/logstash/pipeline/anomaliler.csv"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}

filter {
  csv {
    separator => ","
    columns => ["field1", "field2", "field3"]  
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "anomalies"
  }
  stdout { codec => rubydebug }
}

"""