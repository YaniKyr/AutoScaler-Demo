from prometheus_api_client import PrometheusConnect
import time
import datetime
from flask import Flask 


app = Flask(__name__) 
 
@app.route('/') 
def home(): 
    query = '''avg(sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~'php.*'}[1m])) / sum by (pod) (kube_pod_container_resource_requests{unit='core'})*100)'''
    while 1:
        resp = prom.custom_query(query=query)

        for item in resp:
            timestamp = datetime.datetime.fromtimestamp(item['value'][0])
            tmstmp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            x = {"timestamp":tmstmp,"value":item['value'][1]}
            return x
        time.sleep(5)

    
    
 




prom = PrometheusConnect(url ="http://10.152.183.218:9090", disable_ssl=True)

# Get the list of all the metrics that the Prometheus host scrapes

if __name__ == '__main__': 
    app.run(debug=True,port=8001)
