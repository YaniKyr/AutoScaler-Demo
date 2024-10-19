from flask import Flask, request, jsonify
from prometheus_api_client import PrometheusConnect
import pandas as pd
import json
import requests
import numpy as np

app = Flask(__name__)

@app.route('/get_prometheus_data', methods=['POST'])
def get_prometheus_data():
    query = request.json.get('query')
    prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
    data = prom.custom_query(query=query)
    
    df = pd.DataFrame.from_dict(data[0]['values'])
    df = df.rename(columns={0: 'timestamp', 1: 'value'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(float)

    response_data = {"timestamp":"0/0/0","value":df['value'].iloc[-1]}
    print(response_data)
    return jsonify(response_data)


def liveness():
    try:
        prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
        liveness_query = 'up{job="prometheus"}'
        liveness_data = prom.custom_query(query=liveness_query)
        print("Prometheus is live")
    except requests.exceptions.ConnectionError as e:
       
        print(e, "Prometheus is not live")
        return
    

if __name__ == '__main__':
    liveness()
    app.run(host='0.0.0.0', port=5000)