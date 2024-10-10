from prometheus_api_client import PrometheusConnect, MetricSnapshotDataFrame
import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import socket 
import requests
import json
import time
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def scaler():
    prom = PrometheusConnect(url ="http://10.152.183.236:9090", disable_ssl=True)

    query = '''avg(sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~'php.*'}[1m])) / sum by (pod) (kube_pod_container_resource_requests{pod=~'php.*',unit='core'})*100)[3h:1m]'''
    data = prom.custom_query(query=query)

    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(data[0]['values'])
    df = df.rename(columns={0: 'timestamp', 1: 'value'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(float)


    x = {"timestamp":df['timestamp'],"value":df['value']}
    print(x)

    return x

    

def main():
    #receive liveness message from prometheus
    

    
    try:
        prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
        liveness_query = 'up{job="prometheus"}'
        liveness_data = prom.custom_query(query=liveness_query)
        print("Prometheus is live")
    except requests.exceptions.ConnectionError as e:
       
        print(e, "Prometheus is not live")
        return

    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8082))
    server_socket.listen(1)
    print("Server listening on port 8082")

    while True:
        print("server started iteration")
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")
        predictions = scaler()
        message = json.dumps(predictions, default=str)
        conn.sendall(message.encode('utf-8'))
        conn.close()
        time.sleep(5)

if __name__ == "__main__":
    main()
    

