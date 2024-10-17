from prometheus_api_client import PrometheusConnect
import pandas as pd

def query(prom, query):
    data = prom.custom_query(query=query)
    df = pd.DataFrame.from_dict(data[0]['values'])
    df = df.rename(columns={0: 'timestamp', 1: 'value'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(float)
    return df
def queries():
    numpods = "sum(kube_pod_info{pod='productpage-v1-d5789fdfb-6q9r6'})"
    userRequests = "avg(rate(container_network_receive_bytes_total{pod=~'.*[v1|v2|v3].*',namespace='default'}[1m])/100000)[1h:]"

    cpuUtil = "sum(rate(container_cpu_usage_seconds_total{pod=~'.*[v1|v2|v3].*',namespace='default'}[1m])*100)[1h:]"

    prom = PrometheusConnect(url="http://10.152.183.182:9090", disable_ssl=True)
    cpu = query(prom, cpuUtil)
    user = query(prom, userRequests)
    data = prom.custom_query(query=numpods)
    pods= data[0]['value'][1]

    return cpu, user, int(pods)

def liveness():
    try:
        prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
        liveness_query = 'up{job="prometheus"}'
        liveness_data = prom.custom_query(query=liveness_query)
        print("Prometheus is live")
    except requests.exceptions.ConnectionError as e:
    
        print(e, "Prometheus is not live")
        return