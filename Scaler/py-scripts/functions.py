from prometheus_api_client import PrometheusConnect
import pandas as pd
import requests


class Prometheufunctions:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://prometheus:9090", disable_ssl=True)
        self.queries={
            "numpods": "count(up{namespace='demo'})by (pod)[1h:]",
            "userRequests": "avg(rate(container_network_receive_bytes_total{pod=~'.*[v1|v2|v3].*',namespace='default'}[1m])/100000)[1h:]",
            "cpuUtil": "avg(sum(rate(container_cpu_usage_seconds_total{pod=~'.*[v1|v2|v3].*',namespace='default'}[1m])*100)[1h:])",
            "RT_obs": "histogram_quantile(0.95, sum by(le) (rate(istio_request_duration_milliseconds_bucket[1m])))"
        }
        

    def query(self,query):
        data = self.prom.custom_query(query=query)
        df = pd.DataFrame.from_dict(data[0]['values'])
        df = df.rename(columns={0: 'timestamp', 1: 'value'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)
        return df
    
    def fetchState(self):
        
        cpu = self.query( self.queries['cpuUtil'])
        reqs = self.query( self.queries['userRequests'])
        pods = self.query( self.queries['numpods'])
        merged_df = pd.merge(cpu, reqs, on='timestamp', suffixes=('_cpu', '_user'))
        merged_df = pd.merge(merged_df, pods, on='timestamp')
        merged_df = merged_df.rename(columns={'value': 'num_pods'})
        return merged_df
        

    def liveness(self):
        try:
            liveness_query = 'up{job="prometheus"}'
            liveness_data = self.prom.custom_query(query=liveness_query)
            print("Prometheus is live")
            return True
        except requests.exceptions.ConnectionError as e:
        
            print(e, "Prometheus is not live")
            return  False
        
    def getRTT(self):
        data = self.prom.custom_query(query=self.queries['RT_obs'])
        return data[0]['value'][1]
    
    