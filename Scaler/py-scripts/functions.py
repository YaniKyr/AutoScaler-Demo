from prometheus_api_client import PrometheusConnect
import pandas as pd
import requests
from datetime import datetime, timedelta

class Prometheufunctions:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://10.152.183.212:9090", disable_ssl=True)
        self.queries={
            "numpods": "count(up{namespace='demo'})by (pod)",
            "userRequests": "sum(rate(istio_requests_total{pod=~'product.*'}[1m]))",
            "cpuUtil": "avg(sum(rate(container_cpu_usage_seconds_total{pod=~'productpage.*',namespace='demo'}[1m])*100))",
            "RT_obs": "histogram_quantile(0.95, sum by(le) (rate(istio_request_duration_milliseconds_bucket[1m])))"
        }
        

    def query(self,query):
        #import ipdb; ipdb.set_trace()
        data = self.prom.custom_query_range(query=query,step='1m',start_time=datetime.now() - timedelta(hours=1),end_time=datetime.now())
        df = pd.DataFrame.from_dict(data[0]['values'])
        df = df.rename(columns={0: 'timestamp', 1: 'value'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)
        return df
    
    def fetchState(self):
        
        cpu = self.query(self.queries['cpuUtil'])
        reqs = self.query(self.queries['userRequests'])
        pods = self.query(self.queries['numpods'])
        
        merged_df = pd.merge(cpu, reqs, on='timestamp', suffixes=('_cpu', '_user'))
        merged_df = pd.merge(merged_df, pods, on='timestamp')
        merged_df = merged_df.rename(columns={'value': 'num_pods'})
        
        return merged_df[:-2]
        

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

        if len(data) == 0:
            return 0
        return data[0]['value'][1]
    
    