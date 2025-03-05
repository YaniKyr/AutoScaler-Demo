from prometheus_api_client import PrometheusConnect
import pandas as pd
import requests
from datetime import datetime, timedelta

class Prometheufunctions:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://10.171.51.85:30007", disable_ssl=True)
        self.queries={
            "numpods": "count(up{pod=~'product.*'})",
            "userRequests": "sum(rate(istio_requests_total{pod=~'product.*'}[1m]))",
            "cpuUtil": "avg(rate(container_cpu_usage_seconds_total{container='productpage'}[1m])*100)",
            "RT_obs": "histogram_quantile(0.95, sum by(le) (rate(istio_request_duration_milliseconds_bucket{destination_app='productpage'}[1m])))"
        }
        

    def query(self,query):
        #import ipdb; ipdb.set_trace()
        data = self.prom.custom_query(query=query)
        metric = int(float(data[0]['value'][1]))
        return metric
    def fetchState(self):
        
        cpu = self.query(self.queries['cpuUtil'])
        reqs = self.query(self.queries['userRequests'])
        pods = self.query(self.queries['numpods'])
        return [cpu,reqs,pods]
        

    def liveness(self):
        try:
            liveness_query = 'up{job="prometheus"}'
            liveness_data = self.prom.custom_query(query=liveness_query)
            print("Prometheus is live")
            return True
        except requests.exceptions.ConnectionError as e:
        
            print(e, "Prometheus is not live")
            return  False

    def getSlaVioRange(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        sla_violation_data = self.prom.custom_query_range(
            query=self.queries['RT_obs'],
            start_time=start_time,
            end_time=end_time,
            step='1m')
        sla_violations = [float(point['value'][1]) for point in sla_violation_data]
        if all(value > 1000 for value in sla_violations):
            return True
        
        return False
        
    def getMaxPodsRange(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=10)
        max_pods_data = self.prom.custom_query_range(
            query=self.queries['numpods'],
            start_time=start_time,
            end_time=end_time,
            step='1m')
        max_pods = [int(point['value'][1]) for point in max_pods_data]
        if all(value >= 9 for value in max_pods):
            return True
        return False
    
    def getRTT(self):
        data = self.prom.custom_query(query=self.queries['RT_obs'])

        if len(data) == 0:
            return 0
        return data[0]['value'][1]

#print("Get if persisting Sla Vio",Prometheufunctions().getSlaVioRange())
print("Get if persisting Max Pods",Prometheufunctions().getMaxPodsRange())