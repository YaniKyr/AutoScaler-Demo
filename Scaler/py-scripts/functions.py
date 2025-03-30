from prometheus_api_client import PrometheusConnect
import requests
from datetime import datetime, timedelta

class Prometheufunctions:
    def __init__(self):
        self.prom = PrometheusConnect(url="http://prometheus.istio-system:9090", disable_ssl=True)
        self.queries={
            "numpods": "count(up{pod=~'product.*'})",
            "userRequests": "sum(rate(istio_requests_total{pod=~'product.*'}[1m]))",
            "cpuUtil": "avg(rate(container_cpu_usage_seconds_total{container='productpage'}[1m]))",
            "RT_obs": "histogram_quantile(0.95, sum by(le) (rate(istio_request_duration_milliseconds_bucket{destination_app='productpage'}[1m])))"
        }
        

    def getQueryRange(self, query, start_time=None, end_time=None, step='1m'):
        if start_time and end_time:
            data = self.prom.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step
            )
        else:
            data = self.prom.custom_query(query=query)
        return data
    
    def floodReplayBuffer(self, minutes=1):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        cpu_data = self.getQueryRange(self.queries['cpuUtil'], start_time, end_time)
        reqs_data = self.getQueryRange(self.queries['userRequests'], start_time, end_time)
        pods_data = self.getQueryRange(self.queries['numpods'], start_time, end_time)
        rt_obs = self.getQueryRange(self.queries['RT_obs'], start_time, end_time)

        cpu = [float(point[1]) for point in cpu_data[0]['values']]
        reqs = [float(point[1]) for point in reqs_data[0]['values']]
        pods = [int(point[1]) for point in pods_data[0]['values']]
        rt = [float(point[1]) for point in rt_obs[0]['values']]
        return cpu, reqs, pods, rt
        

    def query(self,query):
        #import ipdb; ipdb.set_trace()
        data = self.prom.custom_query(query=query)
        metric = float(data[0]['value'][1])
        
        return metric
    
    def fetchState(self):
        
        cpu = self.query(self.queries['cpuUtil'])
        reqs = int(self.query(self.queries['userRequests']))
        pods = int(self.query(self.queries['numpods']))
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
        start_time = end_time - timedelta(minutes=8)
        sla_violation_data = self.prom.custom_query_range(
            query=self.queries['RT_obs'],
            start_time=start_time,
            end_time=end_time,
            step='1m')
        sla_violations = [float(point[1]) for point in sla_violation_data[0]['values']]
        if all(value > 500 for value in sla_violations):
            return True
        
        return False
        
    def getMaxPodsRange(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=8)
        max_pods_data = self.prom.custom_query_range(
            query=self.queries['numpods'],
            start_time=start_time,
            end_time=end_time,
            step='1m')
        max_pods = [int(point[1]) for point in max_pods_data[0]['values']]
        if all(value >= 8 for value in max_pods):
            return True
        return False
    
    def getRTT(self):
        data = self.prom.custom_query(query=self.queries['RT_obs'])

        if len(data) == 0:
            return 0
        return data[0]['value'][1]

