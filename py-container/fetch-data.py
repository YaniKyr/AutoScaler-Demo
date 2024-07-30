from prometheus_api_client import PrometheusConnect
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt

prom = PrometheusConnect(url ="http://10.231.59.94:30007", disable_ssl=True)

query = "avg(sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~'php.*'}[1m])) / sum by (pod) (kube_pod_container_resource_requests{unit='core'})*100)[24h:1m]"


data =  prom.custom_query(query=query)

tdf = pd.DataFrame.from_dict(data[0]['values'])

tdf = tdf.rename(columns={0: 'timestamp', 1: 'value'})

tdf['timestamp'] = pd.to_datetime(tdf['timestamp'],unit = 's')

tdf['value'] = tdf['value'].astype(float)

plt.xticks(rotation=70)

plt.plot(tdf['timestamp'],tdf['value'])
plt.show()