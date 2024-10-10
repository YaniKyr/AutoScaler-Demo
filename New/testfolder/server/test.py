import grpc
from concurrent import futures
import time
import prom_pb2
import prom_pb2_grpc
from prometheus_api_client import PrometheusConnect
import pandas as pd
import json
import requests
class PrometheusServiceServicer(prom_pb2_grpc.PrometheusServiceServicer):
    def GetPrometheusData(self, request, context):
        prom = PrometheusConnect(url ="http://10.152.183.236:9090", disable_ssl=True)
        data = prom.custom_query(query=request.query)
        df = pd.DataFrame.from_dict(data[0]['values'])
        df = df.rename(columns={0: 'timestamp', 1: 'value'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)


        x = {"timestamp":df['timestamp'],"value":df['value']}
    
        response = prom_pb2.PrometheusResponse(data=json.dumps(x, default=str))
        return response

def serve():
    try:
        prom = PrometheusConnect(url="http://10.152.183.236:9090", disable_ssl=True)
        liveness_query = 'up{job="prometheus"}'
        liveness_data = prom.custom_query(query=liveness_query)
        print("Prometheus is live")
    except requests.exceptions.ConnectionError as e:
       
        print(e, "Prometheus is not live")
        return
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prom_pb2_grpc.add_PrometheusServiceServicer_to_server(PrometheusServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()