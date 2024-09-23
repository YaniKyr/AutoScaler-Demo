import pmdarima as pm
import warnings
from prometheus_api_client import PrometheusConnect, MetricSnapshotDataFrame
import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
import socket 
import json
#from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def scaler():
    prom = PrometheusConnect(url ="https://prometheus-cpro-server.cpro.svc.cluster.local:80", disable_ssl=True)

    query = '''avg(sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~'php.*'}[1m])) / sum by (pod) (kube_pod_container_resource_requests{pod=~'php.*',unit='core'})*100)[3h:1m]'''
    data = prom.custom_query(query=query)

    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(data[0]['values'])
    df = df.rename(columns={0: 'timestamp', 1: 'value'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(float)

    # Split data into training and testing sets
    size = int(len(df) * 0.66)
    tdf = df['value']
    train, test = tdf[0:size], tdf[size:len(df)]

    # Ensure train data is stationary
    while adfuller(train)[1] > 0.05:
        train = train.diff().dropna()

    # Plot training data
    #train.plot()
    #plt.title('Differenced Training Data')
    #plt.savefig('differenced_training_data.png')
    #plt.show()

    # Ignore warnings from numpy
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    # Fit ARIMA model
    auto_arima = pm.auto_arima(train, stepwise=False, seasonal=False)
    print(auto_arima)

    # Forecast using the ARIMA model
    forecast_test_auto = auto_arima.predict(n_periods=len(test))

    # Combine the forecasts with the original data
    df_forecast = df.copy()
    df_forecast['forecast_auto'] = None
    df_forecast.loc[size:, 'forecast_auto'] = forecast_test_auto

    #print(forecast_test_auto.mean())
    # Calculate metrics
    # Align test data and forecast data by dropping NaNs created by differencing
    aligned_test = test.iloc[:len(forecast_test_auto)]

    #mae = mean_absolute_error(aligned_test, forecast_test_auto)
    #mape = mean_absolute_percentage_error(aligned_test, forecast_test_auto)
    #rmse = np.sqrt(mean_squared_error(aligned_test, forecast_test_auto))

    #print(f'MAE: {mae}')
    #print(f'MAPE: {mape}')
    #print(f'RMSE: {rmse}')

    x = {"timestamp":df['timestamp'],"value":np.abs(forecast_test_auto.mean())}
    return x

    

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8082))
    server_socket.listen(1)
    print("Server listening on port 8081")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")
        predictions = scaler()
        message = json.dumps(predictions, default=str)
        conn.sendall(message.encode('utf-8'))
        conn.close()

if __name__ == "__main__":
    main()
    

