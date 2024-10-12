./deleteall.sh
go build main.go
sudo docker build -t auto-scaler .
sudo docker tag auto-scaler localhost:32000/auto-scaler
sudo docker push localhost:32000/auto-scaler
microk8s kubectl apply -f deploy.yaml


