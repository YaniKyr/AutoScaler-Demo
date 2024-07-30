./deleteall.sh
go build main.go
sudo docker build -t externalscaler .
sudo docker tag externalscaler localhost:32000/externalscaler
sudo docker push localhost:32000/externalscaler
microk8s kubectl apply -f deploy.yaml
microk8s kubectl apply -f scaledobject.yaml
