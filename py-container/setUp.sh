microk8s kubectl delete deployment py-server

sudo docker build -t py-server .
sudo docker tag py-server localhost:32000/py-server
sudo docker push localhost:32000/py-server
microk8s kubectl apply -f deployment.yaml

