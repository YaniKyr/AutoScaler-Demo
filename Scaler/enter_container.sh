#!/bin/bash

# Fetch all pods in the 'app' namespace
pods=$(microk8s kubectl get pods -n app -o jsonpath='{.items[*].metadata.name}')
for pod in $pods; do
    if [[ $pod == auto-scaler* ]]; then
        microk8s kubectl exec -ti $pod -n app -- bash 
    fi
done
