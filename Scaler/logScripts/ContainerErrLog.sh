#!/bin/bash

# Fetch all pods in the 'app' namespace
pods=$(microk8s kubectl get pods -n app -o jsonpath='{.items[*].metadata.name}')
for pod in $pods; do
    if [[ $pod == auto-scaler* ]]; then
        microk8s kubectl exec -ti $pod -n app -- bash -c '
        cat << EOF > log_watcher.sh
#!/bin/bash

# Path to the log file
LOG_FILE="/var/log/supervisor/python_server.err.log"

# Loop indefinitely
while true; do
    # Check if the log file exists
    if [ -f "\$LOG_FILE" ]; then
        echo "---- Log Output (Last Updated: \$(date)) ----"
        cat "\$LOG_FILE"
        echo "-------------------------------------------"
    else
        echo "Log file \$LOG_FILE does not exist. Retrying..."
    fi

    # Wait for 5 seconds before the next iteration
    sleep 5
done
EOF

# Make the script executable
chmod +x log_watcher.sh

# Run the script
./log_watcher.sh
'
    fi
done
