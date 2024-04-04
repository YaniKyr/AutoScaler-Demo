[Kubernetes Info](https://github.com/YaniKyr/5G_Autoscaling_Notes/blob/main/papers/Toward_Highly_Scalable_Load_Balancing_in_Kubernetes_Clusters.pdf)


[Cadvisor](https://www.kubecost.com/kubernetes-devops-tools/cadvisor/)
### Main idea

cAdvisor, short for Container Advisor, is an open-source tool developed by Google to monitor containers. It can collect, aggregate, process, and export container-based metrics such as CPU and memory usage, filesystem and network statistics.

### Limitations of Cadvisor
- cAdvisor only collects basic resource utilization information and may not be sufficient if in-depth metrics are needed
- Different OSs will require specific configurations to gather metrics, such as running in privileged mode for RHEL and CentOS, or enabling memory cgroups in Debian
- Collecting metrics for custom hardware like GPUs requires additional configuration, which will differ depending on the underlying infrastructure
- cAdvisor does not provide a method to modify runtime options after initial configuration. Users will need to redeploy the cAdvisor container with the new runtime options if this is required
- cAdvisor requires external tooling to to store collected data long-term and to run any further analytics

<img src="https://github.com/YaniKyr/Thesis_Notes/blob/main/SharedScreenshot.jpg"  width="50%" height="50%">


# Metrics Server

<img src="https://github.com/YaniKyr/Thesis_Notes/blob/main/SharedScreenshot1.jpg"  width="50%" height="50%">

Metrics Server collects resource metrics from Kubelets and exposes them in Kubernetes apiserver through Metrics API

- **Kubelet**. Provides node/pod/container resource usage information (cAdvisor will be slimmed down to provide only core system metrics). Kubelet acts as a node-level and application-level metrics collector as opposed to cAdvisor responsible for cluster-wide metrics.
- **Resource estimator**. Runs as a DaemonSet that turns raw usage values collected from Kubelet into resource estimates ready for the use by schedulers or HPA to maintain the desired state of the cluster.
-** Metrics-server**. This is a mini-version of Heapster (Heapster is now deprecated) that was previously used as the main monitoring solution on top of cAdvisor for collecting Prometheus-format metrics. Metrics-server stores only the latest metrics values scraped from Kubelet and cAdvisor locally and has no sinks (i.e., does not store historical data).
- **Master Metrics API**. Metrics Server exposes the master metrics API via the Discovery summarizer to external clients.
- **The API server**. The server responsible for serving the master metrics API.                  
Metrics Server is not meant for non-autoscaling purposes

[Kubernetes Cluster using Docker Desktop](https://medium.com/womenintechnology/create-a-kubernetes-cluster-using-docker-desktop-72b493f3faa8)


