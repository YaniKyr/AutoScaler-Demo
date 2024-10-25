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

Metrics Server is a scalable, efficient source of container resource metrics for Kubernetes built-in autoscaling pipelines.

Metrics Server collects resource metrics from Kubelets and exposes them in Kubernetes apiserver through Metrics API for use by Horizontal Pod Autoscaler and Vertical Pod Autoscaler. [Metrics Server](https://kubernetes-sigs.github.io/metrics-server/)

- **Kubelet**. Provides node/pod/container resource usage information (cAdvisor will be slimmed down to provide only core system metrics). Kubelet acts as a node-level and application-level metrics collector as opposed to cAdvisor responsible for cluster-wide metrics.
- **Resource estimator**. Runs as a DaemonSet that turns raw usage values collected from Kubelet into resource estimates ready for the use by schedulers or HPA to maintain the desired state of the cluster.
-** Metrics-server**. This is a mini-version of Heapster (Heapster is now deprecated) that was previously used as the main monitoring solution on top of cAdvisor for collecting Prometheus-format metrics. Metrics-server stores only the latest metrics values scraped from Kubelet and cAdvisor locally and has no sinks (i.e., does not store historical data).
- **Master Metrics API**. Metrics Server exposes the master metrics API via the Discovery summarizer to external clients.
- **The API server**. The server responsible for serving the master metrics API.                  
Metrics Server is not meant for non-autoscaling purposes

[Kubernetes Cluster using Docker Desktop](https://medium.com/womenintechnology/create-a-kubernetes-cluster-using-docker-desktop-72b493f3faa8)

## Some Sum up

Cadvisor captures the state and returns the data of containers. It is container based. 
Metrics Api has access to K8s control plane. Collects resource metrics from Kubelets and exposes them in Kubernetes apiserver through Metrics API. 

Proposal: Metrics pipeline utilizes both metrics api and Cadvisor
In a local-worker node point of view, Cadvisor collects reports and posts thme in kubelet. Then metrics api exposes the data from kubelet to API server (control plane). [Metrics Pipeline](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/)

<img src="https://github.com/YaniKyr/Thesis_Notes/blob/main/SharedScreenshot2.jpg"  width="50%" height="50%">



The question of whether Metrics Server is better than cAdvisor depends on your specific use case and requirements. Both Metrics Server and cAdvisor serve similar but slightly different purposes in the realm of container monitoring within Kubernetes clusters.

## A brief comparison:

- Functionality:
  * Metrics Server: Metrics Server is primarily designed to gather resource usage metrics (such as CPU and memory) from Kubernetes nodes and pods. It provides aggregated metrics, which are useful for autoscaling purposes and general cluster monitoring.
  * cAdvisor: cAdvisor (Container Advisor) is more focused on providing detailed container-level metrics, including resource usage, performance statistics, and information about running processes within containers.

- Integration with Kubernetes:
  * Metrics Server: Metrics Server is tightly integrated with Kubernetes and is the recommended way to gather resource metrics for Horizontal Pod Autoscaler and Vertical Pod Autoscaler.
  * cAdvisor: cAdvisor can be run as a standalone service or as part of a Kubernetes cluster. It's typically deployed as a node-level agent and collects metrics directly from the Docker daemon or container runtime.

- Resource Usage:
  * Metrics Server: Metrics Server is lightweight and optimized for collecting resource usage metrics at scale in Kubernetes clusters.
  * cAdvisor: cAdvisor collects detailed metrics at the container level, which may be more resource-intensive compared to Metrics Server, especially in larger clusters.

- Granularity:
  *  Metrics Server: Metrics Server provides aggregated metrics at the node and pod level, suitable for cluster-level monitoring and autoscaling decisions.
  *  cAdvisor: cAdvisor provides detailed metrics at the container level, offering more granularity for monitoring and troubleshooting individual containers.

In summary, if you're primarily interested in gathering high-level resource usage metrics for autoscaling and cluster monitoring within Kubernetes, Metrics Server is a good choice. However, if you need more detailed container-level metrics or want to monitor containers outside of Kubernetes, cAdvisor may be a better fit. In some cases, both Metrics Server and cAdvisor can be used together to achieve comprehensive monitoring and resource management within Kubernetes clusters.


