# Limitations of Cadvisor
    + cAdvisor only collects basic resource utilization information and may not be sufficient if in-depth metrics are needed
    + Different OSs will require specific configurations to gather metrics, such as running in privileged mode for RHEL and CentOS, or enabling memory cgroups in Debian
    + Collecting metrics for custom hardware like GPUs requires additional configuration, which will differ depending on the underlying infrastructure
    + cAdvisor does not provide a method to modify runtime options after initial configuration. Users will need to redeploy the cAdvisor container with the new runtime options if this is required
    + cAdvisor requires external tooling to to store collected data long-term and to run any further analytics
