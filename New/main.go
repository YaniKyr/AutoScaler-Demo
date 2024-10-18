package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"io/ioutil"
	"bytes"
	pb "github.com/kedacore/keda/v2/pkg/scalers/externalscaler"
	"google.golang.org/grpc"
)

type Log struct {
	Timestamp string `json:"timestamp"`
	Value     float64 `json:"value"`
}
type PrometheusRequest struct {
    Query string `json:"query"`
}

type ExternalScaler struct {
	pb.UnimplementedExternalScalerServer
}

// Fetch data from the service and convert to int64
func getData() Log {

    url := "http://127.0.0.1:5000/get_prometheus_data"

    resp, err := http.Post(url, "application/json")
    if err != nil {
        log.Fatalf("could not fetch data: %v", err)
    }
    defer resp.Body.Close()

    // Read the response body
    bodyBytes, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        log.Fatalf("could not read response body: %v", err)
    }

    var prometheusResponse Log
    err = json.Unmarshal(bodyBytes, &prometheusResponse)
    if err != nil {
        log.Fatalf("could not unmarshal response: %v", err)
    }

    fmt.Printf("Prometheus Response: %+v\n", prometheusResponse)
	return prometheusResponse
}
	

// Check if the scaler is active
func (e *ExternalScaler) IsActive(ctx context.Context, ScaledObject *pb.ScaledObjectRef) (*pb.IsActiveResponse, error) {
	Log := getData()


	isActive := Log.Value > 50
	fmt.Printf("IsActive called: value = %d, Result = %v\n", Log.Value, isActive)
	return &pb.IsActiveResponse{
		Result: isActive,
	}, nil
}

// StreamIsActive is not implemented
func (e *ExternalScaler) StreamIsActive(ref *pb.ScaledObjectRef, stream pb.ExternalScaler_StreamIsActiveServer) error {
	fmt.Println("StreamIsActive called but not implemented")
	return nil
}

// GetMetricSpec provides the metric specification
func (e *ExternalScaler) GetMetricSpec(ctx context.Context, ref *pb.ScaledObjectRef) (*pb.GetMetricSpecResponse, error) {
	metricSpec := &pb.MetricSpec{
		MetricName: "constant_metric",
		TargetSize: 50,
	}
	fmt.Printf("GetMetricSpec called: %v\n", metricSpec)
	return &pb.GetMetricSpecResponse{
		MetricSpecs: []*pb.MetricSpec{metricSpec},
	}, nil
}

// GetMetrics provides the current metric values
func (e *ExternalScaler) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.GetMetricsResponse, error) {
	Log := getData()
	

	// Adjust value within bounds
	

	fmt.Printf("GetMetrics called: Raw value = %d\n", Log.Value)
	return &pb.GetMetricsResponse{
		MetricValues: []*pb.MetricValue{{
			MetricName:  "constant_metric",
			MetricValue: int64(Log.Value),
		}},
	}, nil
}

func main() {
	grpcAddress := "0.0.0.0:50051"
	listener, err := net.Listen("tcp", grpcAddress)
	if err != nil {
		log.Fatalf("failed to listen: %v\n", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterExternalScalerServer(grpcServer, &ExternalScaler{})
	fmt.Printf("Server listening on %s\n", grpcAddress)
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatalf("failed to serve: %v\n", err)
	}
}
