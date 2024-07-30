package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"strings"

	pb "github.com/kedacore/keda/v2/pkg/scalers/externalscaler"
	"google.golang.org/grpc"
)

type Log struct {
	Timestamp string `json:"timestamp"`
	Value     float64 `json:"value"`
}

type ExternalScaler struct {
	pb.UnimplementedExternalScalerServer
}

// Fetch data from the service and convert to int64
func getData() (int64, error) {
	conn, err := net.Dial("tcp", "py-service:8001")
	if err != nil {
		fmt.Println("dial error:", err)
		return 0, err
	}
	defer conn.Close()

	fmt.Fprintf(conn, "GET / HTTP/1.0\r\n\r\n")

	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 256)
	for {
		n, err := conn.Read(tmp)
		if err != nil {
			if err != io.EOF {
				fmt.Println("read error:", err)
			}
			break
		}
		buf = append(buf, tmp[:n]...)
	}

	val := string(buf)
	parts := strings.SplitN(val, "\r\n\r\n", 2)
	if len(parts) < 2 {
		fmt.Println("Invalid HTTP response format")
		return 0, fmt.Errorf("invalid HTTP response format")
	}
	body := parts[1]

	var responseData Log
	if err := json.Unmarshal([]byte(body), &responseData); err != nil {
		fmt.Println("Error unmarshaling JSON:", err)
		return 0, err
	}

	intValue := int64(math.Round(responseData.Value))
	//fmt.Printf("Fetched data: %f, Rounded value: %d\n", data, intValue)
	return intValue, nil
}

// Check if the scaler is active
func (e *ExternalScaler) IsActive(ctx context.Context, ScaledObject *pb.ScaledObjectRef) (*pb.IsActiveResponse, error) {
	value, err := getData()
	if err != nil {
		return nil, err
	}

	isActive := value > 200
	fmt.Printf("IsActive called: value = %d, Result = %v\n", value, isActive)
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
		TargetSize: 300,
	}
	fmt.Printf("GetMetricSpec called: %v\n", metricSpec)
	return &pb.GetMetricSpecResponse{
		MetricSpecs: []*pb.MetricSpec{metricSpec},
	}, nil
}

// GetMetrics provides the current metric values
func (e *ExternalScaler) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.GetMetricsResponse, error) {
	value, err := getData()
	if err != nil {
		return nil, fmt.Errorf("error getting desired value: %w", err)
	}

	// Adjust value within bounds
	

	fmt.Printf("GetMetrics called: Raw value = %d, Adjusted value = %d\n", value, value)
	return &pb.GetMetricsResponse{
		MetricValues: []*pb.MetricValue{{
			MetricName:  "constant_metric",
			MetricValue: value,
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