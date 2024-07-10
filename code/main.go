package scaler

import (
	"context"
	"encoding/json"
	
	"fmt"
	"io"

	

	"net"
	"strconv"
	"strings"

	pb "github.com/kedacore/keda/v2/pkg/scalers/externalscaler"

)

type Log struct {
	Timestamp string `json:"timestamp"`
	Value     string `json:"value"`
}
var _ pb.ExternalScalerServer = &ExternalScaler{}
// Ensures *Constant is a ExternalScalerServer
type ExternalScaler struct{
	logger *Log
}

func NewConstant(logger *Log) *ExternalScaler {
	return &ExternalScaler{
		logger: logger,
	}
}

func getData() (int64, error) {
	conn, err := net.Dial("tcp", "localhost:8001")
	if err != nil {
		fmt.Println("dial error:", err)
		return 0.0, nil
	}
	defer conn.Close()
	fmt.Fprintf(conn, "GET / HTTP/1.0\r\n\r\n")

	buf := make([]byte, 0, 4096) // big buffer
	tmp := make([]byte, 256)     // using small tmo buffer for demonstrating
	for {
		n, err := conn.Read(tmp)
		if err != nil {
			if err != io.EOF {
				fmt.Println("read error:", err)
			}
			break
		}
		//fmt.Println("got", n, "bytes.")
		buf = append(buf, tmp[:n]...)

	}
	val := string(buf)
	parts := strings.SplitN(val, "\r\n\r\n", 2)

	if len(parts) < 2 {
		fmt.Println("Invalid HTTP response format")
		return 0.0, nil
	}
	body := parts[1]

	// Unmarshal the JSON intotmstmp the struct
	var responseData Log
	errr := json.Unmarshal([]byte(body), &responseData)
	if errr != nil {
		fmt.Println("Error unmarshaling JSON:", errr)
		return 0.0, nil
	}

	// Print the struct
	//fmt.Printf("Timestamp: %s\n", responseData.Timestamp)
	//fmt.Printf("Value: %s\n", responseData.Value)
	data, _ := strconv.ParseInt(responseData.Value, 10, 0)
	return data, err
}

func (e *ExternalScaler) IsActive(ctx context.Context, ScaledObject *pb.ScaledObjectRef) (*pb.IsActiveResponse, error) {
	//value := ScaledObject.ScalerMetadata["Value"]
	//timestamp := ScaledObject.ScalerMetadata["Timestamp"]

	value, _ := getData()

	return &pb.IsActiveResponse{
		Result: value > 250,
	}, nil
}

func (e *ExternalScaler) StreamIsActive(ref *pb.ScaledObjectRef, stream pb.ExternalScaler_StreamIsActiveServer) error {
	return nil
}

// GetMetricSpec implements pb.ExternalScalerServer. The target value returned
// is always 1 so that the average gives the desired number of replicas. See
// GetMetrics for details.
func (e *ExternalScaler) GetMetricSpec(ctx context.Context, ref *pb.ScaledObjectRef) (*pb.GetMetricSpecResponse, error) {
	return &pb.GetMetricSpecResponse{
		MetricSpecs: []*pb.MetricSpec{{
			MetricName: "constant_metric",
			TargetSize: 5,
		}},
	}, nil
}

// GetMetrics implements pb.ExternalScalerServer. The metric returned is the
// desired number defined in the ScaledObject.
//
// To illustrate, let's suppose we have set
//
//	desired = 10
//
// in the ScaledObject. GetMetrics is supposed to return the current value,
// which here is always equal to desired. Also, GetMetricSpec returns the target
// value:
//
//	current = 10
//	target = 1
//
// The HPA considers only current and target, and by default is configured to
// use AverageValue. The number of replicas is therefore
//
//	target = (current / replicas)
//
// Then, we solve for replicas
//
//	replicas = (current / target) = (10 / 1) = 10
//
// We end up with 10 replicas, which is the desired number of replicas as per
// the ScaledObject resource.
func (e *ExternalScaler) GetMetrics(ctx context.Context, req *pb.GetMetricsRequest) (*pb.GetMetricsResponse, error) {
	value, err := getData()

	if err != nil {
		err = fmt.Errorf("Error getting desired value: %w", err)
		fmt.Println(err)
		return nil, err
	}

	return &pb.GetMetricsResponse{
		MetricValues: []*pb.MetricValue{{
			MetricName:  "constant_metric",
			MetricValue: value,
		}},
	}, nil
}

