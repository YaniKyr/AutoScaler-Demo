package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "google.golang.org/grpc"
    pb "github.com/YaniKyr/5G_Autoscaling_Notes/New/testfolder/proto"
)

type Log struct {
	Timestamp string `json:"timestamp"`
	Value     float64 `json:"value"`
}

func fetchPrometheusData(client pb.PrometheusServiceClient, query string) {
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()

    req := &pb.PrometheusRequest{Query: query}
    res, err := client.GetPrometheusData(ctx, req)
    if err != nil {
        log.Fatalf("could not fetch data: %v", err)
    }
    fmt.Printf("Prometheus Data: %s\n", res.Data)
}

// Fetch data from the service and convert to int64
func main()  {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure(), grpc.WithBlock())
    if err != nil {
        log.Fatalf("did not connect: %v", err)
    }
    defer conn.Close()

    client := pb.NewPrometheusServiceClient(conn)

    for {
        fetchPrometheusData(client, `up{job="prometheus"}`)
        time.Sleep(5 * time.Second)
    }
    
}
	