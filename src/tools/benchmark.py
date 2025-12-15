import time
import requests
import numpy as np
import argparse
import os
import subprocess
import signal
import sys

def benchmark_latency(url, num_requests=100, auth_token=None):
    """
    Benchmark the latency of the face generation endpoint.
    """
    print(f"Benchmarking {url} with {num_requests} requests...")
    latencies = []
    
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    payload = {"seed": 42, "truncation_psi": 0.7}
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        try:
            requests.post(url, json=payload, headers=headers)
        except:
            pass
            
    print("Running benchmark...")
    for i in range(num_requests):
        start = time.time()
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                latencies.append(time.time() - start)
            else:
                print(f"Request {i} failed: {response.status_code}")
        except Exception as e:
            print(f"Request {i} error: {e}")
            
    if not latencies:
        print("All requests failed.")
        return
    
    latencies = np.array(latencies) * 1000 # convert to ms
    
    print("\n--- Results (ms) ---")
    print(f"Min: {latencies.min():.2f}")
    print(f"Max: {latencies.max():.2f}")
    print(f"Mean: {latencies.mean():.2f}")
    print(f"P50: {np.percentile(latencies, 50):.2f}")
    print(f"P95: {np.percentile(latencies, 95):.2f}")
    print(f"P99: {np.percentile(latencies, 99):.2f}")
    print("--------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080/generate", help="Endpoint URL")
    parser.add_argument("--n", type=int, default=50, help="Number of requests")
    parser.add_argument("--local", action="store_true", help="Spin up local server for testing")
    args = parser.parse_args()
    
    server_process = None
    
    if args.local:
        print("Starting local server for benchmarking...")
        env = os.environ.copy()
        env["MODEL_PATH"] = "work_dirs/stylegan3.onnx"
        env["API_AUTH_TOKEN"] = "bench-token"
        env["AIP_HTTP_PORT"] = "8080"
        env["PYTHONPATH"] = os.getcwd()
        
        # Ensure model exists (dummy)
        if not os.path.exists(env["MODEL_PATH"]):
             subprocess.run([sys.executable, "src/tools/create_dummy_checkpoint.py"], check=True)
             subprocess.run([sys.executable, "src/tools/export.py", "configs/stylegan3_celeba.py", "work_dirs/stylegan3_celeba/epoch_5.pth", "--out", env["MODEL_PATH"]], check=True)

        server_process = subprocess.Popen(
            [sys.executable, "src/apis/vertex_entry.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Wait for server
        time.sleep(5)
        
        # Override URL and Token
        url = "http://localhost:8080/generate"
        token = "bench-token"
    else:
        url = args.url
        token = os.environ.get("API_AUTH_TOKEN")

    try:
        benchmark_latency(url, args.n, token)
    finally:
        if server_process:
            print("Stopping local server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    main()
