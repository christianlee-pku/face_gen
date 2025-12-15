import subprocess
import time
import requests
import os
import sys

def verify_local_vertex_entry():
    print("--- Starting Local Vertex AI Entrypoint Verification ---")
    
    # 1. Configuration
    port = 8081
    env = os.environ.copy()
    env["AIP_HTTP_PORT"] = str(port)
    env["PYTHONPATH"] = os.getcwd() # Ensure src/ can be imported
    
    process = None
    try:
        # 2. Start the server as a subprocess
        print(f"Starting server on port {port}...")
        process = subprocess.Popen(
            [sys.executable, "src/apis/vertex_entry.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it time to start
        print("Waiting for server to start (5s)...")
        time.sleep(5)
        
        # Check if process is still alive
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("Server failed to start!")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return

        # 3. Test the Endpoint
        url = f"http://localhost:{port}/health"
        print(f"Sending GET request to {url}...")
        
        try:
            response = requests.get(url)
            print(f"Status Code: {response.status_code}")
            print(f"Response JSON: {response.json()}")
            
            if response.status_code == 200 and response.json().get("status") == "healthy":
                print("\nSUCCESS: Local Vertex AI entrypoint is working correctly!")
            else:
                print("\nFAILURE: Unexpected response.")
                
        except requests.exceptions.ConnectionError:
            print(f"\nFAILURE: Could not connect to {url}. Server might not be running.")

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # 4. Cleanup
        if process:
            print("Stopping server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("Server stopped.")

if __name__ == "__main__":
    verify_local_vertex_entry()
