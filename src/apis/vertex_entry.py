import os
import signal
import sys
import uvicorn
from src.apis.app import app

# Vertex AI specific environment variables
# Vertex AI maps the model directory to AIP_MODEL_DIR
MODEL_DIR = os.environ.get('AIP_MODEL_DIR', '/opt/ml/model')
# Vertex AI provides the port via AIP_HTTP_PORT, default to 8080
PORT = int(os.environ.get('AIP_HTTP_PORT', '8080'))

def handle_sigterm(signum, frame):
    """
    Handle SIGTERM signal for graceful shutdown.
    Vertex AI sends SIGTERM when updating or stopping endpoints.
    """
    print("Received SIGTERM, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    # Start the FastAPI server
    print(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
