"""Paths and server settings. Override any of them with environment variables."""
import os

# Project root = the folder that holds data/ and outputs/. Defaults to this repo; on Colab set
# PROJECT_ROOT=/content/drive/MyDrive/SIH26166 before starting the server.
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PORT = int(os.environ.get("PORT", "5000"))

# ngrok is only used when LOCAL_MODE is false and a token is set. Never commit the token.
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
