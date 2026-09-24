"""Server settings, overridable by environment variables."""
import os

# Folder holding data/ and outputs/: the repository root unless PROJECT_ROOT is set
# (on Colab: PROJECT_ROOT=/content/drive/MyDrive/SIH26166).
PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

PORT = int(os.environ.get("PORT", "5000"))
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")    # ngrok is used only if set and not LOCAL_MODE
