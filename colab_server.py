"""Start the Trinetra API server from the repository root; the command the Colab notebook runs.

  Local:  LOCAL_MODE=true python colab_server.py
  ngrok:  NGROK_AUTH_TOKEN=<token> python colab_server.py
  Colab:  %env PROJECT_ROOT=/content/drive/MyDrive/SIH26166
          %env NGROK_AUTH_TOKEN=<token>
          !python colab_server.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from trinetra.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
