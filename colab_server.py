# ============================================================
# SIH26166 — OHRC ↔ NAC / TMC Registration Pipeline Server
# ============================================================
# Entry point only. The code lives in backend/:
#   backend/engine.py              shared registration steps 3-18
#   backend/pipelines/ohrc_nac.py  OHRC ↔ LRO NAC (v4)
#   backend/pipelines/ohrc_tmc.py  OHRC ↔ TMC-2  (v5-tmc)
#   backend/app.py                 Flask API + SSE
#
# Run locally:   LOCAL_MODE=true python colab_server.py
# With ngrok:    NGROK_AUTH_TOKEN=<token> python colab_server.py
# On Colab:      clone/upload this folder, then
#                  %env PROJECT_ROOT=/content/drive/MyDrive/SIH26166
#                  %env NGROK_AUTH_TOKEN=<token>
#                  !python colab_server.py
# ============================================================
from backend.app import main

if __name__ == '__main__':
    main()
