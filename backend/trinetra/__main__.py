"""Start the API server: python -m trinetra (from the backend/ folder).

Local:  LOCAL_MODE=true python -m trinetra
ngrok:  NGROK_AUTH_TOKEN=<token> python -m trinetra
"""
import sys

import torch

from . import config
from .api import create_app
from .sensors import SENSORS


def main():
    # Windows consoles default to cp1252, which cannot print the '↔' in pipeline labels.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    app = create_app()

    print("\n" + "=" * 60)
    if not config.LOCAL_MODE and config.NGROK_AUTH_TOKEN:
        from pyngrok import ngrok
        ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
        print("  [OK] SERVER READY!")
        print("  Paste this URL into the website:")
        print(f"     {ngrok.connect(config.PORT).public_url}")
    else:
        print("  [OK] SERVER READY! (Local Mode)")
        print(f"  Connect your website to: http://localhost:{config.PORT}")
    print(f"  GPU: {'Available' if torch.cuda.is_available() else 'CPU only'}")
    print(f"  Data: {config.DATA_RAW}")
    for sensor in SENSORS.values():
        print(f"  {sensor.LABEL:<18} pairs: {', '.join(sensor.available_pairs(config.DATA_RAW)) or '(none)'}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
