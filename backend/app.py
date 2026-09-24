"""Flask API: pipeline discovery, SSE run stream, results and downloads."""
import os, sys, json, time, threading, traceback

import torch
from flask import Flask, Response, request, jsonify, send_file
from flask_cors import CORS

from . import config, events, robust, metadata
from .engine import run_registration
from .pipelines import PIPELINES, DEFAULT_PIPELINE


def run_full_pipeline(pipeline_key, pair_id, run_config):
    events.state["running"] = True
    try:
        mod = PIPELINES[pipeline_key]
        pair = mod.load(pair_id, run_config, config.DATA_RAW, events.emit_event)
        run_registration(pair_id, pair, mod.PROFILE, run_config, events.emit_event,
                         events.pipeline_results, config.OUT_DIR)
    except Exception as e:
        traceback.print_exc()
        events.emit_event(0, "Error", "error", 0, detail=f"{type(e).__name__}: {e}")
    finally:
        events.state["running"] = False


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/connect')
def api_connect():
    gpu = torch.cuda.is_available()
    return jsonify({"status": "ok", "gpu": gpu, "device": "cuda" if gpu else "cpu"})


@app.route('/api/pipelines')
def api_pipelines():
    return jsonify({"default": DEFAULT_PIPELINE, "pipelines": [
        {"id": key, "label": mod.LABEL, "reference": mod.PROFILE.ref_name,
         "version": mod.PROFILE.version, "pairs": mod.available_pairs(config.DATA_RAW)}
        for key, mod in PIPELINES.items()]})


@app.route('/api/pairs')
def api_pairs():
    key = request.args.get('pipeline', DEFAULT_PIPELINE)
    mod = PIPELINES.get(key)
    if mod is None: return jsonify({"error": f"Unknown pipeline '{key}'"}), 404
    return jsonify({"pipeline": key, "pairs": mod.available_pairs(config.DATA_RAW)})


@app.route('/api/pair_info')
def api_pair_info():
    """Sun geometry for the chosen pair, used to pre-fill the website's sun fields."""
    mod = PIPELINES.get(request.args.get('pipeline', DEFAULT_PIPELINE))
    if mod is None: return jsonify({"error": "Unknown pipeline"}), 404
    pair_id = request.args.get('pair_id', '')
    try:
        sun = mod.sun_info(pair_id, config.DATA_RAW)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
    return jsonify({"pipeline": mod.PROFILE.key, "pair_id": pair_id, "reference": mod.PROFILE.ref_name,
                    "ohrc_sun": sun.get("ohrc"), "ref_sun": sun.get("ref"),
                    "options": robust.DEFAULT_OPTIONS})


@app.route('/api/dataset')
def api_dataset():
    """Label metadata of the chosen pair's two products, plus pair-level comparisons."""
    mod = PIPELINES.get(request.args.get('pipeline', DEFAULT_PIPELINE))
    if mod is None: return jsonify({"error": "Unknown pipeline"}), 404
    pair_id = request.args.get('pair_id', '')
    try:
        info = mod.dataset_info(pair_id, config.DATA_RAW)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 400
    return jsonify({"pipeline": mod.PROFILE.key, "pair_id": pair_id, "reference": mod.PROFILE.ref_name,
                    "ohrc": info["ohrc"], "ref": info["ref"],
                    "pair": metadata.compare(info["ohrc"], info["ref"])})


def _run_options(args):
    """Robustness options from the query string. Older clients send only run_ecc."""
    def _bool(name):
        return args.get(name, 'false').lower() == 'true'
    ecc_mode = args.get('ecc_mode') or ('off' if args.get('run_ecc', 'true').lower() == 'false' else 'gated')
    return {'ecc_mode': ecc_mode, 'matcher': args.get('matcher', 'superglue'),
            'shadow_mask': _bool('shadow_mask'), 'tile_rescale': _bool('tile_rescale'),
            'tile_polarity': _bool('tile_polarity'), 'crater_matching': _bool('crater_matching'),
            'auto_retry': _bool('auto_retry')}


def _ref_sun_override(args):
    """Sun values typed on the website (only the fields the user filled in)."""
    def _float(name):
        try: return float(args[name]) if args.get(name, '') != '' else None
        except ValueError: return None
    sun = {'elev_deg': _float('ref_sun_elev'), 'azim_deg': _float('ref_sun_azim'),
           'azim_convention': args.get('ref_sun_conv') or None}
    sun = {k: v for k, v in sun.items() if v is not None}
    return sun or None


@app.route('/api/run')
def api_run():
    if events.state["running"]:
        return jsonify({"error": "Pipeline already running"}), 409

    pipeline_key = request.args.get('pipeline', DEFAULT_PIPELINE)
    if pipeline_key not in PIPELINES:
        return jsonify({"error": f"Unknown pipeline '{pipeline_key}'"}), 404
    pair_id = request.args.get('pair_id', 'pair01_equatorial')
    run_config = {'options': _run_options(request.args), 'ref_sun': _ref_sun_override(request.args)}
    try:
        robust.resolve_options(PIPELINES[pipeline_key].PROFILE.options, run_config['options'])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    events.reset()
    events.state["running"] = True     # set before the thread starts so the stream waits for it
    thread = threading.Thread(target=run_full_pipeline, args=(pipeline_key, pair_id, run_config), daemon=True)
    thread.start()

    def generate():
        sent = 0
        while events.state["running"] or sent < len(events.pipeline_events):
            while sent < len(events.pipeline_events):
                with events.pipeline_lock:
                    event = events.pipeline_events[sent]
                yield f"data: {json.dumps(event, default=str)}\n\n"
                sent += 1
            time.sleep(0.2)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/results')
def api_results():
    return jsonify(events.pipeline_results.get('metrics', {}))


@app.route('/api/results/download/<file_type>')
def api_download(file_type):
    pair_id = request.args.get('pair_id', 'pair01_equatorial')
    mod = PIPELINES.get(request.args.get('pipeline', DEFAULT_PIPELINE))
    if mod is None: return jsonify({"error": "Unknown pipeline"}), 404
    tag = f'{pair_id}_{mod.PROFILE.version}'
    file_map = {
        'metrics': f'{tag}_metrics.json',
        'matches': f'{tag}_matches.csv',
        'report': f'{tag}_REPORT.txt',
        'warped': f'{tag}_OHRC_warped_to_{mod.PROFILE.ref_name}.png',
    }
    filename = file_map.get(file_type)
    if not filename: return jsonify({"error": "Unknown file type"}), 404
    filepath = os.path.join(config.OUT_DIR, filename)
    if not os.path.exists(filepath): return jsonify({"error": "File not yet generated"}), 404
    return send_file(filepath, as_attachment=True)


def main():
    # Windows consoles default to cp1252, which cannot print '↔' in the pipeline labels.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gpu_status = "Available" if torch.cuda.is_available() else "CPU only"
    if not config.LOCAL_MODE and config.NGROK_AUTH_TOKEN:
        from pyngrok import ngrok
        ngrok.set_auth_token(config.NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(config.PORT).public_url
        print("\n" + "=" * 60)
        print("  [OK] SERVER READY!")
        print("  Paste this URL into the website:")
        print(f"     {public_url}")
    else:
        print("\n" + "=" * 60)
        print("  [OK] SERVER READY! (Local Mode)")
        print(f"  Connect your website to: http://localhost:{config.PORT}")
    print(f"  GPU: {gpu_status}")
    print(f"  Data: {config.DATA_RAW}")
    for key, mod in PIPELINES.items():
        print(f"  {mod.LABEL:<18} pairs: {', '.join(mod.available_pairs(config.DATA_RAW)) or '(none)'}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=config.PORT, threaded=True)
