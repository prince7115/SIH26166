"""HTTP endpoints used by the website."""
import os

import torch
from flask import Blueprint, Response, current_app, jsonify, request, send_file

from . import params
from .. import config
from ..products import metadata
from ..registration.options import resolve_options
from ..registration.runner import run_registration
from ..sensors import DEFAULT_SENSOR, SENSORS

bp = Blueprint("api", __name__, url_prefix="/api")

DOWNLOADS = {"metrics": "{tag}_metrics.json"}


class ApiError(Exception):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


@bp.errorhandler(ApiError)
def _api_error(e):
    return jsonify({"error": str(e)}), e.status


def _sensor():
    key = request.args.get("pipeline", DEFAULT_SENSOR)
    sensor = SENSORS.get(key)
    if sensor is None:
        raise ApiError(f"Unknown pipeline '{key}'", 404)
    return sensor


def _sensor_and_pair():
    """The requested pipeline and one of its available pair ids (never a path from the client)."""
    sensor = _sensor()
    pair_id = request.args.get("pair_id", "")
    if pair_id not in sensor.available_pairs(config.DATA_RAW):
        raise ApiError(f"Unknown pair '{pair_id}' for {sensor.PROFILE.key}", 404)
    return sensor, pair_id


def _read_labels(fn, pair_id):
    try:
        return fn(pair_id, config.DATA_RAW)
    except Exception as e:
        raise ApiError(f"{type(e).__name__}: {e}", 400) from e


@bp.route("/connect")
def connect():
    gpu = torch.cuda.is_available()
    return jsonify({"status": "ok", "gpu": gpu, "device": "cuda" if gpu else "cpu"})


@bp.route("/pipelines")
def pipelines():
    return jsonify({"default": DEFAULT_SENSOR, "pipelines": [
        {"id": key, "label": s.LABEL, "reference": s.PROFILE.ref_name, "version": s.PROFILE.version,
         "pairs": s.available_pairs(config.DATA_RAW)}
        for key, s in SENSORS.items()]})


@bp.route("/pair_info")
def pair_info():
    """Sun geometry of the chosen pair, used to pre-fill the website's sun fields."""
    sensor, pair_id = _sensor_and_pair()
    sun = _read_labels(sensor.sun_info, pair_id)
    return jsonify({"pipeline": sensor.PROFILE.key, "pair_id": pair_id, "reference": sensor.PROFILE.ref_name,
                    "ohrc_sun": sun.get("ohrc"), "ref_sun": sun.get("ref")})


@bp.route("/dataset")
def dataset():
    """Label metadata of both products and pair-level comparisons."""
    sensor, pair_id = _sensor_and_pair()
    info = _read_labels(sensor.dataset_info, pair_id)
    return jsonify({"pipeline": sensor.PROFILE.key, "pair_id": pair_id, "reference": sensor.PROFILE.ref_name,
                    "ohrc": info["ohrc"], "ref": info["ref"], "pair": metadata.compare(info["ohrc"], info["ref"])})


@bp.route("/run")
def run():
    """Start a registration and stream its progress as server-sent events."""
    sensor, pair_id = _sensor_and_pair()
    options = params.run_options(request.args)
    ref_sun = params.sun_override(request.args)
    try:
        resolve_options(sensor.PROFILE.options, options)
    except ValueError as e:
        raise ApiError(str(e), 400) from e

    def job(emit):
        pair = sensor.load(pair_id, config.DATA_RAW, emit, ref_sun)
        run_registration(pair_id, pair, sensor.PROFILE, options, emit, config.OUT_DIR)

    runs = current_app.extensions["runs"]
    if not runs.start(job):
        raise ApiError("Pipeline already running", 409)
    return Response(runs.stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.route("/results/download/<file_type>")
def download(file_type):
    sensor, pair_id = _sensor_and_pair()
    pattern = DOWNLOADS.get(file_type)
    if pattern is None:
        raise ApiError("Unknown file type", 404)
    path = os.path.join(config.OUT_DIR, pattern.format(tag=sensor.PROFILE.output_tag(pair_id)))
    if not os.path.exists(path):
        raise ApiError("File not yet generated", 404)
    return send_file(path, as_attachment=True)
