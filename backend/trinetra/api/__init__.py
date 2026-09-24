"""Flask API serving the website: pipelines, dataset metadata, SSE run stream and downloads."""
from flask import Flask
from flask_cors import CORS

from .routes import bp
from .runs import RunManager


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.extensions["runs"] = RunManager()
    app.register_blueprint(bp)
    return app
