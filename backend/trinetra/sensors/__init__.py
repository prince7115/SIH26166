"""Sensor-pair pipelines, keyed by profile key. Each module implements base.SensorPipeline."""
from . import ohrc_nac, ohrc_tmc

SENSORS = {m.PROFILE.key: m for m in (ohrc_nac, ohrc_tmc)}
DEFAULT_SENSOR = ohrc_nac.PROFILE.key
