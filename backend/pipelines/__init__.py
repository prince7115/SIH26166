"""Pipeline registry. To add a reference sensor, add a module exposing PROFILE, LABEL,
REF_KIND, available_pairs(data_raw) and load(pair_id, config, data_raw, emit_event), and
list it here."""
from . import ohrc_nac, ohrc_tmc

PIPELINES = {m.PROFILE.key: m for m in (ohrc_nac, ohrc_tmc)}
DEFAULT_PIPELINE = ohrc_nac.PROFILE.key
