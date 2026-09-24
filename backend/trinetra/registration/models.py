"""What a sensor pipeline hands to the registration: its profile and the loaded pair."""
from dataclasses import dataclass, field


@dataclass
class Profile:
    """The settings that differ between sensor pipelines once both products are loaded."""
    key: str                               # "ohrc_nac" / "ohrc_tmc"
    ref_name: str                          # "NAC" / "TMC": used in step names and metrics
    version: str                           # output file tag: "v4" / "v5"
    fine_gsd_multiplier: float = 1.0       # fine GSD = coarser native GSD x this
    antialiased_resampling: bool = False   # level_view_antialiased instead of level_view
    adaptive_tile: bool = False            # tile size from the OHRC overlap width instead of 640
    options: dict = field(default_factory=dict)   # pipeline defaults over DEFAULT_OPTIONS

    def output_tag(self, pair_id):
        return f"{pair_id}_{self.version}"


@dataclass
class LoadedPair:
    """Steps 1-2 output: both rasters with their footprints and native GSDs."""
    ohrc_raster: object
    ohrc_corners: dict
    ohrc_shape: tuple
    ohrc_gsd: float
    ohrc_product_id: str
    ref_raster: object
    ref_corners: dict
    ref_shape: tuple
    ref_gsd: float
    ref_product_id: str
    extra_metrics: dict = field(default_factory=dict)
    ohrc_sun: dict = None                  # products.sun records, used for the difficulty score
    ref_sun: dict = None
