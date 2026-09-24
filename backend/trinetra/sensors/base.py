"""The interface every sensor-pair pipeline module implements."""
from typing import Callable, Protocol

from ..registration.models import LoadedPair, Profile


class SensorPipeline(Protocol):
    PROFILE: Profile
    LABEL: str          # shown in the website's sensor-pair menu

    def available_pairs(self, data_raw: str) -> list[str]:
        """Pair ids with both products on disk."""

    def sun_info(self, pair_id: str, data_raw: str) -> dict:
        """{"ohrc": sun record, "ref": sun record} used to pre-fill the website's sun fields."""

    def dataset_info(self, pair_id: str, data_raw: str) -> dict:
        """{"ohrc": metadata record, "ref": metadata record} from the labels only."""

    def load(self, pair_id: str, data_raw: str, emit: Callable, ref_sun_override: dict = None) -> LoadedPair:
        """Steps 1-2: read both products."""
