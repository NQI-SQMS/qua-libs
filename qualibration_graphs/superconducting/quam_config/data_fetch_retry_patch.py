"""Patch `XarrayDataFetcher.retrieve_latest_data` to retry on transient QM fetch races.

`XarrayDataFetcher.__iter__` (qualibration_libs.data.fetcher) polls
`job.result_handles.is_processing()` and then immediately calls `fetch_all()` to pull
whatever has been streamed so far. Between those two calls the OPX can still be mid-write
on a stream item (this is especially likely for composite/tuple saves, e.g. per-sequence
RB results), so the fetch sometimes reads a byte count that isn't an integer multiple of
the item size. `qm-qua` surfaces this as:

    qm.exceptions.DataFetchingError: Fetched items count is not a round number: <float>

This is a transient race in the QM SDK's live-fetch path, not a bug in the acquired data
or in this project's code — confirmed by it disappearing on an immediate retry a few
milliseconds later, once the in-flight write finishes.

Importing this module (done once, from quam_config/__init__.py) wraps
`retrieve_latest_data` so every calibration node's live data-fetching loop automatically
retries a few times with a short backoff instead of aborting the run.
"""
import time
import logging

from qm.exceptions import DataFetchingError
from qualibration_libs.data.fetcher import XarrayDataFetcher

logger = logging.getLogger(__name__)

_original_retrieve_latest_data = XarrayDataFetcher.retrieve_latest_data

_MAX_RETRIES = 5
_RETRY_DELAY_SECONDS = 0.1


def _retrieve_latest_data_with_retry(self):
    for attempt in range(_MAX_RETRIES + 1):
        try:
            _original_retrieve_latest_data(self)
            return
        except DataFetchingError:
            if attempt == _MAX_RETRIES:
                raise
            logger.debug(
                f"Transient DataFetchingError on fetch attempt {attempt + 1}/{_MAX_RETRIES}; retrying."
            )
            time.sleep(_RETRY_DELAY_SECONDS)


XarrayDataFetcher.retrieve_latest_data = _retrieve_latest_data_with_retry
