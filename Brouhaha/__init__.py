import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from pyannote.core import Annotation, Timeline, SlidingWindow, SlidingWindowFeature
from pyannote.database.util import load_rttm, load_uem

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol


class NoisySpeakerDiarization(SpeakerDiarizationProtocol):
    SNR_SLIDING_WINDOW = SlidingWindow(duration=2.0, step=0.01, start=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_dir: Optional[Path] = None

    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            raise AttributeError("Set data_dir before using the protocol")
        return self._data_dir

    @data_dir.setter
    def data_dir(self, path: Path):
        self._data_dir = path

    def samples_loader(self, subset: str):
        subset_dir = self.data_dir / subset
        snr_dir = subset_dir / "detailed_snr_labels"
        rttm_dir = subset_dir / "rttm_files"

        c50_values: Dict[str, float] = dict()
        # loading c50 values
        with open(subset_dir / "reverb_labels.txt") as reverb_labels_file:
            reader = csv.reader(reverb_labels_file, delimiter=" ")
            for row in reader:
                c50_values[row[0]] = float(row[1])
        # TODO: no annotated?

        for rttm_file in sorted(rttm_dir.iterdir()):
            uri = rttm_file.stem
            annotation = load_rttm(rttm_file)[uri]
            # TODO: maybe use a specific mmap mode
            snr_array = np.load(str(snr_dir / f"{uri}.npy"))
            snr_array = np.expand_dims(snr_array, axis=1)
            snr_feat = SlidingWindowFeature(snr_array, sliding_window=self.SNR_SLIDING_WINDOW)

            yield {
                # name of the database class
                'database': 'Brouhaha',
                # unique file identifier
                'uri': uri,
                # reference as pyannote.core.Annotation instance
                'annotation': annotation,
                'target_features': {
                    "c50": c50_values[uri],
                    "snr": snr_feat
                }
            }

    def trn_iter(self):
        yield from self.samples_loader("train")

    def dev_iter(self):
        yield from self.samples_loader("dev")

    def tst_iter(self):
        yield from self.samples_loader("test")


class Brouhaha(Database):
    """Brouhaha database"""

    def __init__(self, preprocessors: Optional[Dict]):
        super().__init__(preprocessors=preprocessors)

        # register the first protocol: it will be known as
        # Brouhaha.SpeakerDiarization.MyFirstProtocol
        self.register_protocol(
            'SpeakerDiarization', 'NoisySpeakerDiarization', NoisySpeakerDiarization)
