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

    @property
    def c50_values(self) -> Dict[str, float]:
        """Loads the C50 (reverb) values"""
        raise  NotImplemented()

    def samples_loader(self, subset: str):
        data_dir = Path("todo")
        c50_dir = data_dir / "c50"
        snr_dir = data_dir / "snr"

        annotations: Dict[str, Annotation] = load_rttm(str(data_dir / "babytrain.rttm"))
        annotated: Dict[str, Timeline] = load_uem(str(data_dir / "babytrain.uem"))

        for uri, annotation in sorted(annotations.items()):
            with open(snr_dir / uri + ".npy") as snr_file:
                csv_reader = csv.reader(snr_file, delimiter=" ")
                snr_array = np.array([row[3] for row in csv_reader])
            snr_array = np.expand_dims(snr_array, axis=1)
            snr_feat = SlidingWindowFeature(snr_array, sliding_window=self.SNR_SLIDING_WINDOW)

            yield {
                # name of the database class
                'database': 'Brouhaha',
                # unique file identifier
                'uri': uri,
                # reference as pyannote.core.Annotation instance
                'annotation': annotation,
                'annotated': annotated[uri],
                'target_features': {
                    "c50": self.c50_values[uri],
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
