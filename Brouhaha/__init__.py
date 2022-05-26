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
    C50_SLIDING_WINDOW = SlidingWindow()  # TODO
    SNR_SLIDING_WINDOW = SlidingWindow()  # TODO

    def samples_loader(self, subset: str):
        data_dir = Path("todo")
        c50_dir = data_dir / "c50"
        snr_dir = data_dir / "snr"

        annotations: Dict[str, Annotation] = load_rttm(str(data_dir / "babytrain.rttm"))
        annotated: Dict[str, Timeline] = load_uem(str(data_dir / "babytrain.uem"))

        for uri, annotation in sorted(annotations.items()):
            c50_array = np.load(str(c50_dir / uri + ".npy"))
            snr_array = np.load(str(snr_dir / uri + ".npy"))
            c50_feat = SlidingWindowFeature()
            snr_feat = SlidingWindowFeature()

            yield {
                # name of the database class
                'database': 'Brouhaha',
                # unique file identifier
                'uri': uri,
                # reference as pyannote.core.Annotation instance
                'annotation': annotation,
                'annotated': annotated[uri],
                'target_features': {}
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
