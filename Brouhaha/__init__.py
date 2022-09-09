import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from librosa import get_duration
from pyannote.core import Annotation, Timeline, SlidingWindow, SlidingWindowFeature
from pyannote.core.segment import Segment
from pyannote.database.util import load_rttm, load_uem

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol


C50_NO_REVERB = 60
SNR_MIN = -15
SNR_MAX = 80

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
        audio_dir = subset_dir / "audio_16k"

        c50_values: Dict[str, float] = dict()
        # loading c50 values
        with open(subset_dir / "reverb_labels.txt") as reverb_labels_file:
            reader = csv.reader(reverb_labels_file, delimiter=" ")
            for row in reader:
                c50_value = row[1]
                if c50_value == 'None':
                    c50_value = C50_NO_REVERB
                c50_values[row[0]] = float(c50_value)
        # TODO: no annotated?

        for rttm_file in sorted(rttm_dir.iterdir()):
            uri = rttm_file.stem
            annotation = load_rttm(rttm_file)[uri]
            audio_file = str(audio_dir / uri) + '.flac'
            end = get_duration(filename = audio_file)
            annotated = Timeline([ Segment(start=0, end=end)])
            # TODO: maybe use a specific mmap mode
            snr_array = np.load(str(snr_dir / f"{uri}_snr.npy"))
            print(f"All snr values will be brought back to [{SNR_MIN},{SNR_MAX}]")
            snr_array = np.where(snr_array < SNR_MIN, SNR_MIN, snr_array)
            snr_array = np.expand_dims(snr_array, axis=1)
            snr_feat = SlidingWindowFeature(snr_array, sliding_window=self.SNR_SLIDING_WINDOW)

            yield {
                # name of the database class
                'database': 'Brouhaha',
                # unique file identifier
                'uri': uri,
                # reference as pyannote.core.Annotation instance
                'annotation': annotation,
                'annotated': annotated,
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

    def __init__(self, preprocessors={}, **kwargs):
        super().__init__(preprocessors=preprocessors, **kwargs)

        # register the first protocol: it will be known as
        # Brouhaha.SpeakerDiarization.MyFirstProtocol
        self.register_protocol(
            'SpeakerDiarization', 'NoisySpeakerDiarization', NoisySpeakerDiarization)
