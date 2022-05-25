#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
from typing import Dict, Optional

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

import os.path as op
from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.parser import MDTMParser


class NoisySpeakerDiarization(SpeakerDiarizationProtocol):

    def samples_loader(self, subset: str):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')

        annotations = MDTMParser().read(
            op.join(data_dir, 'protocol1.train.mdtm'))

        for uri in sorted(annotations.uris):
            # get annotations as pyannote.core.Annotation instance
            annotation = annotations(uri)

            yield {
                # name of the database class
                'database': 'Brouhaha',
                # unique file identifier
                'uri': uri,
                # reference as pyannote.core.Annotation instance
                'annotation': annotation,
                'annotated': None,  # TODO from uem
                'target_features': {}
            }

    def trn_iter(self):
        for _ in []:
            yield

    def dev_iter(self):
        for _ in []:
            yield

    def tst_iter(self):
        for _ in []:
            yield


class Brouhaha(Database):
    """Brouhaha database"""

    def __init__(self, preprocessors: Optional[Dict]):
        super().__init__(preprocessors=preprocessors)

        # register the first protocol: it will be known as
        # Brouhaha.SpeakerDiarization.MyFirstProtocol
        self.register_protocol(
            'SpeakerDiarization', 'NoisySpeakerDiarization', NoisySpeakerDiarization)
