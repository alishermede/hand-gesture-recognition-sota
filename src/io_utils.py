"""Image I/O that works under the Cyrillic project path.

cv2.imread / cv2.imwrite cannot open non-ASCII paths on Windows, so we read and
write the raw bytes ourselves and let OpenCV only encode / decode.
"""

import os

import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise IOError(f"could not read {path}")
    return img


def imwrite_unicode(path, img, params=None):
    ok, buf = cv2.imencode(os.path.splitext(str(path))[1], img, params or [])
    if not ok:
        raise IOError(f"failed to encode {path}")
    buf.tofile(str(path))


def load_rgb(path):
    return cv2.cvtColor(imread_unicode(path), cv2.COLOR_BGR2RGB)
