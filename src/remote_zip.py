"""Read members of a remote .zip over HTTP range requests, without downloading the whole archive.

The HaGRID sample is a single 1 GB zip with 19 gesture classes; we need 3 of them.
zipfile only needs a seekable file object, so we give it one backed by Range requests.
"""

import io
import time
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, retries=5):
        self.url, self.retries, self.pos = url, retries, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()  # follow the CDN redirect once
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)
