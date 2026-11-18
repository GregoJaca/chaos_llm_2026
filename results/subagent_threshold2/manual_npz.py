"""Strict reader for npz files whose zip central directory is damaged.

Scans local file headers (PK\x03\x04), reads each member's declared bytes,
verifies the .npy magic and the claimed array shape/dtype, and returns only
members that are byte-complete. Falls back gracefully: callers can check
which keys survived.
"""
import io
import os
import struct

import numpy as np


def read_npz_members(path):
    """Return dict key -> np.ndarray for every byte-complete member."""
    out = {}
    fs = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if hdr != b"PK\x03\x04":
                break
            ver, flags, method = struct.unpack("<HHH", f.read(6))
            f.read(4)  # time, date
            crc, csize, usize = struct.unpack("<III", f.read(12))
            nlen, elen = struct.unpack("<HH", f.read(4))
            name = f.read(nlen).decode("utf-8", "replace")
            extra = f.read(elen)
            if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
                csize, usize = struct.unpack("<QQ", extra[4:20])
            data = f.read(csize)
            if len(data) != csize:  # truncated member: stop here
                break
            if method != 0:
                raise NotImplementedError(f"member {name} is compressed; not handled")
            key = name[:-4] if name.endswith(".npy") else name
            if data[:6] == b"\x93NUMPY":
                try:
                    out[key] = np.load(io.BytesIO(data), allow_pickle=True)
                except Exception:
                    pass
            if flags & 0x08:  # data descriptor (crc/csize repeated after data)
                f.read(16)
    return out


def read_npz(path):
    """np.load if possible, else the strict manual reader."""
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return read_npz_members(path)


if __name__ == "__main__":
    root = r"C:\Users\hjaca\research\chaos_llm_2026\results"
    for name in ["a1_ds_w100_large", "a1_ds_w100", "a1_cyber_w100"]:
        p = os.path.join(root, name, "a1.npz")
        d = read_npz_members(p)
        print(f"== {name}  ({os.path.getsize(p)} bytes)")
        for k, v in d.items():
            print(f"   {k}: shape={v.shape} dtype={v.dtype}")
