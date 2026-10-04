#!/usr/bin/env python3
"""Generate a .binadd file for the MC NPU binary import path.

Default is the 9x9 chunk case: 81 chunks x 98304 elements = 7,962,624 floats for
a and the same for b, which is 63.7 MB on disk.

The file is written in chunks so memory stays flat regardless of n: building an
8M element Python list costs hundreds of MB, and that is the difference between
this finishing in seconds and being killed.

The value formulas match NpuBigAdd.selfTest and the JSON generator, so a result
here is directly comparable with the earlier JSON sweep.

Usage:
    python3 gen9x9_bin.py                          # 9x9, 63.7 MB
    python3 gen9x9_bin.py --chunks 4 --out x.binadd
"""

import argparse
import array
import os
import struct
import sys
import zlib

MAGIC = 0x424E4144          # "BNAD"
VERSION = 1
OP_ADD = 1
HEADER_BYTES = 32

CHUNK_SIZE = 98304          # MC 26.3: 16 x 16 x 384
DEFAULT_WAY = 16384         # largest ADD shape the HTP graph accepts


def build_block(n, fn, chunk=1 << 16):
    """Yield little-endian float32 bytes for fn(0..n-1), `chunk` values at a time."""
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        vals = array.array("f", (fn(i) for i in range(start, stop)))
        if sys.byteorder != "little":
            vals.byteswap()
        yield vals.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=81, help="chunk count (81 = 9x9)")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--way", type=int, default=DEFAULT_WAY, help="way shape written into the header")
    ap.add_argument("--par", type=int, default=4, help="parallelism written into the header")
    ap.add_argument("--op", type=int, default=OP_ADD)
    ap.add_argument("--out", default="MC_NPU_ADD_9x9.binadd")
    args = ap.parse_args()

    n = args.chunks * args.chunk_size
    way = min(max(args.way, 1), DEFAULT_WAY)
    ways = (n + way - 1) // way
    per_call = max(1, min(64, (2 * 1024 * 1024) // (8 * way)))

    a_fn = lambda i: ((i % 97) - 48) / 32.0
    b_fn = lambda i: ((i % 53) + 1) / 64.0

    crc = 0
    written = 0
    with open(args.out, "wb") as f:
        # Checksum is written last: it covers the body, which does not exist yet.
        # 9 fields = 32 bytes: magic I, version H, op H, then n, case_count,
        # way_elements, parallelism, reserved, checksum as six I. An 8-field
        # pack is 28 bytes and silently puts the body where the checksum goes.
        f.write(struct.pack("<IHHIIIIII",
                            MAGIC, VERSION, args.op,
                            n, 1, way, args.par, 0, 0))
        assert f.tell() == HEADER_BYTES, f.tell()
        for block in (build_block(n, a_fn), build_block(n, b_fn)):
            for piece in block:
                crc = zlib.crc32(piece, crc)
                f.write(piece)
                written += len(piece)
        f.seek(28)
        f.write(struct.pack("<I", crc & 0xFFFFFFFF))

    size = os.path.getsize(args.out)
    print("saved: {}  size={:.1f} MB  n={}  ways={}  way={}  par={}".format(
        args.out, size / 1048576.0, n, ways, way, args.par))
    print("crc32={:08x}  body={} bytes  expected_body={}".format(
        crc & 0xFFFFFFFF, written, 8 * n))
    print("npu plan: {} ways, <= {} cases per IPC call -> ~{} calls, body <= {:.1f} MB each".format(
        ways, per_call, (ways + per_call - 1) // per_call,
        8.0 * way * per_call / 1048576.0))
    expect = HEADER_BYTES + 8 * n
    if size != expect:
        print("ERROR: size {} != expected {}".format(size, expect))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
