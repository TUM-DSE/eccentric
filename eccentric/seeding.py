"""Seeds that are the same in every process (Python's hash() is not), so that a configuration compiles to
the same circuit wherever and whenever it runs"""
import zlib


def stable_seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())
