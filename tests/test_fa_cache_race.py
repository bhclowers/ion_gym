"""
A cache store that loses a race to a concurrent writer removes its own
temporary copy (review F9).

store() writes into a hidden temp directory and renames it into place;
when another writer has published the same key first, the rename fails
and the existing entry is served. That branch set ok = True, so the
finally-block skipped removing the temp copy: 4 processes storing 8 keys
left 22 hidden directories (a full field copy each) that entries() never
lists. Here the race is staged by an os.replace that publishes a
competing entry before failing.
"""
import os
import shutil

import numpy as np

from ion_gym.io import fa_cache


def test_a_lost_race_leaves_no_temp_copy(tmp_path, monkeypatch):
    real_replace = os.replace

    def _lose(src, dst):
        if os.path.basename(src).startswith(".") and os.path.isdir(src):
            shutil.copytree(src, dst)         # the other writer wins
            raise OSError(66, "Directory not empty")
        return real_replace(src, dst)
    monkeypatch.setattr(fa_cache.os, "replace", _lose)
    spec = {"name": "race", "geometry": {"w": 1.0}}
    key = fa_cache.store(spec, {"phi": np.arange(12.0).reshape(3, 4)},
                         root=str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [key], names
    arrs, _meta = fa_cache.load(key, root=str(tmp_path), mmap=False)
    assert np.array_equal(arrs["phi"], np.arange(12.0).reshape(3, 4))


def test_load_checks_the_head_hash(tmp_path):
    """Review F5: load's docstring promised a sha_head check it never
    made, so a flipped byte at the start of an array was served."""
    import pytest
    key = fa_cache.store({"name": "hash"}, {"phi": np.arange(64.0)},
                         root=str(tmp_path))
    npy = tmp_path / key / "phi.npy"
    raw = bytearray(npy.read_bytes())
    raw[-64 * 8] ^= 0xFF                     # first byte of the data
    npy.write_bytes(bytes(raw))
    with pytest.raises(IOError, match="head hash mismatch"):
        fa_cache.load(key, root=str(tmp_path))


def test_a_normal_store_publishes_and_cleans(tmp_path):
    key = fa_cache.store({"name": "plain"}, {"phi": np.ones(5)},
                         root=str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == [key]
