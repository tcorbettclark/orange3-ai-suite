"""The deterministic key ordering behind both validation splits.

:class:`lm.pairs.Pairs.split` and :class:`lora.LoRAData.split` make the same
decision with the same mechanism - a salted content hash of each item's key,
keys ordered ascending, a prefix taken - so that mechanism is spelled once
here. What stays with the callers is their own policy, which genuinely
differs: units vs rows, a fraction that must exclude 0 vs one that allows it,
selection atomic per key vs per row.

A key is the bytes the split should be a function of - a row's contents, or a
unit's ids - and the seed enters as a salt, so re-rolling the seed re-orders
the keys while the key contents still decide. Ordering is by unsigned digest
bytes ascending, and ties keep input order, so the result never depends on
anything beyond ``(keys, seed)``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

_DIGEST_SIZE = 8  # 64-bit key space: more keys than any teaching corpus


def key_order(keys: Iterable[bytes], seed: int) -> list[int]:
    """Indices of ``keys``, ordered ascending by the ``seed``-salted digest.

    Deterministic across runs and callers of the same version; not stable
    across content changes - which is the point: the split is a function of
    the keys and the seed, never of the order the keys arrived in.
    """
    salt = str(seed).encode()
    digests = [hashlib.blake2b(salt + key, digest_size=_DIGEST_SIZE).digest() for key in keys]
    return sorted(range(len(digests)), key=digests.__getitem__)
