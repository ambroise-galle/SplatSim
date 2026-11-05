"""Lecture de fichiers PCD (v0.7) : ascii, binary et binary_compressed (LZF, champs rangés par colonne).

Utilisé pour les cartes de référence de DiTer++ (`Lawn_GT.pcd` : binary_compressed) et d'Oxford Spires.
Dépendance pour binary_compressed : `python-lzf`.
"""

from __future__ import annotations

import numpy as np

_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
          ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}


def read_pcd(path: str) -> dict:
    """Renvoie {nom de champ: tableau (N,) ou (N, count)} et la clé spéciale `_header`."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, *v = line.split()
            header[k.upper()] = v
            if k.upper() == "DATA":
                break
        raw = f.read()
    names = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
    n = int(header["POINTS"][0])
    fmt = header["DATA"][0]
    dts = [np.dtype(_TYPES[(t, s)]) for t, s in zip(types, sizes)]
    out = {"_header": header}
    if fmt == "binary":
        dt = np.dtype([(nm, d, (c,)) if c > 1 else (nm, d) for nm, d, c in zip(names, dts, counts)])
        arr = np.frombuffer(raw, dtype=dt, count=n)
        for nm in names:
            out[nm] = arr[nm].copy()
    elif fmt == "binary_compressed":
        import lzf

        csize, usize = np.frombuffer(raw[:8], "<u4")
        data = lzf.decompress(raw[8:8 + int(csize)], int(usize))
        off = 0
        for nm, d, c in zip(names, dts, counts):          # stockage par colonne (SoA)
            nb = n * c * d.itemsize
            a = np.frombuffer(data, dtype=d, count=n * c, offset=off)
            out[nm] = a.reshape(n, c).copy() if c > 1 else a.copy()
            off += nb
    elif fmt == "ascii":
        a = np.loadtxt(raw.decode("ascii").splitlines(), ndmin=2)
        col = 0
        for nm, d, c in zip(names, dts, counts):
            out[nm] = a[:, col:col + c].astype(d).squeeze()
            col += c
    else:
        raise ValueError(f"format PCD non géré : {fmt}")
    return out


def rgb_from_float(rgbf: np.ndarray) -> np.ndarray:
    """Champ `rgb` PCL (float32 contenant 0x00RRGGBB) → uint8 (N, 3)."""
    u = np.ascontiguousarray(rgbf, dtype=np.float32).view(np.uint32)
    return np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.uint8)
