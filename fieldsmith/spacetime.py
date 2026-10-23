"""Spacetime: coordinates, metric, and curvature.

Conventions
-----------
* ``mostly_minus`` (default): signature (+,-,-,-), the particle-physics convention.
* ``mostly_plus``: signature (-,+,+,+), the relativity convention.
* ``euclidean``: positive-definite metric (used for instantons, statics).

The Riemann tensor follows MTW:
    R^r_{s m n} = d_m G^r_{n s} - d_n G^r_{m s} + G^r_{m l} G^l_{n s} - G^r_{n l} G^l_{m s}
"""
from __future__ import annotations

import sympy as sp


class Spacetime:
    def __init__(self, coords, g, ginv=None, sqrt_g=None,
                 signature: str = "mostly_minus", name: str = "spacetime"):
        self.coords = list(coords)
        self.dim = len(self.coords)
        self.g = sp.Matrix(g)
        self.signature = signature
        self.name = name
        self.ginv = sp.Matrix(ginv) if ginv is not None else sp.simplify(self.g.inv())
        if sqrt_g is None:
            det = sp.simplify(self.g.det())
            sqrt_g = sp.sqrt(det if signature == "euclidean" else -det)
        self.sqrt_g = sp.simplify(sqrt_g)
        # +1: L = +1/2 (d phi)^2 - V ;  -1: L = -1/2 (d phi)^2 - V
        self.sign = -1 if signature == "mostly_plus" else 1
        self._christoffel = None

    # ------------------------------------------------------------------ factories
    @classmethod
    def minkowski(cls, dim: int = 4, signature: str = "mostly_minus", coords=None):
        if coords is None:
            names = {2: "t x", 3: "t x y", 4: "t x y z"}.get(dim)
            names = names or " ".join(["t"] + [f"x{i}" for i in range(1, dim)])
            coords = sp.symbols(names, real=True)
        if signature == "mostly_minus":
            diag = [1] + [-1] * (dim - 1)
        elif signature == "mostly_plus":
            diag = [-1] + [1] * (dim - 1)
        else:
            raise ValueError("signature must be 'mostly_minus' or 'mostly_plus'")
        g = sp.diag(*diag)
        return cls(coords, g, ginv=g, sqrt_g=sp.S.One, signature=signature,
                   name=f"Minkowski{dim}")

    @classmethod
    def euclidean(cls, dim: int = 4, coords=None):
        if coords is None:
            coords = sp.symbols(" ".join(f"x{i}" for i in range(1, dim + 1)), real=True)
        g = sp.eye(dim)
        return cls(coords, g, ginv=g, sqrt_g=sp.S.One, signature="euclidean",
                   name=f"Euclid{dim}")

    # ------------------------------------------------------------------ properties
    @property
    def is_flat_cartesian(self) -> bool:
        return (self.g.is_diagonal() and all(abs(self.g[i, i]) == 1 for i in range(self.dim))
                and not any(self.g[i, i].free_symbols for i in range(self.dim)))

    @property
    def n_space(self) -> int:
        return self.dim - 1

    def index(self, coord) -> int:
        """Position of a coordinate (symbol or name) in ``coords``."""
        for i, c in enumerate(self.coords):
            if c == coord or c.name == str(coord):
                return i
        raise KeyError(f"unknown coordinate {coord!r}")

    # ------------------------------------------------------------------ curvature
    def christoffel(self):
        """Gamma[l][m][n] = Gamma^l_{mn}."""
        if self._christoffel is None:
            d, g, gi, X = self.dim, self.g, self.ginv, self.coords
            G = [[[0] * d for _ in range(d)] for _ in range(d)]
            for l in range(d):
                for m in range(d):
                    for n in range(d):
                        G[l][m][n] = sp.simplify(sum(
                            gi[l, s] * (sp.diff(g[s, n], X[m]) + sp.diff(g[s, m], X[n])
                                        - sp.diff(g[m, n], X[s])) for s in range(d)) / 2)
            self._christoffel = G
        return self._christoffel

    def riemann(self):
        """R[r][s][m][n] = R^r_{s m n} (MTW convention)."""
        d, G, X = self.dim, self.christoffel(), self.coords
        R = [[[[0] * d for _ in range(d)] for _ in range(d)] for _ in range(d)]
        for r in range(d):
            for s in range(d):
                for m in range(d):
                    for n in range(d):
                        R[r][s][m][n] = sp.simplify(
                            sp.diff(G[r][n][s], X[m]) - sp.diff(G[r][m][s], X[n])
                            + sum(G[r][m][l] * G[l][n][s] - G[r][n][l] * G[l][m][s]
                                  for l in range(d)))
        return R

    def ricci(self):
        R, d = self.riemann(), self.dim
        return sp.Matrix(d, d, lambda s, n: sp.simplify(sum(R[r][s][r][n] for r in range(d))))

    def ricci_scalar(self):
        Ric = self.ricci()
        return sp.simplify(sum(self.ginv[i, j] * Ric[i, j]
                               for i in range(self.dim) for j in range(self.dim)))

    def einstein(self):
        """Einstein tensor G_{mn} = R_{mn} - 1/2 g_{mn} R."""
        return (self.ricci() - self.g * self.ricci_scalar() / 2).applyfunc(sp.simplify)

    def __repr__(self):
        return f"Spacetime({self.name}, coords={self.coords})"
