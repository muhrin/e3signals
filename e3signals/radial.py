import math
from typing import Callable

import torch

__all__ = 'FixedCosineRadialModel', 'CosineFunctions', 'FadeAtCutoff', 'OrthonormalRadialFunctions'


def FixedCosineRadialModel(max_radius: float, number_of_basis: int, min_radius=0.):
    radii = torch.linspace(min_radius, max_radius, number_of_basis)
    step = radii[1] - radii[0]

    def radial_function(r):
        shape = r.shape
        radial_shape = [1] * len(shape) + [number_of_basis]
        centers = radii.reshape(*radial_shape)
        return (r.unsqueeze(-1) - centers).div(step).add(1).relu().sub(2).neg().relu().add(1).mul(
            math.pi / 2).cos().pow(2)

    return radial_function


class CosineFunctions:
    def __init__(self, n_radials, cutoff):
        self.n_radials = n_radials
        self.rcut = cutoff

    def __call__(self, r):
        factors = (math.pi / self.rcut) * torch.arange(0, self.n_radials, dtype=r.dtype, device=r.device)
        return torch.cos(torch.outer(r, factors))


class SmoothPolynomials:
    def __init__(self, n_radials, cutoff):
        self.n_radials = n_radials
        self.rcut = cutoff

        normalisations = []
        for alpha in range(1, n_radials + 1):
            normalisations.append((cutoff ** (2 * alpha + 5) / (2 * alpha + 5)) ** 0.5)
        self._normalisations = torch.tensor(normalisations)

    def __call__(self, r):
        values = []
        for alpha in range(1, self.n_radials + 1):
            values.append((self.rcut - r) ** (alpha + 2))

        values = torch.vstack(values).T
        return values / self._normalisations


# Have value and 1st derivative both go to 0 at the cutoff.
class FadeAtCutoff:
    def __init__(self, radial_model: Callable, cutoff: float):
        self.radial_model = radial_model
        self.cutoff = cutoff

    def __call__(self, r):
        f = self.radial_model(r)
        n_radial_functions = f.shape[1]
        smoothing_fn = (self.cutoff - r) * (self.cutoff - r)
        smoothing_fn = torch.outer(smoothing_fn, torch.ones((n_radial_functions,), dtype=r.dtype, device=r.device))
        res = f * smoothing_fn
        res[r >= self.cutoff] = 0.  # Set to zero everything beyond the cutoff
        return res


# Orthonormalizes a set of radial basis functions on a sphere.
#   Uses modified Gram-Schmidt w/trapezoidal integration rule.
#   Tabulates radial basis, returns linear interpolation differentiable in r
class OrthonormalRadialFunctions:
    class OrthoBasis:
        def __init__(self, n_radials: int, radial_model: Callable, cutoff, n_samples: int, dtype, device):
            self.n_radials = n_radials
            self.radial_samples = torch.linspace(0, cutoff, n_samples, dtype=dtype, device=device)
            self.radial_step = self.radial_samples[1] - self.radial_samples[0]

            non_orthogonal_samples = radial_model(self.radial_samples)

            self.area_samples = 4 * math.pi * self.radial_samples * self.radial_samples
            self.f_samples = torch.zeros_like(non_orthogonal_samples, dtype=dtype, device=device)

            u0 = non_orthogonal_samples[:, 0]
            self.f_samples[:, 0] = u0 / self.norm(u0)

            for i in range(1, n_radials):
                ui = non_orthogonal_samples[:, i]
                for j in range(i):
                    uj = self.f_samples[:, j]
                    ui -= self.inner_product(uj, ui) / self.inner_product(uj, uj) * uj
                self.f_samples[:, i] = ui / self.norm(ui)

        def __call__(self, r):
            r_normalized = r / self.radial_step
            r_normalized_floor_int = r_normalized.long()
            indices_low = torch.min(
                torch.max(
                    r_normalized_floor_int, torch.tensor([0], dtype=torch.long, device=r.device)),
                torch.tensor([len(self.radial_samples) - 2], dtype=torch.long, device=r.device)
            )

            r_remainder_normalized = r_normalized - indices_low
            r_remainder_normalized = torch.unsqueeze(r_remainder_normalized, -1)  # add a dimension at the end
            r_remainder_normalized = r_remainder_normalized.expand(
                list(r_remainder_normalized.shape[:-1]) + [self.n_radials])
            # r_remainder_normalized = torch.outer(r_normalized - indicesLow, torch.ones((self.nRadialFunctions,)))

            low_samples = self.f_samples[indices_low, :]
            high_samples = self.f_samples[indices_low + 1, :]

            ret = low_samples * (1 - r_remainder_normalized) + high_samples * r_remainder_normalized

            return ret

        def inner_product(self, a, b):
            return torch.trapz(a * b * self.area_samples, self.radial_samples)

        def norm(self, a):
            return torch.sqrt(self.inner_product(a, a))

    def __init__(self, n_radials: int, radial_model: Callable, cutoff, n_samples: int):
        self.n_radials = n_radials
        self.cutoff = cutoff
        self.radial_model = radial_model
        self.n_samples = n_samples
        self._cache = {}

    def __call__(self, r: torch.tensor):
        return self._get_specific(r.dtype, r.device)(r)

    def _get_specific(self, dtype, device):
        if (dtype, device) not in self._cache:
            self._cache[(dtype, device)] = OrthonormalRadialFunctions.OrthoBasis(
                self.n_radials, self.radial_model, self.cutoff, self.n_samples, dtype, device)

        return self._cache[(dtype, device)]
