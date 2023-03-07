from typing import Callable

import math
import torch

from e3nn import o3

__all__ = ('RadialSphericalTensor',)


class RadialSphericalTensor(o3.Irreps):
    r"""representation of a signal in 3-space or in a solid ball

    A 'RadialSphericalTensor' contains the coefficients of a function expansion in 3-space, potentially compactly
    supported on a solid ball.
    Each coefficient corresponds to a single basis function; each basis function is the product of a radial basis
    function and a single spherical harmonic.

    Arguments:

    :param num_radials: int>0, number of radial basis functions
    :param orthonormal_radial_basis: a function or functional that accepts a vector of nR>0 radii,
        and returns an array of shape (nR,nRadialBases) containing the values of
        the orthonormal radial basis functions.
    :param lmax: int, the maximum degree of spherical harmonic functions
    p_val, p_arg: same as in SphericalTensor
    """

    # pylint: disable=signature-differs
    def __new__(
            cls,
            num_radials: int,
            orthonormal_radial_basis: Callable,
            lmax: int,
            p_val,
            p_arg
    ) -> 'RadialSphericalTensor':
        cls.num_radials = num_radials
        cls.radial_basis = orthonormal_radial_basis
        cls.lmax = lmax
        cls.p_val = p_val
        cls.p_arg = p_arg

        multiplicities = [num_radials] * (lmax + 1)

        radial_selector = []
        for l in range(lmax + 1):
            for i_radial in range(num_radials):
                for _m in range(2 * l + 1):
                    radial_selector.append(i_radial)

        cls.radial_selector = torch.tensor(radial_selector)

        parities = {l: (p_val * p_arg ** l) for l in range(lmax + 1)}

        irreps = [(multiplicity, (l, parities[l])) for multiplicity, l in zip(multiplicities, range(lmax + 1))]
        ret = super().__new__(cls, irreps)

        return ret

    def _evaluate_angular_basis(self, vectors, radii=None):
        """Evaluate angular basis functions (spherical harmonics) at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """
        if self[0][1].p != 1:  # pylint: disable=no-member
            raise ValueError(
                "the spherical harmonics are only evaluable when p_val is 1, since the l=0 must have parity 1.")

        if radii is not None:
            angular_coeffs = o3.spherical_harmonics(
                self,
                vectors.view(-1, 3) / radii.view(-1, 1).expand(-1, 3),
                normalize=False,
                normalization='integral') * 2 * math.sqrt(math.pi)
        else:
            angular_coeffs = o3.spherical_harmonics(
                self,
                vectors.view(-1, 3),
                normalize=True,
                normalization='integral') * 2 * math.sqrt(math.pi)

        final_shape = tuple(list(vectors.shape[:-1]) + [self.dim])
        basis_values_not_flat = angular_coeffs.view(final_shape)

        return basis_values_not_flat

    def _evaluate_radial_basis(self, vectors, radii=None):
        """Evaluate radial basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """
        if radii is not None:
            bases_flat = self.radial_basis(radii.view(-1))
        else:
            bases_flat = self.radial_basis(torch.norm(vectors, dim=-1).view(-1))

        bases_flat = bases_flat[:, self.radial_selector]
        final_shape = tuple(list(vectors.shape[:-1]) + [self.dim])

        basis_values_not_flat = bases_flat.view(final_shape)

        return basis_values_not_flat

    def _evaluate_joint_basis(self, vectors, radii=None):
        """Evaluate joint (radial x angular) basis functions at {vectors}

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., 3)``
        radii : `torch.Tensor`
            optional, tensor of shape ``(...)`` containing torch.norm({vectors},dim=-1)
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``
        """

        radii = radii if radii is not None else torch.norm(vectors, dim=-1)
        angular_basis = self._evaluate_angular_basis(vectors, radii)
        radial_basis = self._evaluate_radial_basis(vectors, radii)
        return angular_basis * radial_basis

    def _evaluate_basis_on_grid(self, cutoff_radius, res, crop_bases, cutoff_inner_radius=None, basis=None,
                                device=None):
        sample_points_linear = torch.linspace(start=-cutoff_radius, end=cutoff_radius, steps=res, device=device)
        sample_points_cubic = torch.cartesian_prod(
            sample_points_linear, sample_points_linear, sample_points_linear).view(res, res, res, -1)
        radii = torch.norm(sample_points_cubic, dim=-1)

        if basis is not None:
            samples = basis(sample_points_cubic, radii)
        else:
            samples = self._evaluate_joint_basis(sample_points_cubic, radii)

        if crop_bases:
            samples[radii > cutoff_radius, :] = 0
            if cutoff_inner_radius is not None:
                samples[radii < cutoff_inner_radius, :] = 0

        return sample_points_linear, samples

    _basisGridCache = {}

    def _get_basis_on_grid(
            self,
            rcut: float,
            res,
            crop_basis,
            cutoff_radius_inner=None,
            use_cache=True,
            device=None
    ):
        if not use_cache:
            return self._evaluate_basis_on_grid(
                rcut, res, crop_basis, cutoff_radius_inner, device=device)

        key = (rcut, res, crop_basis, cutoff_radius_inner)
        if key in self._basisGridCache:
            return self._basisGridCache[key]
        else:
            ret = self._evaluate_basis_on_grid(
                rcut, res, crop_basis, cutoff_radius_inner, device=device)
            self._basisGridCache[key] = ret
            return ret

    def with_peaks_at(self, vectors: torch.Tensor, values: torch.Tensor = None) -> torch.Tensor:
        """Create a spherical tensor with peaks
        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`

        :param vectors: :math:`\vec r_i` tensor of shape ``(N, 3)``
        :param values: value on the peak, tensor of shape ``(N)``
        :return:  tensor of shape ``(self.dim,)``
        """
        if values is None:
            values = torch.ones(vectors.shape[:-1], dtype=vectors.dtype, device=vectors.device)

        bases = self._evaluate_joint_basis(vectors)
        bases_self_dots_inv = 1.0 / torch.einsum('...a,...a->...', bases, bases)
        coeffs = torch.einsum('...b,...,...->b', bases, bases_self_dots_inv, values)
        return coeffs

    def _evaluate_signal(self, signals: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Expand signal into a weighted sum of bases

        :param signals:  `torch.Tensor` of shape ``({... signals}, self.dim)``
        :param basis: `torch.Tensor` of shape ``({... points}, self.dim)``
        :return: `torch.Tensor` of shape ``({... signals}, {... points})``
        """
        final_shape = tuple(list(signals.shape[:-1]) + list(basis.shape[:-1]))

        signals_flat = signals.view(-1, self.dim)
        basis_flat = basis.view(-1, self.dim)

        ret_flat = torch.einsum('sd,pd->sp', signals_flat, basis_flat)
        ret = ret_flat.view(final_shape)

        return ret

    def signal_xyz(self, signals, vectors):
        basis_values = self._evaluate_joint_basis(vectors)
        return self._evaluate_signal(signals, basis_values)

    def signal_on_grid(
            self,
            signals: torch.tensor,
            cutoff_radius: float,
            res,
            crop_bases=True,
            cutoff_radius_inner=None):
        sample_points_linear, sample_basis = \
            self._get_basis_on_grid(cutoff_radius, res, crop_bases, cutoff_radius_inner, device=signals.device)

        return sample_points_linear, self._evaluate_signal(signals, sample_basis)

    def index(self, _object):
        raise NotImplementedError
