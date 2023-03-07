"""Module containing helper functions for creating invariants (i.e. scalars) from tensor products"""

from typing import Union

from e3nn import o3


def power_spectrum(
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps = None,
        irrep_normalization=None,
) -> Union[o3.TensorSquare, o3.TensorProduct]:
    """Create a tensor product that will calculations the power spectrum for spherical tensors with the passed irreps"""
    if irreps_in2 is None:
        temp_out = o3.TensorSquare(irreps_in1).irreps_out.simplify()
        # Extract all scalars
        irreps_out = [ir for (_, ir) in temp_out if ir.l == 0]
        return o3.TensorSquare(
            irreps_in1,
            filter_ir_out=irreps_out,
            irrep_normalization=irrep_normalization,
        )

    # Doing PS between potentially different spherical tensors
    temp_out = o3.FullTensorProduct(irreps_in1, irreps_in2).irreps_out
    irreps_out = [ir for (_, ir) in temp_out if ir.l == 0]

    return o3.FullTensorProduct(
        irreps_in1,
        irreps_in2,
        irrep_normalization=irrep_normalization,
        filter_ir_out=irreps_out
    )
