import torch

import e3signals


def test_find_peaks_3d():
    def orthogonalityTest(basisSamples, integrationWeights):
        nBases = basisSamples.shape[-1]
        orthogonalityCheck = torch.einsum('...b,...c,...->bc', basisSamples, basisSamples, integrationWeights)
        orthonormalityError = torch.max(torch.abs(orthogonalityCheck - torch.eye(nBases)))
        orthogonalityCheck2 = orthogonalityCheck / torch.mean(torch.diag(orthogonalityCheck))
        orthogonalityError = torch.max(torch.abs(orthogonalityCheck2 - torch.eye(nBases)))
        return (orthonormalityError, orthogonalityError, orthogonalityCheck)

    rcut = 3.5
    lmax = 11
    p_val = 1
    p_arg = 1
    nRadialFunctions = (lmax + 1)

    # fixedCosineRadialModel = e3signals.FixedCosineRadialModel(rcut, nRadialFunctions)
    cosineModel = e3signals.CosineFunctions(nRadialFunctions, rcut)
    cosineModelFaded = e3signals.FadeAtCutoff(cosineModel, rcut)
    onRadialModel = e3signals.OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, rcut, 1024)

    rst = e3signals.RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    peak_points = torch.rand((3, 3)) - 0.5
    peak_points /= torch.linalg.norm(peak_points)
    peak_points *= 2

    signal = rst.with_peaks_at(peak_points)
    linearSamples, signalOnGrid = rst.signal_on_grid(
        signal,
        rcut,
        100,
        crop_bases=True,
        # useCache=True
    )

    indicesX, indicesY, indicesZ, values = e3signals.find_peaks_3d(signalOnGrid, 0.5)
    peaksX = linearSamples[indicesX]
    peaksY = linearSamples[indicesY]
    peaksZ = linearSamples[indicesZ]
