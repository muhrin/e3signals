import torch

import e3signals

def test_basics():



    def orthogonalityTest(basisSamples, integrationWeights):
        nBases = basisSamples.shape[-1]
        orthogonalityCheck = torch.einsum('...b,...c,...->bc', basisSamples, basisSamples, integrationWeights)
        orthonormalityError = torch.max(torch.abs(orthogonalityCheck - torch.eye(nBases)))
        orthogonalityCheck2 = orthogonalityCheck / torch.mean(torch.diag(orthogonalityCheck))
        orthogonalityError = torch.max(torch.abs(orthogonalityCheck2 - torch.eye(nBases)))
        return (orthonormalityError, orthogonalityError, orthogonalityCheck)


    cutoff = 3.5
    lmax = 8
    p_val = 1
    p_arg = 1
    nRadialFunctions = (lmax + 1)

    fixedCosineRadialModel = e3signals.FixedCosineRadialModel(cutoff, nRadialFunctions)
    cosineModel = e3signals.CosineFunctions(nRadialFunctions, cutoff)
    cosineModelFaded = e3signals.FadeAtCutoff(cosineModel, cutoff)
    onRadialModel = e3signals.OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, cutoff, 1024)

    rst = e3signals.RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # vectors = torch.rand(4,3) - 0.5
    # vectors *= 3.0 / torch.max(torch.norm(vectors,dim=-1))
    # values = torch.rand(vectors.shape[0])
    # signal = rst.with_peaks_at(vectors,values)

    signals = torch.rand((3, rst.dim), dtype=torch.float32)
    points = torch.rand((7, 3), dtype=torch.float32)
    evals = rst.signal_xyz(signals, points)

    # samplePointsLinear, samplesBasis = rst._evaluateBasisOnGrid(radialCutoff, 100, True, None)
    # signalRealSpace = torch.einsum('...b,b->...',samplesBasis,signal)

    # integrationWeightsLinear = torch.ones_like(samplePointsLinear)
    # integrationWeightsLinear[0] = 0.5
    # integrationWeightsLinear[-1] = 0.5
    # integrationWeightsLinear = integrationWeightsLinear / torch.sum(integrationWeightsLinear) * 2*radialCutoff
    # integrationWeightsCubic = torch.einsum('a,b,c->abc',integrationWeightsLinear,integrationWeightsLinear,integrationWeightsLinear)

    # orthonormalityError, orthogonalityError, orthogonalityCheck = orthogonalityTest(samplesBasis, integrationWeightsCubic)
    # print(orthonormalityError, orthogonalityError)
    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrixJoint.txt', orthogonalityCheck.numpy())


    # radialCutoff = 3.5
    # nRadialFunctions = 3

    # def radialBasisOriginalConstant(r):
    #     ret = torch.ones((len(r),nRadialFunctions))
    #     for i in range(nRadialFunctions-1): ret[:,i+1] = ret[:,i] * r
    #     return ret

    # onRadialModel = orthonormalRadialBasis.OrthonormalRadialFunctions(nRadialFunctions, radialBasisOriginalConstant, radialCutoff, 1024)
    # np.savetxt('/mnt/c/temp/radialFunctions3.txt', onRadialModel.fSamples.numpy())

    # lmax = 1
    # p_val = 1
    # p_arg = -1
    # rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # linearSamples, basisGrid = rst._evaluateBasisOnGrid(radialCutoff, 200, True, None)
    # sampleShape = basisGrid.shape[:-1]

    # integrationWeightsLinear = torch.ones_like(linearSamples)
    # integrationWeightsLinear[0] = 0.5
    # integrationWeightsLinear[-1] = 0.5
    # integrationWeightsLinear = integrationWeightsLinear / torch.sum(integrationWeightsLinear) * 2*radialCutoff
    # integrationWeightsCubic = torch.einsum('a,b,c->abc',integrationWeightsLinear,integrationWeightsLinear,integrationWeightsLinear)

    # orthonormalityError, orthogonalityError, orthogonalityCheck = orthogonalityTest(basisGrid, integrationWeightsCubic)
    # print(orthonormalityError, orthogonalityError)
    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrix3.txt', orthogonalityCheck.numpy())

    # fixedCosineRadialModel = orthonormalRadialBasis.FixedCosineRadialModel(radialCutoff, nRadialFunctions)
    # cosineModel = orthonormalRadialBasis.CosineFunctions(nRadialFunctions, radialCutoff)
    # cosineModelFaded = orthonormalRadialBasis.FadeAtCutoff(cosineModel, radialCutoff)

    # r = torch.linspace(0,radialCutoff,15)
    # y1 = fixedCosineRadialModel(r)
    # y2 = cosineModel(r)
    # y3 = cosineModelFaded(r)

    # onRadialModel = orthonormalRadialBasis.OrthonormalRadialFunctions(nRadialFunctions, cosineModelFaded, radialCutoff, 100)

    # lmax = 3
    # p_val = 1
    # p_arg = -1
    # rst = RadialSphericalTensor(nRadialFunctions, onRadialModel, lmax, p_val, p_arg)

    # #positions = torch.tensor([[1.0, 0.0, 0.0],[3.0, 4.0, 0.0]])
    # positions = torch.rand((2,4,6,3))

    # basisPoints = rst._evaluateBasis(positions)

    # samples, basisGrid = rst._evaluateBasisOnGrid(radialCutoff, 75, True, None)

    # basisOverlapMatrix = torch.einsum('xyza,xyzb->ab',basisGrid,basisGrid) * (samples[1]-samples[0])**3
    # print(basisOverlapMatrix)
    # orthogonalityError = torch.max(torch.abs(basisOverlapMatrix - torch.eye(basisOverlapMatrix.shape[0]))) #Should be small relative to 1
    # print(orthogonalityError)

    # np.savetxt('/mnt/c/Users/tjhardi/Documents/overlapMatrix.txt', basisOverlapMatrix.numpy())
