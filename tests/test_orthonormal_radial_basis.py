import torch

import e3signals


def test_basics():
    cutoff = 3.5
    n_radials = 14

    fixed_cos_radial_model = e3signals.FixedCosineRadialModel(cutoff, n_radials)
    cos_model = e3signals.CosineFunctions(n_radials, cutoff)
    smooth_cos_model = e3signals.FadeAtCutoff(cos_model, cutoff)

    r = torch.linspace(0, cutoff, 15)
    y1 = fixed_cos_radial_model(r)
    y2 = cos_model(r)
    y3 = smooth_cos_model(r)

    ortho_radial_model = e3signals.OrthonormalRadialFunctions(n_radials, smooth_cos_model, cutoff, 100)
    r = torch.linspace(0, cutoff, 51, requires_grad=True)
    y = ortho_radial_model(r)
    g = torch.zeros_like(y)
    g[:, -1] += 1
    g2 = y.backward(gradient=g)


def test_smoothing_fn():
    rcut = 3.
    # Let's apply the smoothing function to y = 1
    num_radials = 5
    smoothed = e3signals.FadeAtCutoff(lambda x: torch.ones(x.shape + (num_radials,)), rcut)

    r = torch.linspace(0, 3.6, 7)
    res = smoothed(r)
    assert torch.all(res[r >= rcut] == 0)

    # Now test that it works with a real radial function
    smoothed = e3signals.FadeAtCutoff(e3signals.CosineFunctions(num_radials, rcut), rcut)
    res = smoothed(r)
    assert torch.all(res[r >= rcut] == 0)
