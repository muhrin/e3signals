"""Tools for analysing signals"""

__all__ = ('find_peaks_3d',)


def find_peaks_3d(signal_on_grid, cutoff: float):
    peak_grid = (signal_on_grid >= cutoff)

    peak_grid[:-1, :, :] &= (signal_on_grid[:-1, :, :] > signal_on_grid[1:, :, :])
    peak_grid[1:, :, :] &= (signal_on_grid[1:, :, :] > signal_on_grid[:-1, :, :])

    peak_grid[:, :-1, :] &= (signal_on_grid[:, :-1, :] > signal_on_grid[:, 1:, :])
    peak_grid[:, 1:, :] &= (signal_on_grid[:, 1:, :] > signal_on_grid[:, :-1, :])

    peak_grid[:, :, :-1] &= (signal_on_grid[:, :, :-1] > signal_on_grid[:, :, 1:])
    peak_grid[:, :, 1:] &= (signal_on_grid[:, :, 1:] > signal_on_grid[:, :, :-1])

    indices_x, indices_y, indices_z = peak_grid.nonzero(as_tuple=True)

    values = signal_on_grid[indices_x, indices_y, indices_z]

    return indices_x, indices_y, indices_z, values
