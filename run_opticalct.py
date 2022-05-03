from opticalct.opticalct import Dataset

data = Dataset('data/lemon', invert=True, scale=1.0)
data.display(data.projections, interactive=True)
data.compute_sinogram()
data.display(data.sinogram, interactive=True)
data.compute_volume()
data.display(data.volume, interactive=True)
data.save_volume('data/lemon_reconstructed')
