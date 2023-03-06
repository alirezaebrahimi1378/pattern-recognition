import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load hyperspectral image from .mat file
mat_contents = sio.loadmat('/home/alireza/Desktop/seg/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat')
data = mat_contents['X']

# Select a pixel to perform continuum removal on
x, y = 100, 100
spectrum = data[x, y, :]

# Perform continuum removal using the Savitzky-Golay filter
wavelengths = np.arange(spectrum.shape[0])
deg = 4
cr_spectrum = savgol_filter(spectrum, window_length=15, polyorder=deg)

# Plot original spectrum and continuum removed spectrum
plt.figure()
plt.plot(wavelengths, spectrum, label='Original Spectrum')
plt.plot(wavelengths, cr_spectrum, label=f'Continuum Removed Spectrum with degree of {deg}')
plt.xlabel('BandNumber')
plt.ylabel('Reflectance')
plt.legend()
plt.show()


