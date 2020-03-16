import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from scipy.fftpack import fft, ifft
from functools import partial
from bresenham import bresenham

def radon_transform(image, angle,with_steps=False):
    diag = max(image.shape)*np.sqrt(2)
    pad = [int(np.ceil(diag - i)) for i in image.shape]
    new_center = [(i + j) // 2 for i, j in zip(image.shape, pad)]
    old_center = [i // 2 for i in image.shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(image, pad_width, mode='constant',constant_values=0)

    #if padded_image.shape[0] != padded_image.shape[1]:
    #   raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    angle=[i for i in np.arange(0.0,180.0,angle)]
    radon_image = np.zeros((padded_image.shape[0], len(angle)),dtype='float64')

    for i, angle in enumerate(np.deg2rad(angle)):
        cos_a = np.cos(angle), np.sin(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    return radon_image

def iradon_transform(sinogram, angle,with_steps=False):
    angle = [i for i in np.arange(0.0, 180.0, angle)]

    angles_count = len(angle)
    if angles_count != sinogram.shape[1]:
        raise ValueError("The angle does'ot match the number of projections in sinogram.")

    img_shape = sinogram.shape[0]
    output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(sinogram, pad_width, mode='constant', constant_values=0)

    n = np.concatenate((np.arange(1, projection_size_padded / 2 + 1, 2, dtype=np.int),
                        np.arange(projection_size_padded / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(projection_size_padded)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fft(f))[:, np.newaxis]
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2

    for col, angle in zip(radon_filtered.T, np.deg2rad(angle)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        #interpolant = bresenham(np.interp, x, col, 0)
        interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        reconstructed += interpolant(t)

    return reconstructed * np.pi / (2 * angles_count)

def main():
    image = cv2.imread('CT_ScoutView.jpg', 0).astype('float64')
    with_steps=True
    radon = radon_transform(image, 0.125,with_steps)
    iradon=iradon_transform(radon,0.125)

    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(radon, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(iradon, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
