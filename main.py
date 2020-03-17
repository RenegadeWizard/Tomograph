import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp, ProjectiveTransform
from scipy.fftpack import fft, ifft
from functools import partial
from bresenham import bresenham
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Tomograf"
        self.left = 200
        self.top = 50
        self.width = 1000
        self.height = 750
        self.file = None
        self.label = None
        self.init()

    def init(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        layout = QVBoxLayout()
        layout.addWidget(self.add_label("Tomograf", 45))
        self.label = self.add_label("Nie wybrano pliku")
        layout.addWidget(self.label)
        layout.addWidget(self.add_button("Wybierz plik", self.choose_file))
        layout.addWidget(self.add_button("Rozpocznij tomograf", self.start_tomograph))
        self.setLayout(layout)
        self.show()

    def choose_file(self):
        file_chooser = FileChooser()
        self.file = file_chooser.file
        self.label.setText(self.file)

    def start_tomograph(self):  # TODO
        print("Starting tomograph")

    @staticmethod
    def add_label(title, size=16):
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont(None, size, QFont.AnyStyle))
        return label

    @staticmethod
    def add_button(title, method):
        button = QPushButton(title)
        button.clicked.connect(method)
        return button


class FileChooser(QWidget):
    def __init__(self):
        super().__init__()
        self.file = None
        self.init()

    def init(self):
        self.setWindowTitle("Wybierz plik")
        self.setGeometry(400, 100, 640, 480)

        self.open_file_name_dialog()

        self.show()

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                   "All Files (*);;Python Files (*.py)", options=options)
        if file_name:
            self.file = file_name


def start_gui():
    app = QApplication([])
    ex = App()
    app.exec_()


def radon_transform(image, angle, with_steps=False):
    diag = max(image.shape) * np.sqrt(2)
    pad = [int(np.ceil(diag - i)) for i in image.shape]
    new_center = [(i + j) // 2 for i, j in zip(image.shape, pad)]
    old_center = [i // 2 for i in image.shape]
    pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
    pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)

    # if padded_image.shape[0] != padded_image.shape[1]:
    #     raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    angle = [i for i in np.arange(0.0, 180.0, angle)]
    radon_image = np.zeros((padded_image.shape[0], len(angle)), dtype='float64')

    for i, angle in enumerate(np.deg2rad(angle)):
        cos_a = np.cos(angle)  # , np.sin(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    return radon_image


def iradon_transform(sinogram, angle, with_steps=False):
    angle = [i for i in np.arange(0.0, 180.0, angle)]

    angles_count = len(angle)
    if angles_count != sinogram.shape[1]:
        raise ValueError("The angle doesn't match the number of projections in sinogram.")

    img_shape = sinogram.shape[0]
    output_size = int(np.floor(np.sqrt(img_shape ** 2 / 2.0)))
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
        # interpolant = bresenham(np.interp, x, col, 0)
        interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        reconstructed += interpolant(t)

    return reconstructed * np.pi / (2 * angles_count)


def main():
    start_gui()
    image = cv2.imread('CT_ScoutView.jpg', 0).astype('float64')
    with_steps = True
    radon = radon_transform(image, 0.125, with_steps)
    iradon = iradon_transform(radon, 0.125)

    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(radon, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(iradon, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
