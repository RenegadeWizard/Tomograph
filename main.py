import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.transform import warp
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
        file_chooser = self.FileChooser()
        self.file = file_chooser.file
        if self.file is None:
            return None
        self.label.setText(self.file)

    def start_tomograph(self):
        if self.file is None:
            return None
        tomograph = Tomograph(180, 180, 1)
        image = cv2.imread(self.file, 0).astype('float64')
        radon = tomograph.radon_transform(image)
        iradon = tomograph.iradon_transform(radon)

        plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(radon, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 3), plt.imshow(iradon, cmap='gray')
        plt.show()

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
                                                       "Pliki graficzne (*.jpg);;Wszystkie pliki (*)", options=options)
            if file_name:
                self.file = file_name


class Tomograph:
    def __init__(self, emiter_detector_count, angular_extent, step_angle):
        self.emiter_detector_count = emiter_detector_count
        self.angular_extent = angular_extent
        self.step_angle = step_angle

    def radon_transform(self, image, with_steps=False):
        # prepare useful values
        angle = [i for i in np.arange(0.0, 180.0, self.step_angle)]

        # prepare image to be a square and easy to transform
        diag = max(image.shape) * math.sqrt(2)
        pad = [int(math.ceil(diag - i)) for i in image.shape]
        new_mean = [(i + j) // 2 for i, j in zip(image.shape, pad)]
        old_mean = [i // 2 for i in image.shape]
        old_width = [n - o for n, o in zip(new_mean, old_mean)]
        width = [(o, p - o) for o, p in zip(old_width, pad)]
        squared_image = np.pad(image, width, mode='constant', constant_values=0)

        # count center of squared_image and prepare matrix filled with zeros to apply radon transform
        center = squared_image.shape[0] // 2
        r = (squared_image.shape[0] * math.sqrt(2)) // 2
        radon_image = np.zeros((squared_image.shape[0], len(angle)), dtype='float64')

        # iterate through angle list to obtain the result sinogram
        for i, angle in enumerate(np.deg2rad(angle)):
            lines_sum = []
            for j in range(0, self.emiter_detector_count):
                x1 = int(math.ceil(r * math.cos(
                    angle + math.pi - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)) / (
                                self.emiter_detector_count - 1))+center))
                y1 = int(math.ceil(r * math.sin(
                    angle + math.pi - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)) / (
                                self.emiter_detector_count - 1))+center))
                x2 = int(math.ceil(r * math.cos(angle - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)))+center))
                y2 = int(math.ceil(r * math.sin(angle - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)))+center))
                points = list(bresenham(x1, y1, x2, y2))
                actual_sum=0
                actual_sum_vec=[0 for i in range(0,squared_image.shape[0])]
                for p in points:
                    if -1 < p[0] < squared_image.shape[0] and -1 < p[1] < squared_image.shape[1]:
                        actual_sum = actual_sum + squared_image[p[0]][p[1]]
                lines_sum.append(actual_sum)
            radon_image[:,i]=sum(lines_sum)
            # if with_steps==True:
            #   plt.imshow(radon_image, cmap='gray')
            #    plt.xticks([]), plt.yticks([])
            #    plt.show()
            """cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                          [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                         [0, 0, 1]])
            rotated = warp(squared_image, R, clip=False)
            radon_image[:, i] = rotated.sum(0)"""
        return radon_image

    def iradon_transform(self, sinogram, with_steps=False):
        angle = [i for i in np.arange(0.0, 180.0, self.step_angle)]

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
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
            reconstructed += interpolant(t)

        return reconstructed * np.pi / (2 * angles_count)


def main():
    app = QApplication([])
    ex = App()
    app.exec_()


if __name__ == "__main__":
    main()
