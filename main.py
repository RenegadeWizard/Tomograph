import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from bresenham import bresenham
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import PyQt5
import pydicom
from pydicom.dataset import Dataset, FileDataset
import threading
from sklearn.metrics import mean_squared_error
from skimage.transform import rescale, downscale_local_mean


def norm(arr: np.ndarray):
    arr = arr.astype(np.int64)
    arr = arr - np.amin(arr)
    arr = (arr / np.amax(arr)) * 255
    return arr.astype(np.int16)


class Communicate(QObject):
    signal = pyqtSignal()
    end = pyqtSignal()


def get_qimage(image: np.ndarray):
    assert (np.max(image) <= 255)
    # image8 = image.astype(np.uint8, order='C', casting='unsafe')
    image = image.astype(np.uint8, order='C', casting='unsafe')
    height, width = image.shape
    bytesPerLine = 3 * width

    image = QImage(image.data, width//3, height//3, bytesPerLine,
                       QImage.Format_RGB888)

    image = image.rgbSwapped()
    return image


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Tomograf"
        self.left = 200
        self.top = 50
        self.width = 1000
        self.height = 750
        self.layout = QVBoxLayout()
        self.file = None
        self.label = None
        self.tomograph = None
        self.progress = None
        self.slider = None
        self.with_steps = False
        self.with_convolve = False
        self.with_dicom = False
        self.init()
        self.setLayout(self.layout)
        self.show()

    def init(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.layout.addWidget(self.add_label("Tomograf", 45))
        self.label = self.add_label("Nie wybrano pliku")
        self.layout.addWidget(self.label)
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.add_button("Wybierz plik", self.choose_file))
        self.steps = QCheckBox("show steps")
        self.steps.stateChanged.connect(lambda: self.box_state(self.steps))
        self.layout.addWidget(self.steps)
        self.conv = QCheckBox("with convolution")
        self.conv.stateChanged.connect(lambda: self.box_state(self.conv))
        self.layout.addWidget(self.conv)
        self.dicom = QCheckBox("with dicom")
        self.dicom.stateChanged.connect(lambda: self.box_state(self.dicom))
        self.layout.addWidget(self.dicom)

        self.onlyInt = QIntValidator()

        self.layout.addWidget(self.add_label("Liczba detektorów/emiterów:", 10))
        self.info1 = QLineEdit()
        self.info1.setValidator(self.onlyInt)
        self.layout.addWidget(self.info1)

        self.layout.addWidget(self.add_label("Rozpiętość kątowa (w stopniach):", 10))
        self.info2 = QLineEdit()
        self.info2.setValidator(self.onlyInt)
        self.layout.addWidget(self.info2)

        self.layout.addWidget(self.add_label("Krok układu (w stopniach):", 10))
        self.info3 = QLineEdit()
        self.info3.setValidator(self.onlyInt)
        self.layout.addWidget(self.info3)

        self.layout.addWidget(self.add_button("Rozpocznij tomograf", self.start_tomograph))

    def init2(self):
        hbox = QHBoxLayout()
        hbox.addWidget(self.add_image(self.tomograph.sinogram))
        hbox.addWidget(self.add_image(self.tomograph.iradon))
        self.layout.addLayout(hbox)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)
        self.slider.setMaximum(100)
        self.slider.setMinimum(1)
        self.slider.setValue(100)

        self.layout.addWidget(self.slider)

    def on_slider(self):
        print(self.slider.value())

    def box_state(self, type):
        if type.text() == "show steps":
            if type.isChecked():
                self.with_steps = True
        if type.text() == "with convolution":
            if type.isChecked():
                self.with_convolve = True
        if type.text() == "with dicom":
            if type.isChecked():
                self.with_dicom = True

    def choose_file(self):
        file_chooser = self.FileChooser()
        if file_chooser.file is None:
            return None
        self.file = file_chooser.file
        self.label.setText(self.file)

    def update_progress(self):
        self.progress.setValue(self.tomograph.progress)

    def show_results(self):
        index = self.layout.count() - 1
        while index >= 0:
            myWidget = self.layout.itemAt(index).widget()
            myWidget.setParent(None)
            index -= 1
        self.init2()

    def start_tomograph(self):
        if self.file is None:
            return None

        sig = Communicate()
        sig.signal.connect(self.update_progress)
        sig.end.connect(self.show_results)
        self.tomograph = Tomograph(sig, int(self.info1.text()), int(self.info2.text()), int(self.info3.text()))
        image = cv2.imread(self.file, 0).astype('int16')

        rad_th = self.RadonThread(self.tomograph, image, self.with_steps, self.with_convolve, self.with_dicom)
        rad_th.setDaemon(True)
        rad_th.start()

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

    @staticmethod
    def add_image(image, label="image"):
        img = QLabel(label)
        img.setAlignment(Qt.AlignCenter)
        img.setPixmap(QPixmap(get_qimage(image)).scaled(256, 256, Qt.KeepAspectRatio))
        return img

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

    class RadonThread(threading.Thread):
        def __init__(self, tomograph, image, with_steps, with_convolve, with_dicom):
            super().__init__()
            self.tomograph = tomograph
            self.image = image
            self.radon = None
            self.iradon = None
            self.with_steps = with_steps
            self.with_convolve = with_convolve
            self.with_dicom = with_dicom

        def run(self):
            self.radon = self.tomograph.radon_transform(self.image, self.with_steps)
            self.iradon = self.tomograph.iradon_transform(self.image, self.radon, self.with_steps, self.with_convolve)

            if self.with_dicom:
                self.iradon = norm(self.iradon)
                self.tomograph.write_dicom(self.iradon, "iradon")

            plt.subplot(2, 2, 1), plt.imshow(self.image, cmap='gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 2), plt.imshow(self.radon, cmap='gray')
            plt.xticks([]), plt.yticks([])
            if self.with_dicom:
                plt.subplot(2, 2, 3), plt.imshow(self.iradon.astype(np.int16), cmap='gray')
                plt.subplot(2, 2, 4), plt.imshow(self.tomograph.read_dicom("out/iradon"), cmap='gray')
            else:
                plt.subplot(2, 2, 3), plt.imshow(self.iradon, cmap='gray')
            plt.show()


class Tomograph:
    def __init__(self, signal, emiter_detector_count=500, angular_extent=180, step_angle=1):
        self.emiter_detector_count = emiter_detector_count
        self.angular_extent = angular_extent
        self.step_angle = step_angle
        self.progress = 0
        self.signal = signal
        self.sinogram = None
        self.iradon = None

    def count_RMSE(self,image, iradonimage):
        image = norm(image)
        iradonimage=norm(np.flipud(iradonimage))
        to_normA = iradonimage.shape[0] /image.shape[0]
        image_to_RMSE=downscale_local_mean(image,(int(1/to_normA)+1,int(1/to_normA)+1))
        to_norm_B=image_to_RMSE.shape[0]/iradonimage.shape[0]
        image_to_RMSE=rescale(image_to_RMSE,1/to_norm_B)

        return math.sqrt(mean_squared_error(image_to_RMSE,iradonimage))

    def radon_transform(self, image, with_steps=False):
        # prepare useful values
        angle_ = [i for i in np.arange(0.0, 180.0, self.step_angle)]

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
        radon_image = []

        # iterate through angle list to obtain the result sinogram
        for i, angle in enumerate(np.deg2rad(angle_)):
            lines_sum = []
            for j in range(self.emiter_detector_count):
                
                x1 = int(r * math.cos(
                    angle + math.pi - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)) / (
                            self.emiter_detector_count - 1)) + center)
                y1 = int(r * math.sin(
                    angle + math.pi - np.deg2rad(self.angular_extent) / 2 + (j * np.deg2rad(self.angular_extent)) / (
                            self.emiter_detector_count - 1)) + center)
                x2 = int(r * math.cos(
                    angle + np.deg2rad(self.angular_extent) / 2 - (j * np.deg2rad(self.angular_extent)) / (
                            self.emiter_detector_count - 1)) + center)
                y2 = int(r * math.sin(
                    angle + np.deg2rad(self.angular_extent) / 2 - (j * np.deg2rad(self.angular_extent)) / (
                            self.emiter_detector_count - 1)) + center)

                points = list(bresenham(x1, y1, x2, y2))
                actual_sum = 0

                for p in points:
                    if -1 < p[0] < squared_image.shape[0] and -1 < p[1] < squared_image.shape[1]:
                        actual_sum = actual_sum + squared_image[p[0]][p[1]]
                lines_sum.append(actual_sum)

            radon_image.append(lines_sum)
            self.progress = 100 * i/len(angle_) + 1
            self.signal.signal.emit()
            # if with_steps==True:
            #   plt.imshow(radon_image, cmap='gray')
            #    plt.xticks([]), plt.yticks([])
            #    plt.show()
        self.sinogram = norm(np.rot90(radon_image))
        return np.rot90(radon_image)

    def iradon_transform(self, image, sinogram, with_steps=False,with_convolve=False):
        # prepare useful values
        angle_ = [i for i in np.arange(0.0, 180.0, self.step_angle)]
        size = sinogram.shape[0]
        base_iradon = np.zeros((size, size))

        # create coordinate system centered at (x,y = 0,0)
        x = np.arange(size) - size / 2
        y = x.copy()
        Y, X = np.meshgrid(-x, y)

        #in each iteration:
        #1. set rotated x in mesh grid form
        #2. move back to original image coords, round values
        #3. take only available cords from base grid
        #4. take one projection from sinogram
        #5. proceed the part of backprojection
        for i, angle in enumerate(np.deg2rad(angle_)):
            Xrot = X * math.sin(angle) - Y * math.cos(angle)
            XrotCor = (np.round(Xrot + size / 2)).astype('int')
            step = np.zeros((size, size))
            k, l = np.where((XrotCor >= 0) & (XrotCor <= (size - 1)))
            sinogram_part = sinogram[:, i]
            if (with_convolve):
                filter1=[5/8 for i in range(0,len(sinogram_part))]
                sinogram_part = np.convolve(sinogram_part, np.array(filter1),mode='same')
            step[k, l] = sinogram_part[XrotCor[k, l]]
            base_iradon += step
            print("Actual RMSE is " + str(self.count_RMSE(image, base_iradon)))

        iradon = np.flipud(base_iradon)
        self.iradon = norm(iradon)
        self.signal.end.emit()
        return iradon

    def write_dicom(self, image, file_name):
        meta = Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = '1.2.840.10008.1.2'

        ds = FileDataset(file_name + '.dcm', {}, file_meta=meta, preamble=b"\0" * 128)
        # ds = FileDataset(file_name + '.dcm', {}, file_meta=meta)

        # patient info
        # ds.PatientsName = name
        # ds.PatientsBirthDate = birth_day
        # ds.PatientsSex = sex
        # ds.PatientsAge = str(age)
        # today = date.today()
        # ds.StudyDate = today.strftime("%d/%m/%Y")

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME1"  # check MONOCHROME1
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = b'\\x00\\x00'
        ds.LargestImagePixelValue = b'\\xff\\xff'
        ds.Columns = image.shape[1]
        ds.Rows = image.shape[0]
        ds.PixelData = image.tostring()
        ds.save_as("out/"+file_name + '.dcm')

    def read_dicom(self, filename):
        ds = pydicom.dcmread(filename+".dcm")
        return ds.pixel_array


def main():
    app = QApplication([])
    ex = App()
    app.exec_()


if __name__ == "__main__":
    main()