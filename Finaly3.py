from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg , NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtWidgets import QFileDialog
from sympy import S, symbols, printing
from matplotlib.figure import Figure
import threading
import os
import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
import matplotlib.pyplot as plt
import time
import multiprocessing


# matplotlib: force computer modern font set
plt.rc('mathtext', fontset='cm')


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)


        #Add your path before the backslash
        uic.loadUi(r'\Finalfinal_1.ui', self)

        # Button connections
        self.openButton.clicked.connect(lambda: self.load())
        self.errorMapRunButton.clicked.connect(lambda: self.startErrorMapThread())
        self.polynomialDegree.textChanged.connect(lambda: self.update_interpolation())
        self.noOfChunks.textChanged.connect(lambda: self.update_interpolation())
        self.extrapolationSlider.valueChanged.connect(lambda: self.updateExtraPolate())
        self.chunkEqCombobox.currentIndexChanged.connect(lambda: self.eqChoice())
        self.noChunksHorizontal.toggled.connect(lambda: self.checkableradiobuttons())
        self.noChunksVertical.toggled.connect(lambda: self.checkableradiobuttons())
        self.polyDegreeHorizontal.toggled.connect(lambda: self.checkableradiobuttons())
        self.polyDegreeVertical.toggled.connect(lambda: self.checkableradiobuttons())
        self.overlapHorizontal.toggled.connect(lambda: self.checkableradiobuttons())
        self.overlapVertical.toggled.connect(lambda: self.checkableradiobuttons())

        self.graph = pg.PlotItem()
        pg.PlotItem.hideAxis(self.graph, 'left')
        pg.PlotItem.hideAxis(self.graph, 'bottom')

        # Intiating canvas
        self.originalCanvas = MplCanvas(
            self.signalPlot,  width=5.5, height=4.5, dpi=90)
        self.originalLayout = QtWidgets.QVBoxLayout()
        self.plotToolbar = NavigationToolbar(self.originalCanvas, self.signalPlot)
        self.plotToolbar.setStyleSheet("background-color:white;")
        self.originalLayout.addWidget(self.originalCanvas)
        self.originalLayout.addWidget(self.plotToolbar)
        self.originalCanvas.draw()
        self.signalPlot.setCentralItem(self.graph)
        self.signalPlot.setLayout(self.originalLayout)

        self.errorCanvas = MplCanvas(
            self.errorMap,  width=5.5, height=4.5, dpi=90)
        self.errorLayout = QtWidgets.QVBoxLayout()

        self.errorLayout.addWidget(self.errorCanvas)
        self.errorCanvas.draw()
        self.errorMap.setCentralItem(self.graph)
        self.errorMap.setLayout(self.errorLayout)


        self.polynomialDegree.setReadOnly(True)
        self.noOfChunks.setReadOnly(True)
        self.extrapolationLabel.setText(str(0))

        self.interPolatedYData = 0
        self.interPolatedXData = 0
        self.extraPolatedYData = 0
        self.extraPolatedXData = 0
        self.stopThread = False
        self.polyfitDataArray = []
        self.pArray = []
        self.eqArray = []
        self.x_chuncks = []
        self.y_chuncks = []
        self.count = 0

    def startErrorMapThread(self):
        if self.errorMapRunButton.text() == "Run":
            self.progressBar.setValue(0)
            self.count = 0
            self.stopThread = False
        elif self.errorMapRunButton.text() =="Stop":
            self.stopThread = True
            self.progressBar.setValue(0)
            self.errorMapRunButton.setText("Run")
        self.errorThread = threading.Thread(target=self.createErrorMap)
        self.errorThread.start()


    def updateProgressBar(self):
        time.sleep(0.05)
        self.count = self.count + 1
        self.progressBar.setValue(self.count)

    def checkableradiobuttons(self):
        self.noChunksHorizontal.setCheckable(True)
        self.noChunksVertical.setCheckable(True)
        self.polyDegreeHorizontal.setCheckable(True)
        self.polyDegreeVertical.setCheckable(True)
        self.overlapHorizontal.setCheckable(True)
        self.overlapVertical.setCheckable(True)
        if self.noChunksVertical.isChecked():
            self.noChunksHorizontal.setCheckable(False)
        elif self.noChunksHorizontal.isChecked():
            self.noChunksVertical.setCheckable(False)
        elif self.polyDegreeVertical.isChecked():
            self.polyDegreeHorizontal.setCheckable(False)
        elif self.polyDegreeHorizontal.isChecked():
            self.polyDegreeVertical.setCheckable(False)
        elif self.overlapVertical.isChecked():
            self.overlapHorizontal.setCheckable(False)
        elif self.overlapHorizontal.isChecked():
            self.overlapVertical.setCheckable(False)


    def ErrorMapParam(self , chunkIndex , degreeIndex , overLapIndex , arrIndex1 , arrIndex2  ):
        dataLength = len(self.yAxisData)
        self.createChunks(int(dataLength/chunkIndex), overLapIndex)
        self.interpolation(degreeIndex)
        normOriginal = np.linalg.norm(self.newydata)
        normInterpolated = np.linalg.norm(
            self.interPolatedYData)
        errorVal = abs(
            normOriginal - normInterpolated) / normOriginal * 100
        self.errorMapArr[arrIndex1-1][arrIndex2-1] = errorVal
        self.updateProgressBar()

    def choseRadioButtons(self):
        if (self.noChunksVertical.isChecked() and self.polyDegreeHorizontal.isChecked()) or (self.noChunksHorizontal.isChecked() and self.polyDegreeVertical.isChecked()):
            self.chunks = np.arange(1, self.numOfChunks + 1, 1)
            self.degrees = np.arange(1, self.degree + 1, 1)
            self.overLap = [0]
            self.progressBar.setMaximum( len(self.chunks) *len(self.degrees) )
            self.errorMapArr = np.zeros((self.degree , self.numOfChunks))
            self.userChoose = "chunks and degree"
        elif (self.polyDegreeVertical.isChecked() and self.overlapHorizontal.isChecked()) or (self.polyDegreeHorizontal.isChecked() and self.overlapVertical.isChecked()):
            self.chunks = [int(self.noOfChunks.text())]
            self.degrees  = np.arange(1, self.degree + 1, 1)
            self.overLap = np.arange(0, 0.3, 0.05)
            self.progressBar.setMaximum(len(self.degrees) * len(self.overLap))
            self.errorMapArr = np.zeros((self.degree, len(self.overLap)))
            self.userChoose = "degree and overlap"
        elif (self.noChunksVertical.isChecked() and self.overlapHorizontal.isChecked()) or (self.noChunksHorizontal.isChecked() and self.overlapVertical.isChecked()):
            self.chunks = np.arange(1, self.numOfChunks + 1, 1)
            self.degrees =[int(self.polynomialDegree.text())]
            self.overLap = np.arange(0, 0.3, 0.05)
            self.progressBar.setMaximum(len(self.overLap) * len(self.chunks))
            self.errorMapArr = np.zeros((self.numOfChunks, int(len(self.overLap))))
            self.userChoose = "chunk and overlap"

    def createErrorMap(self):
        self.errorMapRunButton.setText("Stop")
        self.choseRadioButtons()
        self.errorCanvas.fig.clear()
        dataLength = len(self.yAxisData)
        for i in self.degrees:
            if self.stopThread:
                break
            for j in self.chunks:
                if self.stopThread:
                    break
                for k in range(0, len(self.overLap), 1):
                    if self.stopThread:
                        break
                    if self.userChoose == "chunks and degree":
                        self.ErrorMapParam( j , i ,0 , i , j )
                    elif self.userChoose == "degree and overlap":
                        self.ErrorMapParam(int(dataLength/self.numChunksData), i , int(self.overLap[k]*self.numChunksData) , i , k+1 )
                    elif self.userChoose == "chunk and overlap":
                        self.ErrorMapParam( j , self.degree , int(self.overLap[k]*self.numChunksData) , j , k+1)
        if self.userChoose == "chunk and degree":
            if self.noChunksVertical.isChecked():
                self.errorMapArr = self.errorMapArr.transpose()

        elif self.userChoose== "degree and overlap":
            if self.polyDegreeVertical.isChecked():
                self.errorMapArr = self.errorMapArr.transpose()
        elif self.userChoose== "chunk and overlap":
            if self.noChunksVertical.isChecked():
                self.errorMapArr = self.errorMapArr.transpose()
        if self.stopThread:
            self.progressBar.setValue(0)
            self.errorMapRunButton.setText("Run")
            return
        self.axes = self.errorCanvas.fig.add_subplot(111)
        contourPLot = self.axes.contourf(self.errorMapArr)
        self.errorCanvas.fig.colorbar(contourPLot)
        self.errorCanvas.draw()
        self.errorMap.setCentralItem(self.graph)
        self.errorMap.setLayout(self.errorLayout)
        self.errorMapRunButton.setText("Run")


    def load(self):
            self.polynomialDegree.setReadOnly(False)
            self.noOfChunks.setReadOnly(False)
            self.fileName = QFileDialog.getOpenFileName(
                None, "Select a file...", os.getenv('HOME'), filter="All files (*)")
            path = self.fileName[0]
            data = pd.read_csv(path)
            self.yAxisData = data.values[:, 1]
            self.xAxisData = data.values[:, 0]
            self.noOfChunks.setText(str("1"))
            self.polynomialDegree.setText(str("1"))
            self.update_interpolation()

    def plot(self):
            self.originalCanvas.axes.clear()
            self.originalCanvas.axes.scatter(
                self.xAxisData, self.yAxisData, s=1.5, color='k')
            self.originalCanvas.axes.plot(
                self.interPolatedXData, self.interPolatedYData, color='r')
            self.originalCanvas.axes.plot(
                self.extraPolatedXData, self.extraPolatedYData, color='g')
            self.originalCanvas.draw()
            self.signalPlot.setCentralItem(self.graph)
            self.signalPlot.setLayout(self.originalLayout)

    def updateExtraPolate(self):
            self.noOfChunks.setText(str("1"))
            self.noOfChunks.setReadOnly(True)
            if self.extrapolationSlider.value() == 100 :
                self.noOfChunks.setReadOnly(False)
            self.extrapolationLabel.setText(str(100 - self.extrapolationSlider.value()))
            self.extrapolationValue = (100 - self.extrapolationSlider.value()) / 100
            self.extrapolation()
            self.interpolationLatex()
            self.plot()

    def extrapolation(self):
            if self.numOfChunks > 1:
                pass
            dataLength = len(self.yAxisData)
            extraLength = int(self.extrapolationValue*dataLength)
            self.interPolatedYData = []
            self.interPolatedXData = []
            self.extraPolatedYData = []
            self.extraPolatedXData = []
            self.originalCanvas.axes.clear()

            interXData = self.xAxisData[0: (dataLength - extraLength)]
            interYData = self.yAxisData[0: (dataLength - extraLength)]
            extraXData = self.xAxisData[(
                dataLength - extraLength) - 1: (dataLength - 1)]

            polyfitData = np.polyfit(interXData, interYData, self.degree)
            polynomialFunc = np.poly1d(polyfitData)
            self.interPolatedYData = [*self.interPolatedYData, *polynomialFunc(interXData)]
            self.interPolatedXData = [*self.interPolatedXData, *interXData]
            self.extraPolatedYData = [*self.extraPolatedYData, *polynomialFunc(extraXData)]
            self.extraPolatedXData = [*self.extraPolatedXData, *extraXData]

    def update_interpolation(self):
            self.updateInterpolate()
            self.eqChunks()
            self.createChunks(self.numChunksData, 0)
            self.interpolation(self.degree)
            self.plot()
            self.interpolationLatex()

    def updateInterpolate(self):
            self.numOfChunks = int(1)
            self.degree = int(1)
            dataLength = len(self.yAxisData)

            if self.noOfChunks.text() != "":
                self.numOfChunks = int(self.noOfChunks.text())

            if self.polynomialDegree.text() != "":
                self.degree = int(self.polynomialDegree.text())

            self.numChunksData = int(dataLength/self.numOfChunks)
            self.numOverLap = int(0.25*self.numChunksData)

    def eqChunks(self):
            self.chunkEqCombobox.clear()
            for i in range(0, self.numOfChunks, 1):
                self.chunkEqCombobox.addItem("chunk #" + str(i+1))

    def createChunks(self, chunksDataLen, overLapLen):
            self.polyfitDataArray = []
            dataLength = len(self.yAxisData)
            self.interPolatedYData = []
            self.x_chuncks = []
            self.interPolatedXData = []
            self.originalCanvas.axes.clear()
            self.x_chuncks = [self.xAxisData[i:i+chunksDataLen]
                              for i in range(0, dataLength, chunksDataLen - overLapLen)]
            self.y_chuncks = [self.yAxisData[i:i+chunksDataLen]
                              for i in range(0, dataLength, chunksDataLen - overLapLen)]


    def interpolation(self, degree):
            self.newydata = []
            for i in range(0, len(self.y_chuncks), 1):
                polyfitData = np.polyfit(self.x_chuncks[i], self.y_chuncks[i], degree)
                polynomialFunc = np.poly1d(polyfitData)
                self.newydata = [*self.newydata, *self.y_chuncks[i]]
                self.interPolatedYData = [
                    *self.interPolatedYData, *polynomialFunc(self.x_chuncks[i])]
                self.interPolatedXData = [
                    *self.interPolatedXData, *self.x_chuncks[i]]
                self.polyfitDataArray.append(polyfitData)

    def interpolationLatex(self):
            self.eqArray = []
            x = symbols("x")
            chunksnumber = len(self.y_chuncks)
            for index in range(0,chunksnumber , 1):
                poly = sum(S("{:6.2f}".format(v))*x**i for i,
                           v in enumerate(self.polyfitDataArray[index][::-1]))
                eqLatex = printing.latex(poly)
                eqLatexLabel = self.latexWrap(eqLatex)
                self.eqArray.append(eqLatexLabel)

            # percentage relative error
            normOriginal = np.linalg.norm(self.yAxisData)
            normInterpolated = np.linalg.norm(self.interPolatedYData)

            error = abs((normOriginal - normInterpolated)) / normOriginal * 100
            eqError = "percentage error = " + str(round(error)) + " %"
            self.eqLabel.setText(eqError)

            title = "$f(x) = $" + self.eqArray[0]
            self.originalCanvas.axes.set_title(title, fontsize="medium")
            self.chunkEqCombobox.setCurrentIndex(0)

        # function to wrap latex eaution if too big
    def latexWrap(self, eqUnwrapped):
            eqParse = eqUnwrapped.replace('$', '')
            eqParse = eqParse.split(' ')
            eqLength = len(eqParse)

            eqWrapped = ""
            while eqLength > 16:
                temp = eqParse[0:15]
                temp = ' '.join(temp)
                temp = "${}$".format(temp)
                eqWrapped += temp + "\n"

                eqParse = eqParse[15:]
                eqLength = len(eqParse)

            if eqLength <= 16:
                temp = ' '.join(eqParse)
                temp = "${}$".format(temp)
                eqWrapped += temp

            return eqWrapped

    def eqChoice(self):
            if self.chunkEqCombobox.count() == 0:
                raise ValueError('the combobox is empty cant be accessed')
            else:
                i = self.chunkEqCombobox.currentIndex()
                eqCurrent = self.eqArray[i]
                title = "$f(x) = $" + eqCurrent
                self.createChunks(self.numChunksData, 0)
                self.interpolation(self.degree)
                self.plot()
                self.originalCanvas.axes.set_title(title, fontsize="medium")
                self.originalCanvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
