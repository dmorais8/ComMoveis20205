__author__ = "David Morais"
__copyright__ = "Copyright 2020, Projeto 01, Comunicacoes Moveis"
__credits__ = ["David Morais"]
__maintainer__ = "David Morais"
__email__ = "moraisdavid8@gmail.com"
__status__ = "Development"

import numpy as np
import sys as system
from matplotlib import pyplot as plt
from math import e


def cost_walfish_ikegami(dfc, distbs):
    pldb_micro = 55.5 + 38 * np.log10(distbs / 1e3) + (24.5 + 1.5 * dfc / 925) * np.log10(dfc)

    return pldb_micro


def okumura_hata_evaluation(dFc, dHBs, mtDistEachBs, dAhm):
    mt_pld_b = 69.55 + 26.16 * np.log10(dFc) + (44.9 - 6.55 * np.log10(
        dHBs)) * np.log10(mtDistEachBs / 1E3) - 13.82 * np.log10(dHBs) - dAhm

    return mt_pld_b


def cost231_evaluation(dFc, dHBs, mtDistEachBs, dAhm):
    _A = 46.3 + (33.9 * np.log10(dFc)) - (13.28 * np.log10(dHBs)) - dAhm
    _B = 44.9 - 6.55 * np.log10(dHBs)
    _C = 3

    mt_pld_b = _A + (_B * np.log10(mtDistEachBs / 1E3)) + _C

    return mt_pld_b


class Outage(object):

    def __init__(self, vtFc, dRrange, dOffset, dPtdBm, dSensitivity, dHBs, dHMob, model):

        self.dRrange = dRrange
        self.dOffset = dOffset
        self.vtFc = vtFc
        self.dHBs = dHBs
        self.dHMob = dHMob
        self.dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2. - 4.97
        self.dPtdBm = dPtdBm
        self.dSensitivity = dSensitivity
        self.model = model

    def outageCalc(self):

        for dFc in self.vtFc:

            for dR in self.dRrange:

                dPasso = np.ceil(dR / 50)
                dRMin = dPasso
                dDimX = 5*dR
                dDimY = 6*np.sqrt(3/4)*dR

                vtBs = [0]

                for iBs in range(7):

                    vtBs.append(
                        dR*np.sqrt(3)*np.exp(1j * ((iBs-2) *
                                                   (np.pi/3) + self.dOffset))
                    )

                vtBs.pop(7)

                vtBs = np.array(vtBs, dtype=complex)
                vtBs = vtBs + (dDimX/2 + 1j*(dDimY/2))

                dDimX += (dDimX % dPasso)
                dDimY += (dDimY % dPasso)

                x = np.arange(0, dDimX, dPasso)
                y = np.arange(0, dDimY, dPasso)

                mtPosx, mtPosy = np.meshgrid(x, y)

                mtPosEachBS = []
                mtDistEachBs = []
                mtPowerEachBSdBm = []
                mtPldB = []
                mtPowerFinaldBm = np.where(
                    mtPosy != -(np.Infinity), -np.Infinity, mtPosy)

                for iBsD in range(7):

                    singleBsPos = (mtPosx + 1j*mtPosy) - vtBs[iBsD]
                    mtPosEachBS.append(singleBsPos)

                    singleBsdist = np.absolute(mtPosEachBS[iBsD])
                    mtDistEachBs.append(singleBsdist)

                    mtDistEachBs[iBsD][mtDistEachBs[iBsD] < dRMin] = dRMin

                    if self.model == 1:
                        singlePldB = okumura_hata_evaluation(
                            dFc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)
                    else:
                        singlePldB = cost231_evaluation(
                            dFc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)

                    mtPldB.append(singlePldB)

                    singlePowerBSdBm = self.dPtdBm - np.asarray(mtPldB[iBsD])
                    mtPowerEachBSdBm.append(singlePowerBSdBm)

                    mtPowerFinaldBm = np.array(mtPowerFinaldBm, dtype=float)
                    mtPowerFinaldBm = np.maximum(
                        mtPowerFinaldBm, mtPowerEachBSdBm[iBsD])

                # Outage (limite 10%)
                dOutRate = 100 * \
                    np.size(np.where(mtPowerFinaldBm < self.dSensitivity),
                            axis=1) / np.size(mtPowerFinaldBm)

                list(vtBs).clear()

                if dOutRate <= 10.:
                    print(f'-----------------------\nCarrier Frequency: {dFc}\n'
                          f'Outage rate: {dOutRate: .4f}%\nCoverage radius: {dR}')

                    break


class MeasurePoints(object):
    """Class for storing/setting parameters for the main execution"""

    def __init__(self, dR, dOffset, dFc, dHBs, dHMob, dPtdBm):
        self.dR = dR
        self.dPasso = np.ceil(dR/50)
        self.dRMin = self.dPasso
        self.dDimX = 5 * self.dR
        self.dDimY = 6*np.sqrt(3/4)*self.dR
        self.dOffset = dOffset
        self.dFc = dFc
        self.dHBs = dHBs
        self.dHMob = dHMob
        self.dPtdBm = dPtdBm
        self.dAhm = 3.2 * (np.log10(11.75 * self.dHMob)) ** 2. - 4.97

    @staticmethod
    def fdrawsector(dR, dCenter):

        vtHex = np.zeros([0], dtype=complex)

        for ie in range(6):
            vtHex = np.append(
                vtHex, [(dR * (np.cos((ie - 1) * np.pi / 3) + 1j * np.sin((ie - 1) * np.pi / 3)))])
            vtHex[ie] = vtHex[ie] + dCenter

        vtHexp = np.append(vtHex, vtHex[0])

        plt.axis('equal')
        plt.plot(vtHexp.real, vtHexp.imag, 'k')

    @staticmethod
    def fdrawDeploy(dR, vtBs):

        for iBsD in range(len(vtBs)):
            MeasurePoints.fdrawsector(dR, vtBs[iBsD])
            plt.axis('equal')
            plt.plot(vtBs[iBsD].real, vtBs[iBsD].imag, 'sk', markersize=5.8, fillstyle='none')

    def fdrawMeasurementPoints(self):

        vtBs = [0]

        for iBs in range(7):
            vtBs.append(self.dR * np.sqrt(3) * pow(e, (1j * ((iBs - 2) * np.pi / 3 + self.dOffset))))

        vtBs.pop(7)
        vtBs = np.array(vtBs, dtype=complex)
        vtBs = vtBs + (self.dDimX / 2 + 1j * (self.dDimY / 2))
        MeasurePoints.fdrawDeploy(self.dR, vtBs)
        self.dDimX += (self.dDimX % self.dPasso)
        self.dDimY += (self.dDimY % self.dPasso)

        x = np.arange(0, self.dDimX, self.dPasso)
        y = np.arange(0, self.dDimY, self.dPasso)

        mtPosx, mtPosy = np.meshgrid(x, y)

        mtPosEachBS = []
        mtDistEachBs = []
        mtPowerEachBSdBm = []
        mtPldB = []
        mtPowerFinaldBm = np.where(mtPosy != -(np.Infinity), -np.Infinity, mtPosy)

        for iBsD in range(7):
            singleBsPos = (mtPosx + 1j * mtPosy) - vtBs[iBsD]
            mtPosEachBS.append(singleBsPos)

            singleBsdist = np.absolute(mtPosEachBS[iBsD])
            mtDistEachBs.append(singleBsdist)

            mtDistEachBs[iBsD][mtDistEachBs[iBsD] < self.dRMin] = self.dRMin

            singlePldB = okumura_hata_evaluation(self.dFc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)
            mtPldB.append(singlePldB)

            singlePowerBSdBm = self.dPtdBm - np.asarray(mtPldB[iBsD])
            mtPowerEachBSdBm.append(singlePowerBSdBm)

            # plt.figure(iBsD)
            # plt.pcolor(mtPosx, mtPosy, mtPowerEachBSdBm[iBsD], cmap='hsv')
            # plt.colorbar()
            # MeasurePoints.fdrawDeploy(self.dR, vtBs)
            # plt.axis('equal')
            # plt.title(f"ERB {iBsD + 1}")

            mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm[iBsD])

        plt.figure()
        plt.pcolor(mtPosx, mtPosy, mtPowerFinaldBm, cmap='hsv')
        plt.colorbar()
        MeasurePoints.fdrawDeploy(self.dR, vtBs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs')
        plt.show()


if __name__ == "__main__":

    print("----------Menu----------")
    print("Select an action: ")
    print("1 - calculate outage rate by carrier frequency(Entrega 01 e 02)")
    print("2 - draw measurement points")
    option = int(input("Option: "))

    if option not in [1, 2]:

        print("Choose a correct option")
        print("Terminating")

        system.exit(1)

    elif option == 1:

        dOffset = np.pi / 6
        dPtdBm = 57
        dSensitivity = -104
        dHMob = 1.8
        dHBs = 30
        dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2. - 4.97

        vtFc = [800, 900, 1800, 1900, 2100]

        dRrange = np.arange(1000, 12000, 10).tolist()

        dRrange.reverse()

        dRrange = np.array(dRrange, dtype=int)

        print("Choose a model: 1 - Okumura Hata, 2 - Cost Hata (Cost 231), Any key - Fechar o programa")
        model = (int(input("Option: ")))

        outage = Outage(vtFc, dRrange, dOffset, dPtdBm, dSensitivity, dHBs, dHMob, model)

        if model in [1, 2]:

            outage.outageCalc()

        system.exit(0)

    else:
        dR = 5E3
        dOffset = np.pi / 6
        dFc = 800
        dHBs = 30
        dHMob = 1.8
        dAhm = 3.2 * (np.log10(11.75 * dHMob)) ** 2. - 4.97
        dPtdBm = 57

        draw = MeasurePoints(dR, dOffset, dFc, dHBs, dHMob, dPtdBm)
        draw.fdrawMeasurementPoints()

        system.exit(0)


