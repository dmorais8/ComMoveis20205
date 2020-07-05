__author__ = "David Morais"
__copyright__ = "Copyright 2020, Projeto 01, Comunicacoes Moveis"
__credits__ = ["David Morais"]
__maintainer__ = "David Morais"
__email__ = "moraisdavid8@gmail.com"
__status__ = "Development"
__version__ = "latest"

import sys as system
import numpy as np
from matplotlib import pyplot as plt, cm
import math

def cost_walfish_ikegami(dfc, distbs):
    pldb_micro = 55.5 + 38 * np.log10(distbs / 1e3) + (24.5 + 1.5 * dfc / 925) * np.log10(dfc)

    return pldb_micro


def okumura_hata(dFc, dHBs, mtDistEachBs, dAhm):
    pldb_okumura = 69.55 + 26.16 * np.log10(dFc) + (44.9 - 6.55 * np.log10(
        dHBs)) * np.log10(mtDistEachBs / 1E3) - 13.82 * np.log10(dHBs) - dAhm

    return pldb_okumura


def cost_231(dFc, dHBs, mtDistEachBs, dAhm):
    _A = 46.3 + (33.9 * np.log10(dFc)) - (13.28 * np.log10(dHBs)) - dAhm
    _B = 44.9 - 6.55 * np.log10(dHBs)
    _C = 3

    pldb_cost = _A + (_B * np.log10(mtDistEachBs / 1E3)) + _C

    return pldb_cost

def surfplot(X, Y, Z):
    """

    :param X:
    :param Y:
    :param Z:
    :return:
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    ax.set_zlim(-40, 40)
    plt.ylim(0, 800)
    plt.xlim(0, 800)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


# Classe com as funcoes metodos utilizados para execuacao do projeto.
class Functions:

    def __init__(self, dR, dFc, dShad, dSigmaShad, dPtdBm, dHMob, dHBs, dOffset, drMicro, dPtdBmMicro):

        """
        Construtor da classe: Parametros recebido na instanciacao.

        :param dR (int): Raio de um unica celula no grid
        :param dFc(int): Frequencia da portadora
        :param dShad(int): Fator de sombreamento
        :param dSigmaShad (int): Desvio padrao do sobreamento
        :param dPtdBm:
        :param dHmob (float): Altura da estacao movel
        :param dHBs (int): Altura da estacao radio base em cada celular do grid
        :param dOffset (float): offset
        """

        self.dPtdBmMicro = dPtdBmMicro
        self.drMicro = drMicro
        self.dShad = dShad
        self.dSigmaShad = dSigmaShad
        self.dR = dR
        self.dPasso = 10
        self.dRMin = self.dPasso
        self.dDimX = 5 * self.dR
        self.dDimY = 6 * np.sqrt(3 / 4) * self.dR
        self.dOffset = dOffset
        self.dFc = dFc
        self.dHBs = dHBs
        self.dHMob = dHMob
        self.dPtdBm = dPtdBm
        self.dAhm = 3.2 * (np.log10(11.75 * self.dHMob)) ** 2. - 4.97

    @staticmethod
    def draw_sector(dR, dCenter):

        """
               Metodo estatico para realizar o desenho de um unico setor(celula) dentro do grid

               Parametros:

               dR (int): Raio da celula
               dCenter (complex): posicao do centro da celula
        """

        vtHex = np.zeros([1, 0], dtype=complex)

        for ie in range(6):
            vtHex = np.append(
                vtHex, [(dR * (np.cos((ie - 1) * np.pi / 3) + 1j * np.sin((ie - 1) * np.pi / 3)))])

        vtHex = vtHex + dCenter

        vtHexp = np.append(vtHex, vtHex[0])

        plt.axis('equal')
        plt.plot(vtHexp.real, vtHexp.imag, 'k')

    @staticmethod
    def draw_deploy(dR, vtBs):

        """
            Metodo estatico para realizar o desenho de um unico setor(celula) dentro do grid

            Parametros:

            dR (int): Raio da celula
            vtBs (array complex): Array com os centros de cada celula no grid
        """

        for iBsD in range(len(vtBs)):
            Functions.draw_sector(dR, vtBs[iBsD])
            plt.axis('equal')
            plt.plot(vtBs[iBsD].real, vtBs[iBsD].imag, 'sk', markersize=5.8, fillstyle='none')

    def draw_shad_graphs(self):

        """
            Metodo estatico para realizar o plot do grafico com sobreamento nao correlacionado

        """

        vtBs = np.array([0], dtype=complex)

        for iBs in range(1, 7):
            vtBs = np.append(vtBs, self.dR * np.sqrt(3) * pow(np.e, (1j * ((iBs - 2) * np.pi / 3 + self.dOffset))))
        vtBs = vtBs + (self.dDimX / 2 + 1j * self.dDimY / 2)

        self.dDimX += np.ceil(self.dDimX % self.dPasso)
        self.dDimY += np.ceil(self.dDimY % self.dPasso)

        x = np.arange(0, self.dDimX, self.dPasso)
        y = np.arange(0, self.dDimY, self.dPasso)

        mtPosx, mtPosy = np.meshgrid(x, y)

        mtPosEachBS = []
        mtDistEachBs = []

        mtPowerEachBSShaddBm = []
        mtPowerEachBSdBm = []

        mtPldB = []

        mtPowerFinaldBm = np.where(mtPosy != -(np.Infinity), -np.Infinity, mtPosy)
        mtPowerFinalShaddBm = np.where(mtPosy != -(np.Infinity), -np.Infinity, mtPosy)

        mtshddim = np.shape(mtPosy)
        mtShadowing = np.random.randn(mtshddim[0], mtshddim[1]) * self.dSigmaShad

        for iBsD in range(7):
            singleBsPos = (mtPosx + 1j * mtPosy) - vtBs[iBsD]
            mtPosEachBS.append(singleBsPos)

            singleBsdist = np.absolute(mtPosEachBS[iBsD])
            mtDistEachBs.append(singleBsdist)

            mtDistEachBs[iBsD][mtDistEachBs[iBsD] < self.dRMin] = self.dRMin

            singlePldB = okumura_hata(self.dFc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)
            mtPldB.append(singlePldB)

            singleshd = self.dPtdBm - np.asarray(mtPldB[iBsD])
            singleshd = singleshd - mtShadowing
            mtPowerEachBSShaddBm.append(singleshd)

            singlePowerBSdBm = self.dPtdBm - np.asarray(mtPldB[iBsD])
            mtPowerEachBSdBm.append(singlePowerBSdBm)

            mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm[iBsD])
            mtPowerFinalShaddBm = np.maximum(mtPowerFinalShaddBm, mtPowerEachBSShaddBm[iBsD])

        plt.figure()
        plt.pcolor(mtPosx, mtPosy, mtPowerFinalShaddBm, cmap='hsv')
        plt.colorbar(label="Potência em DB")
        self.draw_deploy(self.dR, vtBs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs com shadowing')

        plt.figure()
        plt.pcolor(mtPosx, mtPosy, mtPowerFinaldBm, cmap='hsv')
        plt.colorbar(label="Potência em DB")
        self.draw_deploy(self.dR, vtBs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs sem shadowing')

        plt.show()

    def generate_mt_pontos_medicao(self, dDimXOri, dDimYOri):

        """
        Metodos que gera uma matriz de pontos de medicao

        :param dDimXOri:
        :param dDimYOri:
        :return: mt_pontos_medicao (Matriz com os pontos de medicao para o sombreamento)
        """
        d_dim_y = np.ceil(dDimYOri + (dDimYOri % self.dPasso))
        d_dim_x = np.ceil(dDimXOri + (dDimXOri % self.dPasso))
        mt_posx, mt_posy = np.meshgrid(np.arange(0, d_dim_x + self.dPasso, self.dPasso),
                                       np.arange(0, d_dim_y, self.dPasso))
        mt_pontos_medicao = mt_posx + 1j * mt_posy

        return mt_pontos_medicao

    def genetare_mt_x_y_pos(self, dDimXOri, dDimYOri):

        """

        :param dDimXOri:
        :param dDimYOri:
        :return: mt_posx, mt_posy (tuple)
        """

        d_dim_y = np.ceil(dDimYOri + (dDimYOri % self.dPasso))
        d_dim_x = np.ceil(dDimXOri + (dDimXOri % self.dPasso))

        mt_posx, mt_posy = np.meshgrid(np.arange(0, d_dim_x + self.dPasso, self.dPasso), np.arange(0, d_dim_y,
                                                                                                   self.dPasso))

        return mt_posx, mt_posy

    def f_corr_shadowing(self, mt_points, dAlphaCorr, d_dim_x_ori, d_dim_y_ori):

        """

        :param mt_points:
        :param dAlphaCorr:
        :param d_dim_x_ori:
        :param d_dim_y_ori:
        :return: mtShadowingCorr: matriz de sobreamento correlacionado
        """

        dimXS = math.ceil(d_dim_x_ori + (d_dim_x_ori % self.dShad))
        dimYS = math.ceil(d_dim_y_ori + (d_dim_y_ori % self.dShad))
        mtPosxShad, mtPosyShad = np.meshgrid(np.arange(0, dimXS + self.dShad, self.dShad),
                                             np.arange(0, dimYS + self.dShad, self.dShad))

        mtShadowingSamples = []
        for erb in range(8):
            ShadowingSamples = self.dSigmaShad * np.random.randn(np.shape(mtPosyShad)[0], np.shape(mtPosyShad)[1])
            mtShadowingSamples.append(ShadowingSamples)
        mtShadowingSamples = np.asarray(mtShadowingSamples)

        sizeL, sizeC = np.shape(mt_points)  # pt4

        mtShadowingCorr = np.empty([7, sizeL, sizeC])
        for linha in range(sizeL):  # pt4
            for coluna in range(sizeC):  # pt4
                dShadPoint = mt_points[linha, coluna]  # pt4

                dxIndexP1 = np.real(dShadPoint) / self.dShad
                dyIndexP1 = np.imag(dShadPoint) / self.dShad

                if dxIndexP1 % 1 == 0 and dyIndexP1 % 1 == 0:
                    dxIndexP1 = math.floor(dxIndexP1) + 1
                    dyIndexP1 = math.floor(dyIndexP1) + 1

                    shadowingC = mtShadowingSamples[7][dyIndexP1 - 1][dxIndexP1 - 1]  # pt3

                    for i in range(7):
                        mtShadowingERB = mtShadowingSamples[i][dyIndexP1 - 1][dxIndexP1 - 1]
                        mtShadowingCorr[i][linha][coluna] = np.sqrt(dAlphaCorr) * shadowingC + np.sqrt(
                            1 - dAlphaCorr) * mtShadowingERB

                else:
                    dxIndexP1 = math.floor(dxIndexP1) + 1
                    dyIndexP1 = math.floor(dyIndexP1) + 1
                    if dxIndexP1 == np.shape(mtPosyShad)[0] and dyIndexP1 == np.shape(mtPosyShad)[1]:
                        dxIndexP2 = dxIndexP1 - 1
                        dyIndexP2 = dyIndexP1
                        dxIndexP4 = dxIndexP1 - 1
                        dyIndexP4 = dyIndexP1 - 1
                        dxIndexP3 = dxIndexP1
                        dyIndexP3 = dyIndexP1 - 1

                    elif dyIndexP1 == np.shape(mtPosyShad)[0]:
                        dxIndexP2 = dxIndexP1 + 1
                        dyIndexP2 = dyIndexP1
                        dxIndexP4 = dxIndexP1 + 1
                        dyIndexP4 = dyIndexP1 - 1
                        dxIndexP3 = dxIndexP1
                        dyIndexP3 = dyIndexP1 - 1

                    elif dxIndexP1 == np.shape(mtPosyShad)[1]:
                        dxIndexP2 = dxIndexP1 - 1
                        dyIndexP2 = dyIndexP1
                        dxIndexP4 = dxIndexP1 - 1
                        dyIndexP4 = dyIndexP1 + 1
                        dxIndexP3 = dxIndexP1
                        dyIndexP3 = dyIndexP1 + 1

                    else:
                        dxIndexP2 = dxIndexP1 + 1
                        dyIndexP2 = dyIndexP1
                        dxIndexP4 = dxIndexP1 + 1
                        dyIndexP4 = dyIndexP1 + 1
                        dxIndexP3 = dxIndexP1
                        dyIndexP3 = dyIndexP1 + 1

                    distX = (dShadPoint.real % self.dShad) / self.dShad
                    distY = (dShadPoint.imag % self.dShad) / self.dShad

                    stdNormalFactor = np.sqrt(
                        (1 - 2 * distY + 2 * (distY ** 2)) * (1 - 2 * distX + 2 * (distX ** 2)))  # pt3

                    Sample1 = mtShadowingSamples[7][dyIndexP1 - 1, dxIndexP1 - 1]  # pt3
                    Sample2 = mtShadowingSamples[7][dyIndexP2 - 1, dxIndexP2 - 1]  # pt3
                    Sample3 = mtShadowingSamples[7][dyIndexP3 - 1, dxIndexP3 - 1]  # pt3
                    Sample4 = mtShadowingSamples[7][dyIndexP4 - 1, dxIndexP4 - 1]  # pt3
                    shadowingC = ((1 - distY) * (Sample1 * (1 - distX) + Sample2 * distX) +  # pt4 #pt4
                                  distY * (Sample3 * (1 - distX) + Sample4 * distX)) / stdNormalFactor  # pt4 #pt4

                    for i in range(7):
                        Sample1 = mtShadowingSamples[i][dyIndexP1 - 1, dxIndexP1 - 1]  # pt3
                        Sample2 = mtShadowingSamples[i][dyIndexP2 - 1, dxIndexP2 - 1]  # pt3
                        Sample3 = mtShadowingSamples[i][dyIndexP3 - 1, dxIndexP3 - 1]  # pt3
                        Sample4 = mtShadowingSamples[i][dyIndexP4 - 1, dxIndexP4 - 1]  # pt3
                        shadowingERB = ((1 - distY) * (Sample1 * (1 - distX) + Sample2 * distX) +  # pt4 #pt4
                                        distY * (Sample3 * (
                                        1 - distX) + Sample4 * distX)) / stdNormalFactor  # pot4 #pt4
                        mtShadowingCorr[i][linha][coluna] = np.sqrt(dAlphaCorr) * shadowingC + np.sqrt(
                            1 - dAlphaCorr) * shadowingERB

        # surfplot(mtPosyShad, mtPosxShad, mtShadowingCorr)
        return mtShadowingCorr

    def draw_graphs_for_corr_shadowing(self, dalphacorr, dDimXOri, dDimYOri):

        vtbs = [0]

        for ibs in range(6):
            bs = self.dR * np.sqrt(3) * np.exp((ibs * np.pi / 3 + self.dOffset) * 1j)
            vtbs.append(bs)

        vtbs = np.asarray(vtbs)
        vtbs = vtbs + ((dDimXOri / 2) + (1j * dDimYOri / 2))

        mt_posx, mt_posy = self.genetare_mt_x_y_pos(dDimXOri,
                                                    dDimYOri)

        mt_pontos_medicao = self.generate_mt_pontos_medicao(dDimXOri, dDimYOri)
        mt_shadowing_corr = self.f_corr_shadowing(mt_pontos_medicao,  dalphacorr, dDimXOri, dDimYOri)

        mt_pos_each_bs = []
        mt_dist_each_bs = []

        mt_power_each_bs_shadd_bm = []
        mt_power_each_b_sd_bm = []
        mt_power_each_bs_shad_corrd_bm = []
        mt_pld_b = []

        mt_power_finald_bm = np.where(mt_posy != -np.Infinity, -np.Infinity, mt_posy)
        mt_power_final_shadd_bm = np.where(mt_posy != -np.Infinity, -np.Infinity, mt_posy)
        mt_power_final_shad_corr_dbm = np.where(mt_posy != -np.Infinity, -np.Infinity, mt_posy)

        mtshddim = np.shape(mt_posy)
        mt_shadowing = np.random.randn(mtshddim[0], mtshddim[1]) * self.dSigmaShad

        for iBsD in range(7):
            single_bs_pos = (mt_posx + 1j * mt_posy) - vtbs[iBsD]
            mt_pos_each_bs.append(single_bs_pos)

            single_bsdist = np.absolute(mt_pos_each_bs[iBsD])
            mt_dist_each_bs.append(single_bsdist)

            mt_dist_each_bs[iBsD][mt_dist_each_bs[iBsD] < self.dRMin] = self.dRMin

            single_pld_b = okumura_hata(self.dFc, self.dHBs, mt_dist_each_bs[iBsD], self.dAhm)
            mt_pld_b.append(single_pld_b)

            singleshd = self.dPtdBm - np.asarray(mt_pld_b[iBsD])
            singleshd = singleshd + mt_shadowing
            mt_power_each_bs_shadd_bm.append(singleshd)

            single_power_b_sd_bm = self.dPtdBm - mt_pld_b[iBsD]
            mt_power_each_b_sd_bm.append(single_power_b_sd_bm)

            single_power_bs_cordbm = self.dPtdBm - np.asarray(mt_pld_b[iBsD]) + mt_shadowing_corr[iBsD]
            mt_power_each_bs_shad_corrd_bm.append(single_power_bs_cordbm)

            mt_power_finald_bm = np.maximum(mt_power_finald_bm, mt_power_each_b_sd_bm[iBsD])
            mt_power_final_shadd_bm = np.maximum(mt_power_final_shadd_bm, mt_power_each_bs_shadd_bm[iBsD])
            mt_power_final_shad_corr_dbm = np.maximum(mt_power_final_shad_corr_dbm,
                                                      mt_power_each_bs_shad_corrd_bm[iBsD])

        plt.figure()
        plt.pcolor(mt_posx, mt_posy, mt_power_finald_bm, cmap='hsv')
        plt.colorbar(label="Potência em DB")
        Functions.draw_deploy(self.dR, vtbs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs sem shadowing')

        plt.figure()
        plt.pcolor(mt_posx, mt_posy, mt_power_final_shadd_bm, cmap='hsv')
        plt.colorbar(label="Potência em DB")
        Functions.draw_deploy(self.dR, vtbs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs com shadowing')

        plt.figure()
        plt.pcolor(mt_posx, mt_posy, mt_power_final_shad_corr_dbm, cmap='hsv')
        plt.colorbar(label="Potência em DB")
        Functions.draw_deploy(self.dR, vtbs)
        plt.axis('equal')
        plt.title('Todas as 7 ERBs com shadowing correlacionado')

        plt.show()

    def verify_std_deviation(self, d_alpha_corr, dDimXOri, dDimYOri):

        """

        :param d_alpha_corr:
        :param dDimXOri:
        :param dDimYOri:
        :return: std_deviation (float): desvio padrao da matriz de sombreamento correlacionado.
        """

        mt_pontos_medicao = self.generate_mt_pontos_medicao(dDimXOri, dDimYOri)

        mt_shadowing_corr = self.f_corr_shadowing(mt_pontos_medicao, d_alpha_corr, dDimXOri, dDimYOri)

        std_deviation = np.std(mt_shadowing_corr)

        return std_deviation

    def rem_microcell_outage(self, model, sensitivy, dfc):

        vtbs_micro = self.micro_cells_positions()[0]
        vtbs = self.micro_cells_positions()[1]

        dPasso = np.ceil(self.dR / 50)
        dRMin = dPasso
        dDimX = 5 * self.dR
        dDimY = 6 * np.sqrt(3 / 4) * self.dR

        dDimX += (dDimX % dPasso)
        dDimY += (dDimY % dPasso)

        x = np.arange(0, dDimX, dPasso)
        y = np.arange(0, dDimY, dPasso)

        mtPosx, mtPosy = np.meshgrid(x, y)

        mtPosEachBS = []
        mtDistEachBs = []
        mtPowerEachBSdBm = []
        mtPldB = []
        mtPowerFinaldBm = np.where(mtPosy != -np.Infinity, -np.Infinity, mtPosy)
        mtPowerFinaldBm = np.array(mtPowerFinaldBm, dtype=float)

        for iBsD in range(7):

            singleBsPos = (mtPosx + 1j * mtPosy) - vtbs[iBsD]
            mtPosEachBS.append(singleBsPos)

            singleBsdist = np.absolute(mtPosEachBS[iBsD])
            mtDistEachBs.append(singleBsdist)

            mtDistEachBs[iBsD][mtDistEachBs[iBsD] < dRMin] = dRMin

            if model == 1:
                singlePldB = okumura_hata(
                    dfc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)
            else:
                singlePldB = cost_231(
                    dfc, self.dHBs, mtDistEachBs[iBsD], self.dAhm)

            mtPldB.append(singlePldB)

            singlePowerBSdBm = self.dPtdBm - np.asarray(mtPldB[iBsD])
            mtPowerEachBSdBm.append(singlePowerBSdBm)

            mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm[iBsD])

        dict_out_rate = Functions.two_colors_mt(mtPowerFinaldBm, sensitivy)

        mtPosEachBS_micro = []
        mtDistEachBs_micro = []
        mtPowerEachBSdBm_micro = []
        mt_pl_db_micro = []

        for iBsD in range(6):

            singleBsMicroPos = (mtPosx + 1j * mtPosy) - vtbs_micro[iBsD]
            mtPosEachBS_micro.append(singleBsMicroPos)

            singleBsMicrodist = np.absolute(mtPosEachBS_micro[iBsD])
            mtDistEachBs_micro.append(singleBsMicrodist)

            mtDistEachBs_micro[iBsD][mtDistEachBs_micro[iBsD] < dRMin] = dRMin

            singlePldB_micro = cost_walfish_ikegami(dfc, mtDistEachBs_micro[iBsD])

            mt_pl_db_micro.append(singlePldB_micro)

            singlePowerBSdBm_micro = self.dPtdBmMicro - np.asarray(mt_pl_db_micro[iBsD])
            mtPowerEachBSdBm_micro.append(singlePowerBSdBm_micro)

            mtPowerFinaldBm = np.maximum(mtPowerFinaldBm, mtPowerEachBSdBm_micro[iBsD])

        dict_out_rate_with_mcll = Functions.two_colors_mt(mtPowerFinaldBm, sensitivy)

        plt.figure()
        plt.pcolor(mtPosx, mtPosy, dict_out_rate['outage_matrix'])
        Functions.draw_deploy(self.dR, vtbs)
        plt.axis('equal')
        plt.title(f'REM sem microcelulas {dfc} MHz')

        plt.figure()
        plt.pcolor(mtPosx, mtPosy, dict_out_rate_with_mcll['outage_matrix'])
        Functions.draw_deploy(self.dR, vtbs)
        Functions.draw_deploy(self.drMicro, vtbs_micro)
        plt.axis('equal')
        plt.title(f'REM com microcelulas {dfc} MHz')
        # plt.show()

    @staticmethod
    def two_colors_mt(mtpowerfinaldbm, sensitivy):

        """

        :param mtpowerfinaldbm:
        :param sensitivy:
        :return:
        """
        count_zero = 0
        count_one = 0

        d_sizel, d_sizec = np.shape(mtpowerfinaldbm)
        two_colors_matrix = np.empty((d_sizel, d_sizec), dtype=int)

        for linha in range(d_sizel):
            for coluna in range(d_sizec):
                if mtpowerfinaldbm[linha][coluna] > sensitivy:
                    two_colors_matrix[linha][coluna] = 0
                    count_zero += 1
                else:
                    two_colors_matrix[linha][coluna] = 1
                    count_one += 1

        return {'outage_matrix':  two_colors_matrix, 'ones': count_one, 'zeros': count_zero}

    def micro_cells_positions(self):

        """
            Retorna as posicoes dos centros da macrocelulas e das microcelulas

        """

        vt_bs = np.array([0], dtype=complex)
        vt_bs_micro = np.array([0], dtype=complex)

        for iBs in range(1, 7):
            vt_bs = np.append(vt_bs, self.dR * np.sqrt(3) * pow(np.e, (1j * ((iBs - 2) * np.pi / 3 + self.dOffset))))
        vt_bs = vt_bs + (self.dDimX / 2 + 1j * self.dDimY / 2)

        for iBs in range(6):
            vt_bs_micro = np.append(vt_bs_micro, self.dR * pow(np.e, (1j * ((iBs - 2) * np.pi / 3 + np.pi))))
        vt_bs_micro = vt_bs_micro + (self.dDimX / 2 + 1j * self.dDimY / 2)
        vt_bs_micro = np.delete(vt_bs_micro, 0)

        return vt_bs_micro, vt_bs

    def bar_graph_calc(self, dict_outage):

        total_points = np.shape(dict_outage['outage_matrix'])[0] * np.shape(dict_outage['outage_matrix'])[1]
        percentile_zeros = (dict_outage['zeros'] / total_points) * 100
        percentile_ones = (dict_outage['ones'] / total_points) * 100
        print(f'Area sem outage: {percentile_zeros:.2f}%')
        print(f'Area com outage: {percentile_ones:.2f}%')


if __name__ == '__main__':

    dR = 500
    drMicro = 100
    dShad = 50
    dPasso = 10
    dFc = 800
    dSigmaShad = 8
    dAlphaCorr = 0.5
    dRMin = dPasso
    dIntersiteDistance = 2 * np.sqrt(3 / 4) * dR
    dDimXOri = 5 * dR
    dDimYOri = 6 * np.sqrt(3 / 4) * dR
    dPtdBm = 21
    dPtdBmMicro = 20
    dPtLinear = 10 ** (dPtdBm / 10) * 1e-3
    dHMob = 1.5
    dHBs = 32
    dAhm = 3.2 * ((np.log10(11.75 * dHMob)) ** 2) - 4.97
    dOffset = np.pi / 6

    operation = Functions(dR, dFc, dShad, dSigmaShad, dPtdBm, dHMob, dHBs, dOffset, drMicro, dPtdBmMicro)

    carriers = [800, 1800, 2100]

    for dFc in carriers:

        operation.rem_microcell_outage(1, -90, dFc)

    plt.show()

    system.exit(0)
