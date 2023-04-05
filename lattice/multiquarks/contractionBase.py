# distillation contraction base 
# @shichunjiang
# March 2022

import numpy as np
import cupy
import os
from typing import  Union
import random
from hashlib import md5
from ..insertion.gamma import gamma
# from utils import mytimer

class LatticeDesc:
    def __init__(self) -> None:
        self.dtype_all = np.dtype("<c16")
        self.dtype_size = self.dtype_all.itemsize
        print(f"data type = {self.dtype_all}, itersize = {self.dtype_size}.")

        self.Nvec = 50          #you used vec number 
        self.totNvec = 50       #total vec number
        self.nx = 16
        self.ny = 16
        self.nz = 16
        self.TimeSlice = 128
        self.spin = 4
        self.sets = 16  # (TimeSlice%Sets) should be zero
        # Sets=31 #(TimeSlice%Sets) should be zero
        self.tSub = int(self.TimeSlice/self.sets)


# Euclidean Gamma Matrix: 
# gamma_i(Euclid) = (-i * gamma_i(Minkov, Dirac rep)).conj()  
# gamma_0(Euclid) = gamma_0(Minkov)  
Gamma0 = gamma(0)
Gamma1 = gamma(1)
Gamma2 = gamma(2)
Gamma3 = gamma(4)
Gamma4 = gamma(8)
Gamma5 = gamma(15)

gammaDict = {
    "a0": [Gamma0],
    "b0": [Gamma4],
    "pi": [Gamma5],
    "pi2": [Gamma4 @ Gamma5],
    "rho":  [Gamma1, Gamma2, Gamma3],
    "rho2":  [Gamma4 @ Gamma1,
               Gamma4 @ Gamma2,
               Gamma4 @ Gamma3],
    "a1": [Gamma1 @ Gamma5,
            Gamma2 @ Gamma5,
            Gamma3 @ Gamma5],
    "b1":  [Gamma2 @ Gamma3,
             Gamma3 @ Gamma1,
             Gamma1 @ Gamma2],

    "identity": (Gamma0),
    "gamma1": (Gamma1),
    "gamma2": (Gamma2),
    "gamma3": (Gamma3),
    "gamma4": (Gamma4),
    "gamma5": (Gamma5),
    
    "gamma23": (Gamma2 @ Gamma3),
    "gamma31": (Gamma3 @ Gamma1),
    "gamma12": (Gamma1 @ Gamma2),
                 
    "gamma5gamma1": (Gamma5 @ Gamma1),
    "gamma5gamma2": (Gamma5 @ Gamma2),
    "gamma5gamma3": (Gamma5 @ Gamma3),

    "gamma2gamma3": (Gamma2 @ Gamma3),
    "gamma3gamma1": (Gamma3 @ Gamma1),
    "gamma1gamma2": (Gamma1 @ Gamma2),

    "gamma4gamma1": (Gamma4 @ Gamma1),
    "gamma4gamma2": (Gamma4 @ Gamma2),
    "gamma4gamma3": (Gamma4 @ Gamma3),
    "gamma4gamma5": (Gamma4 @ Gamma5),

    "gamma4gamma5gamma1": (Gamma4 @ Gamma5 @ Gamma1),
    "gamma4gamma5gamma2": (Gamma4 @ Gamma5 @ Gamma2),
    "gamma4gamma5gamma3": (Gamma4 @ Gamma5 @ Gamma3),
}

gammaSum3Dict = {
    "a0": Gamma0,
    "b0": Gamma4,
    "pi": Gamma5,
    "pi2": Gamma4 @ Gamma5,
    "rho":  (Gamma1 + Gamma2 + Gamma3),
    "rho2":(Gamma4 @ Gamma1 +
             Gamma4 @ Gamma2 +
             Gamma4 @ Gamma3),
    "a1":  (Gamma1 @ Gamma5 +
             Gamma2 @ Gamma5 +
             Gamma3 @ Gamma5),
    "b1":  (Gamma2 @ Gamma3 +
             Gamma3 @ Gamma1 +
             Gamma1 @ Gamma2),

    "identity": (Gamma0),
    "gamma1": (Gamma1),
    "gamma2": (Gamma2),
    "gamma3": (Gamma3),
    "gamma4": (Gamma4),
    "gamma5": (Gamma5),

    "gamma23": (Gamma2 @ Gamma3),
    "gamma31": (Gamma3 @ Gamma1),
    "gamma12": (Gamma1 @ Gamma2),

    "gamma5gamma1": (Gamma5 @ Gamma1),
    "gamma5gamma2": (Gamma5 @ Gamma2),
    "gamma5gamma3": (Gamma5 @ Gamma3),

    "gamma2gamma3": (Gamma2 @ Gamma3),
    "gamma3gamma1": (Gamma3 @ Gamma1),
    "gamma1gamma2": (Gamma1 @ Gamma2),

    "gamma4gamma1": (Gamma4 @ Gamma1),
    "gamma4gamma2": (Gamma4 @ Gamma2),
    "gamma4gamma3": (Gamma4 @ Gamma3),
    "gamma4gamma5": (Gamma4 @ Gamma5),

    "gamma4gamma5gamma1": (Gamma4 @ Gamma5 @ Gamma1),
    "gamma4gamma5gamma2": (Gamma4 @ Gamma5 @ Gamma2),
    "gamma4gamma5gamma3": (Gamma4 @ Gamma5 @ Gamma3),
}

from ..insertion import momDict_mom1, momDict_mom3, momDict_mom9
from ..insertion import derivDict as derivDict

def momlist(n, que):
    imax = int(np.sqrt(n))
    mom = []
    # num = 0
    if n==0:
        momDict0 = momDict_mom1
    elif n==3:
        momDict0 = momDict_mom3                          
    elif n==9:
        momDict0 = momDict_mom9
    else:
        raise ValueError(f"Unsupport mom = {n}")                           
    for i in range(-imax, imax+1, 1):
        for j in range(-imax, imax+1, 1):
            for l in range(-imax, imax+1, 1):
                if(i**2+j**2+l**2 <= n):
                    if(i**2+j**2+l**2 == que):
                        strKey = f"{l} {j} {i}"
                        mom.append(momDict0[strKey])
                        print(f"mom list: {momDict0[strKey]}:({l},{j},{i})")
                    # num = num+1
    return mom


class Elementals(LatticeDesc):
    def __init__(self, elementalInfo: dict, dtype: Union[str, np.dtype]) -> None:
        super().__init__()
        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            raise TypeError(f"Unkown dtype {dtype}")
        self.elementalName = elementalInfo["name"]
        self.elementalFile = elementalInfo["path"]
        self.dataDnum = elementalInfo["dataDnum"]
        self.derivType = elementalInfo["derivType"]
        self.dataMomMax = elementalInfo["dataMomMax"]
        self.dataMomNum = elementalInfo["dataMomNum"]
        self.elemDataShape = elementalInfo["elemDataShape"]

        # mom imput
        self.mom = elementalInfo["mom"]

        # self.termNum is the number of gamma term!!
        # Gamma just pass info, do not effect loading data
        self.Gamma = elementalInfo["Gamma"]
        if "coeff" in elementalInfo.keys():
            self.coeff = elementalInfo["coeff"]
            if isinstance(self.coeff,int) or isinstance(self.coeff, float) or isinstance(self.coeff, complex):
                self.coeff = [self.coeff]
        else:
            self.coeff = [1]
        if isinstance(self.derivType, str): 
            self.derivType = [self.derivType]
        if isinstance(self.Gamma, str):
            self.Gamma = [self.Gamma]
        self.termNum = len(self.Gamma)
        if len(self.derivType) != len(self.Gamma) or len(self.derivType) != len(self.coeff):
            raise ValueError("Input term num not consistant!")

        self.data = None


    def load(self, isLoadAll=True, isHalfMom=False):
        momPosList = momlist(self.dataMomMax, self.mom)
        self.momPairNum = len(momPosList) % 2 + len(momPosList)//2
        if isHalfMom:
            self.calcMomNum = len(momPosList) % 2 + len(momPosList)//2
        else:
            self.calcMomNum = len(momPosList) 
        print(f"DBUG: mom list: {momPosList}, calMomNum = {self.calcMomNum}")

        self.data = cupy.zeros(shape=(self.calcMomNum, self.termNum, 2, self.TimeSlice, self.Nvec, self.Nvec), dtype=self.dtype) 
        # !!! Note !!! 
        # self.momPairNum : momentum pair number, which multi 2 equal to |p| number when mom not 0
        # self.calcMomNum = len(momPosList) 
        # self.termNum: gamma term number
        # factor 2: two particle
        
        timer0 = mytimer(
            f"loaded elementals({self.elementalFile})", self.TimeSlice*self.Nvec*self.Nvec)
        if isLoadAll:
            if self.elementalFile.endswith(".npy"):
                elemCache = np.load(self.elementalFile).reshape(self.elemDataShape)
                print(f"Note: load .npy file, elem shape:{self.elemDataShape}.")
            else:
                # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
                elemCache = np.fromfile(self.elementalFile, dtype=self.dtype).reshape(self.elemDataShape)
                print(f"Note: load binary file, elem shape:{self.elemDataShape}.")
            for iterm in range(self.termNum):
                print(f"Use Deriv : {derivDict[self.derivType[iterm]]}, dataMomNum: {self.dataMomNum}, dataDnum: {self.dataDnum}")

            # if self.elementalFile[-4:]==".npy":
            #     # In jiangxy's style, shape = [Ntime, deriv, mom, Nvec, Nvec]
            #     hostTmp = np.load(self.elementalFile, "r")[:,derivDict[self.derivType[iterm]],:,:,:].transpose((1,0,2,3))
            #     allMomData = cupy.asarray(hostTmp)
            #     del hostTmp
            #     print(f"Note: load .npy file, elem shape:{allMomData.shape}.")
            # else:
            #     # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
            #     count = self.dataMomNum*self.TimeSlice*self.Nvec*self.Nvec
            #     offset = derivDict[self.derivType[iterm]] * count * self.dtype_size
            #     allMomData = cupy.fromfile(self.elementalFile, dtype=self.dtype, offset=offset, count=count).reshape(
            #         self.dataMomNum, self.TimeSlice, self.Nvec, self.Nvec)
            #     print(f"Note: load binary file, elem shape:{allMomData.shape}.")


                for k in range(self.calcMomNum):
                    # #!!!!!!!!!!!!!!!!!CAUTION!!!!!!!!!
                    # In jiangxy's style, shape = [Ntime, deriv, mom, Nvec, Nvec]
                    # 20220401: In xyjiang's test, memmap works better.
                    # self.data[k, iterm, 0] = cupy.asarray(elemCache[:,derivDict[self.derivType[iterm]],momPosList[k],:,:])
                    # self.data[k, iterm, 1] = cupy.asarray(elemCache[:,derivDict[self.derivType[iterm]],momPosList[-k-1],:,:])
                    self.data[k, iterm, 0] = cupy.asarray(elemCache[derivDict[self.derivType[iterm]],momPosList[k],:,:,:])
                    self.data[k, iterm, 1] = cupy.asarray(elemCache[derivDict[self.derivType[iterm]],momPosList[-k-1],:,:,:])


        else: 
            for k in range(self.calcMomNum):
                for iterm in range(self.termNum):
                    # A = 0,  B = 1
                    print(f"DBUG: derivDict[self.derivType[iterm]] = {derivDict[self.derivType[iterm]]}, k = {k}, momPosList[k] = {momPosList[k]}")
                    if self.elementalFile.endswith(".npy"):
                        # In jiangxy's style, shape = [Ntime, deriv, mom, Nvec, Nvec]
                        # 20220401: In xyjiang's test, memmap works better.
                        elemCache = np.memmap(self.elementalFile, dtype=self.dtype, mode="r", shape=self.elemDataShape)[derivDict[self.derivType[iterm]],momPosList[k],:,:,:]
                        self.data[k, iterm, 0] = cupy.asarray(elemCache)
                        elemCache = np.memmap(self.elementalFile, dtype=self.dtype, mode="r", shape=self.elemDataShape)[derivDict[self.derivType[iterm]],momPosList[-k-1],:,:,:]
                        self.data[k, iterm, 1] = cupy.asarray(elemCache)

                        # tmpLoad0 = cupy.zeros(shape=( self.TimeSlice, self.Nvec, self.Nvec), dtype=self.dtype)
                        # tmpLoad1 = cupy.zeros(shape=( self.TimeSlice, self.Nvec, self.Nvec), dtype=self.dtype)
                        # for it in range(self.TimeSlice):
                        #     count = self.Nvec*self.Nvec
                        #     offset = it * (13  * 123) * count * self.dtype_size + (derivDict[self.derivType[iterm]] * self.dataMomNum + momPosList[k]) * count * self.dtype_size 
                        #     tmpLoad0[it] = cupy.fromfile(self.elementalFile, 
                        #                         dtype=self.dtype,
                        #                         offset=offset + 128, 
                        #                         count=count
                        #                         ).reshape(
                        #                                     self.Nvec, self.Nvec)                       
                        #     offset = it * (13  * 123) * count * self.dtype_size + (derivDict[self.derivType[iterm]] * self.dataMomNum + momPosList[-k-1]) * count * self.dtype_size 
                        #     tmpLoad1[it] = cupy.fromfile(self.elementalFile, 
                        #                         dtype=self.dtype,
                        #                         offset=offset + 128, 
                        #                         count=count
                        #                         ).reshape(
                        #         self.Nvec, self.Nvec)
                        # self.data[k, iterm, 0] = tmpLoad0
                        # self.data[k, iterm, 1] = tmpLoad1
                        print(f"Note: load .npy file, elem shape:{self.data[k, iterm, 0].shape}.")
                    else:
                        # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
                        count = self.TimeSlice*self.Nvec*self.Nvec
                        offset = (derivDict[self.derivType[iterm]] * self.dataMomNum + momPosList[k]) * count * self.dtype_size 
                        self.data[k, iterm, 0] = cupy.fromfile(self.elementalFile, 
                                            dtype=self.dtype,
                                            offset=offset, 
                                            count=count
                                            ).reshape(
                                                        self.TimeSlice, self.Nvec, self.Nvec)
                        offset = (derivDict[self.derivType[iterm]] * self.dataMomNum + momPosList[-k-1]) * count * self.dtype_size 
                        self.data[k, iterm, 1] = cupy.fromfile(self.elementalFile, 
                                            dtype=self.dtype,
                                            offset=offset, 
                                            count=count
                                            ).reshape(
                                                        self.TimeSlice, self.Nvec, self.Nvec)
                        print(f"Note: load binary file, elem shape:{self.data[k, iterm, 0].shape}.")
        timer0.end()


class ElementalsAB(LatticeDesc):
    def __init__(self, elementalInfo: dict, dtype: Union[str, np.dtype]) -> None:
        super().__init__()
        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            raise TypeError(f"Unkown dtype {dtype}")
        self.elementalName = elementalInfo["name"]
        self.elementalFile = elementalInfo["path"]
        self.dataDnum = elementalInfo["dataDnum"]
        self.dataMomMax = elementalInfo["dataMomMax"]
        self.dataMomNum = elementalInfo["dataMomNum"]
        self.elemDataShape = elementalInfo["elemDataShape"]

        # mom imput
        self.mom = elementalInfo["mom"]

        self.derivTypeA = elementalInfo["derivTypeA"]
        self.derivTypeB = elementalInfo["derivTypeB"]

        # self.termNum is the number of gamma term!!
        # Gamma just pass info, do not effect loading data
        self.GammaA = elementalInfo["GammaA"]
        self.GammaB = elementalInfo["GammaB"]
        if "coeffA" in elementalInfo.keys():
            self.coeffA = elementalInfo["coeffA"]
            if isinstance(self.coeffA,int) or isinstance(self.coeffA, float) or isinstance(self.coeffA, complex):
                self.coeffA = [self.coeffA]
        else:
            self.coeffA = [1]

        if "coeffB" in elementalInfo.keys():
            self.coeffB = elementalInfo["coeffB"]
            if isinstance(self.coeffB,int) or isinstance(self.coeffB, float) or isinstance(self.coeffB, complex):
                self.coeffB = [self.coeffB]
        else:
            self.coeffB = [1]

        if isinstance(self.derivTypeA, str): 
            self.derivTypeA = [self.derivTypeA]
        if isinstance(self.derivTypeB, str): 
            self.derivTypeB = [self.derivTypeB]

        if isinstance(self.GammaA, str):
            self.GammaA = [self.GammaA]
        if isinstance(self.GammaB, str):
            self.GammaB = [self.GammaB]

        self.termNumA = len(self.GammaA)
        self.termNumB = len(self.GammaB)
        if len(self.derivTypeA) != len(self.GammaA) or len(self.derivTypeA) != len(self.coeffA):
            raise ValueError("Input term num not consistant!")
        if len(self.derivTypeB) != len(self.GammaB) or len(self.derivTypeB) != len(self.coeffB):
            raise ValueError("Input term num not consistant!")
        self.data = None

    def load(self, isLoadAll=True, isHalfMom=False):
        momPosList = momlist(self.dataMomMax, self.mom)
        self.momPairNum = len(momPosList) % 2 + len(momPosList)//2
        if isHalfMom:
            self.calcMomNum = len(momPosList) % 2 + len(momPosList)//2
        else:
            self.calcMomNum = len(momPosList) 
        print(f"DBUG: mom list: {momPosList}, calcMomNum = {self.calcMomNum}")

        self.dataA = cupy.zeros(shape=(self.calcMomNum, self.termNumA, self.TimeSlice, self.Nvec, self.Nvec), dtype=self.dtype) 
        self.dataB = cupy.zeros(shape=(self.calcMomNum, self.termNumB, self.TimeSlice, self.Nvec, self.Nvec), dtype=self.dtype) 
        # !!! Note !!! 
        # self.momPairNum : momentum pair number, which multi 2 equal to |p| number when mom not 0
        # self.calcMomNum = len(momPosList) 
        # self.termNum: gamma term number
        # factor 2: two particle

        
        timer0 = mytimer(
            f"loaded elementals({self.elementalFile})", self.TimeSlice*self.Nvec*self.Nvec)


        if isLoadAll:
            if self.elementalFile.endswith(".npy"):
                elemCache = np.load(self.elementalFile).reshape(self.elemDataShape)
                print(f"Note: load .npy file, elem shape:{self.elemDataShape}.")
            else:
                # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
                elemCache = np.fromfile(self.elementalFile, dtype=self.dtype).reshape(self.elemDataShape)
                print(f"Note: load binary file, elem shape:{self.elemDataShape}.")
            for k in range(self.calcMomNum):
                for itermA in range(self.termNumA):
                    print(f"DBUG: derivDict[self.derivTypeA[iterm]] = {derivDict[self.derivTypeA[itermA]]}, k = {k}, momPosList[k] = {momPosList[k]}")                  
                    self.dataA[k, itermA] = cupy.asarray(elemCache[derivDict[self.derivTypeA[itermA]],momPosList[k],:,:,:])
                for itermB in range(self.termNumB):
                    print(f"DBUG: derivDict[self.derivTypeB[iterm]] = {derivDict[self.derivTypeB[itermB]]}, k = {k}, momPosList[k] = {momPosList[-k-1]}")                  
                    self.dataB[k, itermB] = cupy.asarray(elemCache[derivDict[self.derivTypeB[itermB]],momPosList[-k-1],:,:,:])

        else:    
            for k in range(self.calcMomNum):
                for iterm in range(self.termNumA):
                    print(f"DBUG: derivDict[self.derivTypeA[iterm]] = {derivDict[self.derivTypeA[iterm]]}, k = {k}, momPosList[k] = {momPosList[k]}")
                    if self.elementalFile.endswith(".npy"):
                        # If In jiangxy's style, shape = [Ntime, deriv, mom, Nvec, Nvec]
                        # 20220401: In xyjiang's test, memmap works better.
                        elemCache = np.memmap(self.elementalFile, dtype=self.dtype, mode="r", shape=self.elemDataShape)[derivDict[self.derivTypeA[iterm]],momPosList[k],:,:,:]
                        self.data[k, iterm] = cupy.asarray(elemCache)
                        print(f"Note: A: load .npy file, elem shape:{self.data[k, iterm].shape}.")
                    else:
                        # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
                        count = self.TimeSlice*self.Nvec*self.Nvec
                        offset = (derivDict[self.derivTypeA[iterm]] * self.dataMomNum + momPosList[k]) * count * self.dtype_size 
                        self.dataA[k, iterm] = cupy.fromfile(self.elementalFile, 
                                            dtype=self.dtype,
                                            offset=offset, 
                                            count=count
                                            ).reshape(
                                                        self.TimeSlice, self.Nvec, self.Nvec)
                        print(f"Note: A: load binary file, elem shape:{self.data[k, iterm].shape}.")
                for iterm in range(self.termNumB):
                    print(f"DBUG: derivDict[self.derivTypeB[iterm]] = {derivDict[self.derivTypeA[iterm]]}, k = {k}, momPosList[k] = {momPosList[k]}")
                    if self.elementalFile.endswith(".npy"):
                        # If In jiangxy's style, shape = [Ntime, deriv, mom, Nvec, Nvec]
                        # 20220401: In xyjiang's test, memmap works better.
                        elemCache = np.memmap(self.elementalFile, dtype=self.dtype, mode="r", shape=self.elemDataShape)[derivDict[self.derivType[iterm]],momPosList[-k-1],:,:,:]
                        self.data[k, iterm, 1] = cupy.asarray(elemCache)
                        print(f"Note: B: load .npy file, elem shape:{self.data[k, iterm].shape}.")
                    else:
                        # In Sunwei's style, shape = [deriv, mom, Ntime, Nvec, Nvec]
                        count = self.TimeSlice*self.Nvec*self.Nvec
                        offset = (derivDict[self.derivTypeB[iterm]] * self.dataMomNum + momPosList[-k-1]) * count * self.dtype_size 
                        self.dataB[k, iterm] = cupy.fromfile(self.elementalFile, 
                                            dtype=self.dtype,
                                            offset=offset, 
                                            count=count
                                            ).reshape(
                                                        self.TimeSlice, self.Nvec, self.Nvec)
                        print(f"Note: load binary file, elem shape:{self.data[k, iterm].shape}.")
            ###

            ### !!! Your Oerators MUST KEEP Hermitian !!!
            ## 3.1 2020 update: automatically keep Hermitian. 
        timer0.end()


class ForwardPeram(LatticeDesc):
    def __init__(self, perambulatorFile: str, dtype: Union[str, np.dtype]) -> None:
        super().__init__()
        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            raise TypeError(f"Unkown dtype {dtype}")
        self.data = None
        self.perambulatorFile = ""
        self.perambulatorFile = perambulatorFile

    def load(self, iload):
        count = self.tSub*self.TimeSlice*self.spin*self.spin*self.Nvec*self.Nvec
        time0 = mytimer(f"loading peram:{self.perambulatorFile}", count)
        self.data = cupy.fromfile(self.perambulatorFile, dtype=self.dtype, count=count,
                                  offset=iload*count*self.dtype_size).reshape(self.tSub, self.TimeSlice, self.spin, self.spin, self.Nvec, self.Nvec)
        time0.end()
        #cupy.zeros(shape=(tSub, TimeSlice, Ns, Ns, Nvec, Nvec),dtype=self.dtype)
        # tmp = cupy.einsum('ij,kljmno->klimno',Gamma5, self.data)

    def toBackwardPeram(self):
        ret = BackwardPeram(perambulatorFile="", dtype=self.dtype)
        ret.data = self.data
        return ret


class BackwardPeram(LatticeDesc):
    def __init__(self, perambulatorFile: str, dtype: Union[str, np.dtype]) -> None:
        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            raise TypeError(f"Unkown dtype {dtype}")
        self.data = None
        self.perambulatorFile = ""
        self.perambulatorFile = perambulatorFile

    # def load(self, iload):
    #     count = self.tSub*self.TimeSlice*self.spin*self.spin*self.Nvec*self.Nvec
    #     self.data = cupy.fromfile(self.perambulatorFile, dtype=self.dtype, count=count,
    #                               offset=iload*count*self.dtype_size).reshape(self.tSub, self.TimeSlice, self.spin, self.spin,self.Nvec, self.Nvec)
    #     self.data = cupy.conj(cupy.einsum(
    #         'ic,abcdef,dj->abjife', Gamma5, self.data, Gamma5))

class ContractionBase(LatticeDesc):
    def __init__(self, dtype: Union[str, np.dtype]) -> None:
        super().__init__()
        self.perambulatorFile = ""
        self.corrOut = None
        self.dimCorr = 1
        self.operatorVarList = []
        self.srcVarIndiceList = []
        self.sinkVarIndiceList = []
        self.corrSaveDir = "./"
        self.cfgKeyword = ""
        self.onlyDiag = False
        self.useNoisyElem = False
        self.varDim = None
        # self.eigenNumList = [70, 60, 50, 40, 30, 20, 10]
        # self.numNe = len(self.eigenNumList)

        if isinstance(dtype, str):
            self.dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            self.dtype = dtype
        else:
            raise TypeError(f"Unkown dtype {dtype}")

    def setPerambulatorFile(self, perambulatorFile: str):
        self.perambulatorFile = perambulatorFile

    def setCharmPerambulatorFile(self, perambulatorFile: str):
        self.charmPerambulatorFile = perambulatorFile

    def setLightPerambulatorFile(self, perambulatorFile: str):
        self.lightPerambulatorFile = perambulatorFile
 
    def setAllOperatorVarList(self, operatorVarList: list):
        self.operatorVarList = operatorVarList
        self.varDim = len(self.operatorVarList)
        for iElemSrc in range(self.varDim):
            for iElemSink in range(self.varDim):
                if self.onlyDiag and iElemSink!=iElemSrc:
                    print("NOTE: ONLY calc diag part!")
                    continue
                if True:  #iElemSink >= iElemSrc:   ## 20220218: calc all off-diag!!!
                    savepath=self.corrSaveDir+"/"+self.operatorVarList[iElemSrc]["name"]+"_"+self.operatorVarList[iElemSink]["name"]
                    if not os.path.exists(f"{savepath}/coor.{self.cfgKeyword}.npy"):
                        self.srcVarIndiceList.append(iElemSrc)
                        self.sinkVarIndiceList.append(iElemSink)
        self.calcLen = len(self.srcVarIndiceList)
        print(f"CALC INFO: vardim = {self.varDim}, calclen={self.calcLen}")

    def setCorrSaveDir(self, path: str):
        self.corrSaveDir = path

    def setCfgKeyword(self, keywords: str):
        self.cfgKeyword = keywords
        print(f"cfgkeword={self.cfgKeyword}")
