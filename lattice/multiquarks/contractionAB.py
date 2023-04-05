import numpy as np
import cupy
import os
from typing import Union
from .contractionBase import ContractionBase, ForwardPeram, BackwardPeram, LatticeDesc, ElementalsAB, Gamma, gammaSum3Dict
from ..timer import mytimer

class ContractionAB(ContractionBase):
    def __init__(self, dtype: Union[str, np.dtype]) -> None:
        super().__init__(dtype=dtype)
        ### Change self.saveCorrDim base on your designed contractio!!  
        self.saveCorrDim = 9

    def execution(self):
        if self.calcLen == 0:
            return None
        time_exe = mytimer("all execution.")

        # first time to load all used elementals
        timerElem = mytimer("Load All Elementals.")
        allElementalList = []
        for iElem in range(self.varDim):
            tmpElemtal1 = ElementalsAB(
                self.operatorVarList[iElem], dtype=self.dtype)
            tmpElemtal1.load(isLoadAll=True, isHalfMom=True)
            allElementalList.append(tmpElemtal1)
        timerElem.end()

        self.corrOut = np.zeros(shape=(
            self.calcLen, self.saveCorrDim, self.TimeSlice, self.TimeSlice), dtype=self.dtype)

        charmPeramFw = ForwardPeram(self.charmPerambulatorFile, dtype=self.dtype)
        # lightPeramFw = ForwardPeram(self.lightPerambulatorFile, dtype=self.dtype)
        for iset in range(self.sets):
            charmPeramFw.load(iset)
            # lightPeramFw.load(iset)
            # peramBw = peramFw.toBackwardPeram()
            time3 = mytimer(info=f"GPU calc: {iset}")

            for icalc in range(self.calcLen):
                iElemSrc = self.srcVarIndiceList[icalc]
                iElemSink = self.sinkVarIndiceList[icalc]

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!sum i j for Gamma!
                for i in range(1):
                    for j in range(1):
                        # time2 = mytime(f"do peram {iset}.")
                        self.corrOut[icalc, :, self.tSub*iset:self.tSub*(iset+1), :] += self.ContractionABLocal(charmPeramFw, allElementalList[iElemSrc], allElementalList[iElemSink], iset, g1=i+1, g2=j+1).get()
            # time2.end()
            # del time2
            time3.end()
            del time3

        for icalc in range(self.calcLen):
            iElemSrc = self.srcVarIndiceList[icalc]
            iElemSink = self.sinkVarIndiceList[icalc]
            savepath=self.corrSaveDir+"/"+self.operatorVarList[iElemSrc]["name"]+"_"+self.operatorVarList[iElemSink]["name"]
            if not os.path.exists(savepath):
                os.system(f"mkdir -p {savepath}")
            print(f"Save corr: ", f"{savepath}/coor.{self.cfgKeyword}.npy")
            np.save(f"{savepath}/coor.{self.cfgKeyword}.npy",self.corrOut[icalc, :, :, :].sum(1))

        time_exe.end()
        del time_exe

    def ContractionABLocal(self, charmPeramFw: ForwardPeram, elementalSrc: ElementalsAB, elementalSink: ElementalsAB, iset: int, g1: int=None, g2: int=None):
        if not isinstance(charmPeramFw, ForwardPeram):
            raise TypeError("peram should be (class) ForwardPeram!")
        if not isinstance(elementalSrc, ElementalsAB):
            raise TypeError("elemSrc should be (class) Elementals!")
        if not isinstance(elementalSink, ElementalsAB):
            raise TypeError("elemSink should be (class) Elementals!")

        corr = cupy.zeros(
            shape=(self.saveCorrDim, self.tSub, self.TimeSlice), dtype=self.dtype)

        # for i in range(elementalSrc.termNum)
        # print(f"DEBUG: {elementalSrc.elementalName} = {elementalSrc.coeff} * {elementalSrc.Gamma} , {elementalSink.elementalName} = {elementalSink.coeff} * {elementalSink.Gamma}")
        gammaSrcA = cupy.array([Gamma.G4 @ (elementalSrc.coeffA[i] * gammaSum3Dict[elementalSrc.GammaA[i]]).conj().T @ Gamma.G4 @ Gamma.G5  for i in range(elementalSrc.termNumA)])
        gammaSrcB = cupy.array([Gamma.G4 @ (elementalSrc.coeffB[i] * gammaSum3Dict[elementalSrc.GammaB[i]]).conj().T @ Gamma.G4 @ Gamma.G5  for i in range(elementalSrc.termNumB)])
        gammaSinkA = cupy.array([Gamma.G5 @ (elementalSink.coeffA[i] * gammaSum3Dict[elementalSink.GammaA[i]]) for i in range(elementalSink.termNumA)])
        gammaSinkB = cupy.array([Gamma.G5 @ (elementalSink.coeffB[i] * gammaSum3Dict[elementalSink.GammaB[i]]) for i in range(elementalSink.termNumB)])
        
        charmPeramBw = BackwardPeram("", dtype=self.dtype)
        charmPeramBw.data = charmPeramFw.data.conj()
        # lightPeramBw = BackwardPeram("", dtype=self.dtype)
        # lightPeramBw.data = lightPeramFw.data.conj()



        # A = 0,  B = 1
        # add for multi-term implement!
        # shape=(self.calcMomNum,self.termNum, 2, TimeSlice, Nvec, Nvec)
        elementalSrcA = cupy.conj(
            elementalSrc.dataA[:,:, self.tSub*iset:self.tSub*(iset+1),:,:])
        elementalSrcB = cupy.conj(
            elementalSrc.dataB[:,:, self.tSub*iset:self.tSub*(iset+1),:,:])
        elementalSinkA = elementalSink.dataA[:,:,:,:,:]
        elementalSinkB = elementalSink.dataB[:,:,:,:,:]



        # Note: spin indice: abcdefgh
        # Note: elementals->vectors indice: mnklopqr
        # Note: elementals->Gamma term indice: ijxyz
        # Bote: elementals->momentum pair indice: wu
        # MY TENSOR CONTRACTION RULE:
        # 1. bottom-up ordering to read freymann diagram
        # 2. elementalSrc is dagger from elementals.data! conj + reverse indice
        # 3. peramBw.data is dagger from peramFw.data! conj + reverse indice

        for t in range(self.tSub):
            tmpElementalSinkA = cupy.roll(
                        elementalSinkA, -(t+iset*self.tSub), axis=2)
            tmpElementalSinkB = cupy.roll(
                        elementalSinkB, -(t+iset*self.tSub), axis=2)

            for ipsrc in range(elementalSrc.calcMomNum):
                for ipsink in range(elementalSink.calcMomNum):
                                
                                        #' tabmn, bc, tnk,  tcdkl, da, lm ->t'
                    srcAsinkA = cupy.einsum(' tbanm,ibc,itnk,  tcdkl,jda,jml -> t',
                                        charmPeramBw.data[t, :, :, :, :, :],
                                        gammaSinkA,
                                        tmpElementalSinkA[ipsink],

                                        charmPeramFw.data[t, :, :, :, :, :],
                                        gammaSrcA,
                                        elementalSrcA[ipsrc,:,t, :, :],
                                        
                                        optimize=True
                                        )
                                        #' tabmn, bc, tnk,  tcdkl, da, lm ->t'
                    srcBsinkB = cupy.einsum(' tbanm,ibc,itnk,  tcdkl,jda,jml -> t',
                                        charmPeramBw.data[t, :, :, :, :, :],
                                        gammaSinkB,
                                        tmpElementalSinkB[ipsink],

                                        charmPeramFw.data[t, :, :, :, :, :],
                                        gammaSrcB,
                                        elementalSrcB[ipsrc,:,t, :, :],
                                        
                                        optimize=True
                                        )


                    # corr[0, t]
                    tosave0 = cupy.einsum('t,t->t', srcAsinkA, srcBsinkB)

                                        #'tabmn, bc,  tnk,  tcdkl, de,  lo,  tefop, fg,  tpq,  tghqr, ha,  rm->t',
                    tosave1 = cupy.einsum('tbanm,ibc, itnk,  tcdkl,jde, jol,  tfepo,xfg, xtpq,  tghqr,yha, ymr->t',
                                                    charmPeramBw.data[t,
                                                                :, :, :, :, :],
                                                    gammaSinkB,
                                                    tmpElementalSinkB[ipsink],  # SinkB

                                                    charmPeramFw.data[t,
                                                                :, :, :, :, :],
                                                    gammaSrcA,
                                                    # SrcA
                                                    elementalSrcA[ipsrc,:,t, :, :],

                                                    charmPeramBw.data[t,
                                                                :, :, :, :, :],
                                                    gammaSinkA,
                                                    tmpElementalSinkA[ipsink],  # SinkA

                                                    charmPeramFw.data[t,
                                                                :, :, :, :, :],
                                                    gammaSrcB,
                                                    # SrcB
                                                    elementalSrcB[ipsrc,:,t, :, :],
                                                    optimize=True
                                                    )

                    if elementalSrc.mom == elementalSink.mom:
                        corr[2, t] += (-cupy.einsum('t->t', srcAsinkA))
                        corr[3, t] += (-cupy.einsum('t->t', srcBsinkB))
                    # else:
                    #     corr[2, t] = cupy.zeros(128)
                    #     corr[3, t] = cupy.zeros(128)

                    # Add: for difficient A B
                    srcAsinkB = cupy.einsum(' tbanm,ibc, itnk,  tcdkl,jda, jml -> t',
                                        charmPeramBw.data[t, :, :, :, :, :],
                                        gammaSinkB,
                                        tmpElementalSinkB[ipsink],

                                        charmPeramFw.data[t, :, :, :, :, :],
                                        gammaSrcA,
                                        elementalSrcA[ipsrc,:, t, :, :],

                                        optimize=True
                                        )
                                        #' tabmn, bc, tnk,  tcdkl, da, lm ->t'
                    srcBsinkA = cupy.einsum(' tbanm,ibc, itnk,  tcdkl,jda, jml -> t',
                                        charmPeramBw.data[t, :, :, :, :, :],
                                        gammaSinkA,
                                        tmpElementalSinkA[ipsink],

                                        charmPeramFw.data[t, :, :, :, :, :],
                                        gammaSrcB,
                                        elementalSrcB[ipsrc,:,t, :, :],
                                        
                                        optimize=True
                                        )


                    tosave4 = cupy.einsum('t,t->t', srcAsinkB, srcBsinkA)

                                            #'tabmn, bc,  tnk,  tcdkl, de,  lo,  tefop, fg,  tpq,  tghqr, ha,  rm->t',
                    tosave5 = cupy.einsum('tbanm,ibc, itnk,  tcdkl,jde, jol,  tfepo,xfg, xtpq,  tghqr,yha, ymr->t',
                                                charmPeramBw.data[t,
                                                            :, :, :, :, :],
                                                gammaSinkA,
                                                tmpElementalSinkA[ipsink],  # SinkA

                                                charmPeramFw.data[t,
                                                            :, :, :, :, :],
                                                gammaSrcA,
                                                # SrcA
                                                elementalSrcA[ipsrc,:,t, :, :],

                                                charmPeramBw.data[t,
                                                            :, :, :, :, :],
                                                gammaSinkB,
                                                tmpElementalSinkB[ipsink],  # SinkB

                                                charmPeramFw.data[t,
                                                            :, :, :, :, :],
                                                gammaSrcB,
                                                # SrcB
                                                elementalSrcB[ipsrc,:,t, :, :],
                                                optimize=True
                                                )
                    
                    if elementalSrc.mom == elementalSink.mom:
                        corr[6, t] += (-cupy.einsum('t->t', srcAsinkB))
                        corr[7, t] += (-cupy.einsum('t->t', srcBsinkA))
                    # else:
                    #     corr[6, t] = cupy.zeros(128)
                    #     corr[7, t] = cupy.zeros(128)
                    
                    corr[0, t] += tosave0
                    corr[1, t] += tosave1
                    corr[4, t] += tosave4
                    corr[5, t] += tosave5
                    corr[8, t] += tosave0 - tosave1 + tosave4 - tosave5

        return corr