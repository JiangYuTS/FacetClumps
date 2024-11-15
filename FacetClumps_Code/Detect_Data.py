#usr JiangYu
from FacetClumps.FacetClumps_2D_Funs import Detect_FacetClumps as Detect_FacetClumps_2D
from FacetClumps.FacetClumps_3D_Funs import Detect_FacetClumps as Detect_FacetClumps_3D


if __name__ == '__main__':
    origin_data = 'origin_data'

    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    SWindow = 3
    KBins = 35  # [10,...,60]
    FwhmBeam = 2
    VeloRes = 2
    SRecursionLBV = [16, 5]  # [(2+FwhmBeam)**2,3+VeloRes]

    #2D
    SRecursionLB = SRecursionLBV[0]
    did_FacetClumps = Detect_FacetClumps_2D(RMS,Threshold,SWindow,KBins,SRecursionLB,origin_data)
    #3D
    did_FacetClumps = Detect_FacetClumps_3D(RMS,Threshold,SWindow,KBins,FwhmBeam,VeloRes,SRecursionLBV,origin_data)

