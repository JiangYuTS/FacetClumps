#usr JiangYu
from FacetClumps.FacetClumps_2D_Funs import Detect_Facet as Detect_Facet_2D
from FacetClumps.FacetClumps_3D_Funs import Detect_Facet as Detect_Facet_3D

if __name__ == '__main__':
    origin_data = 'origin_data'

    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    KBins = 35  # [10,...,60]

    #2D
    Vmm = 27  # Vmm = 16  # [9,16,27,36]
    did_Facet = Detect_Facet_2D(RMS,Threshold,KBins,Vmm,origin_data)
    #3D
    Vmm = 64  # [27,64,125]
    did_Facet = Detect_Facet_3D(RMS,Threshold,KBins,Vmm,origin_data)