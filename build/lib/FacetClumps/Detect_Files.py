#usr JiangYu
from FacetClumps.Detect_Files_Funs import Detect

if __name__ == '__main__':
    # 2D、3D
    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    KBins = 35  # [10,...,70]
    Vmm = 64  # [9,16,25],[27,64,125]

    parameters = [RMS, Threshold, KBins, Vmm]
    file_name = 'file_name'
    mask_name = 'mask.fits'
    outcat_name = 'outcat.txt'
    outcat_wcs_name = 'outcat_wcs.txt'
    did_Facet = Detect(file_name, parameters, mask_name, outcat_name, outcat_wcs_name)
