#usr JiangYu
from FacetClumps.Detect_Files_Funs import Detect
from FacetClumps.Detect_Files_SN_Funs import Detect_SN

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # 2D、3D
    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    SWindow = 3 # [3,5,7]
    KBins = 35  # [10,...,60]
    FwhmBeam = 2
    VeloRes = 2
    SRecursionLBV = [16, 5]  # [(2+FwhmBeam)**2,3+VeloRes]

    parameters = [RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV]
    file_name = 'file_name'
    rms_file = 'rms_file'
    mask_name = 'mask.fits'
    outcat_name = 'outcat.csv'
    outcat_wcs_name = 'outcat_wcs.csv'
    did_FacetClumps = Detect(file_name, parameters, mask_name, outcat_name, outcat_wcs_name)

    Detect_SN(file_name, parameters, mask_name, outcat_name, outcat_wcs_name, rms_file, detect_SN_data=True)