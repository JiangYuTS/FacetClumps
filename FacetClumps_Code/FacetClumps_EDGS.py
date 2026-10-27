import os
import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure
from tqdm import tqdm

import FacetClumps
from FacetClumps.Cal_Tables_From_Mask_Funs import Cal_Tables_From_Mask as DF_FacetClumps_Mask


# ============================================================
# Equivalent Global Detection Scheme based on Sub-Regions (SRs)
# ============================================================


def Save_Fits(data,data_header,file_name):
    hdu_sub = fits.PrimaryHDU()
    hdu_sub.data = data
    hdu_sub.header = data_header
    hdu_sub.writeto(file_name, overwrite=True)
    

def Cal_SRs_List(RMS, Threshold, file_name):
    """
    Generate sub-region (SR) candidates from a 3D FITS data cube.

    This function performs a two-stage region growing procedure:
    1. Initial segmentation using RMS and intensity threshold.
    2. Secondary refinement based on previously detected regions.

    Returns a list of sub-regions and the original data cube.
    """
    origin_data = fits.getdata(file_name)

    # Initial coarse segmentation (seed regions)
    regions_1, regions_array_1 = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
        origin_data, RMS, Threshold, np.array([0])
    )

    # Refined segmentation using initial region constraints
    srs_list, srs_array = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
        origin_data, RMS, Threshold, regions_array_1
    )

    return srs_list, origin_data


def Cal_SRs_List_SN(RMS, Threshold, file_name, file_rms_name):
    """
    Generate sub-regions (SRs) using a signal-to-noise normalized cube.

    The input data cube is normalized by a 2D RMS map (broadcast along velocity axis)
    to improve robustness against spatially varying noise.

    Returns:
    - SR list
    - S/N normalized data cube
    """
    origin_data = fits.getdata(file_name)
    rms_data = fits.getdata(file_rms_name)

    # Ensure RMS is 2D and expand to match 3D cube
    rms_data = np.squeeze(rms_data)
    rms_data_expanded = np.repeat(rms_data[np.newaxis, :, :], origin_data.shape[0], axis=0)

    # Convert to signal-to-noise cube
    origin_data_SN = origin_data / rms_data_expanded

    # Two-stage SR detection in S/N space
    regions_1, regions_array_1 = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
        origin_data_SN, RMS, Threshold, np.array([0])
    )

    srs_list, srs_array = FacetClumps.FacetClumps_3D_Funs.Get_Regions_FacetClumps(
        origin_data_SN, RMS, Threshold, regions_array_1
    )

    return srs_list, origin_data_SN


def Detect_By_SR(srs_list, file_name, origin_data_T, parameters, save_folder):
    """
    Perform full clump detection inside each sub-region (SR).

    This is the main processing pipeline:
    - Iterate over all SRs
    - Extract local 3D sub-cubes with padding
    - Run FacetClumps detection on each SR
    - Merge detected clumps into a global mask
    - Save intermediate timing and final outputs

    Parameters
    ----------
    srs_list : list
        List of sub-regions to process.
    file_name : str
        Original FITS file path.
    origin_data_T : ndarray
        3D data cube used for detection.
    parameters : list
        Detection parameters [RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV].
    save_folder : str
        Output directory.

    Returns
    -------
    did_tables_FacetClumps : table output
        Final clump catalog generated from mask.
    clump_mask_array : ndarray
        Global labeled clump mask cube.
    numbers : list
        Number of detected clumps per SR.
    """

    start_1 = time.time()

    # -----------------------------
    # Unpack detection parameters
    # -----------------------------
    RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV = parameters

    numbers = []
    delta_time_record = []

    # Global clump mask (3D label map)
    clump_mask_array = np.zeros_like(origin_data_T, dtype='uint32')
    regions_array_shape = origin_data_T.shape

    # ============================================================
    # Loop over all sub-regions (SR-based detection strategy)
    # ============================================================
    for index in tqdm(range(len(srs_list))):

        print('---')
        print('Index:', index)

        start_1_i = time.time()

        # Extract voxel coordinates of current SR
        coords_item = srs_list[index].coords

        # Define padded bounding box around SR
        l_max = np.min([coords_item[:, 2].max() + 3, regions_array_shape[2]])
        l_min = np.max([0, coords_item[:, 2].min() - 3])
        b_max = np.min([coords_item[:, 1].max() + 3, regions_array_shape[1]])
        b_min = np.max([0, coords_item[:, 1].min() - 3])
        v_max = np.min([coords_item[:, 0].max() + 3, regions_array_shape[0]])
        v_min = np.max([0, coords_item[:, 0].min() - 3])

        # Extract local 3D sub-cube for this SR
        signal_region_data_i = np.zeros([v_max - v_min, b_max - b_min, l_max - l_min])

        signal_region_data_i[
            coords_item[:, 0] - v_min,
            coords_item[:, 1] - b_min,
            coords_item[:, 2] - l_min
        ] = origin_data_T[
            coords_item[:, 0],
            coords_item[:, 1],
            coords_item[:, 2]
        ]

        # Run FacetClumps detection inside the SR
        detect_infor_dict = FacetClumps.FacetClumps_3D_Funs.Detect_FacetClumps(
            RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes,
            SRecursionLBV, signal_region_data_i
        )

        # If valid peaks are detected in this SR
        if len(detect_infor_dict['peak_value']) != 0:

            number = len(detect_infor_dict['peak_value'])
            regions_data = detect_infor_dict['regions_data']

            # Convert detected regions to connected components
            regions_list = measure.regionprops(regions_data)

            # Merge local labels into global mask
            if len(numbers) == 0:
                label_offset = 0
            else:
                label_offset = np.sum(numbers)

            for i in range(len(regions_list)):
                coords = regions_list[i].coords
                clump_mask_array[
                    coords[:, 0] + v_min,
                    coords[:, 1] + b_min,
                    coords[:, 2] + l_min
                ] = i + 1 + label_offset

            numbers.append(number)

        # Record runtime per SR
        end_1_i = time.time()
        delta_time_record.append(np.around(end_1_i - start_1_i, 2))

    # ============================================================
    # Save runtime statistics
    # ============================================================
    delta_time_record_T = Table(np.vstack(delta_time_record), names=['Time of SubRegion'])
    delta_time_record_T.write(save_folder + '/FacetClumps_SRegion_Time_Record.csv', overwrite=True)

    # ============================================================
    # Save final outputs (mask + catalogs)
    # ============================================================
    mask_name = save_folder + '/mask.fits'
    outcat_name = save_folder + '/outcat.csv'
    outcat_wcs_name = save_folder + '/outcat_wcs.csv'

    data_header = fits.getheader(file_name)
    Save_Fits(clump_mask_array, data_header, mask_name)

    # Convert mask into catalog tables
    did_tables_FacetClumps = DF_FacetClumps_Mask(
        file_name, mask_name, outcat_name, outcat_wcs_name
    )

    end_1 = time.time()
    print('Total Time:', np.around(end_1 - start_1, 2))

    return did_tables_FacetClumps, clump_mask_array, numbers


# ============================================================
# Example execution
# ============================================================
if __name__ == '__main__':

    RMS = 1
    Threshold = 5 * RMS
    SWindow = 3
    KBins = 35
    FwhmBeam = 2
    VeloRes = 2
    SRecursionLBV = [16, 5]

    parameters = [RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV]

    file_name = '../../../Data/M16_test.fits'
    file_rms_name = '../../../Data/M16_test_rms.fits'
    save_folder = 'Detect_Data_Cube_By_SRegion'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # ============================================================
    # Way 1: detection in S/N normalized cube
    # ============================================================
    srs_list, origin_data_T = Cal_SRs_List_SN(
        RMS, Threshold, file_name, file_rms_name
    )

    did_tables_FacetClumps, clump_mask_array, numbers = Detect_By_SR(
        srs_list, file_name, origin_data_T, parameters, save_folder
    )
    
    # # ============================================================
    # # Way 2: Classical detection (non-S/N normalized cube)
    # # ============================================================
    
    # # Read FITS header (used for optional metadata-based RMS extraction)
    # data_header = fits.getheader(file_name)
    
    # # Manually define RMS (can alternatively be read from header)
    # RMS = 0.22  # RMS = data_header['RMS']
    
    # # Detection threshold based on RMS
    # Threshold = 5 * RMS
    
    # # Pack all detection parameters into a single list
    # parameters = [RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV]
    
    # # Generate sub-regions (SRs) directly from raw intensity cube
    # srs_list, origin_data_T = Cal_SRs_List(RMS, Threshold, file_name)
    
    # # Run SR-based clump detection pipeline
    # did_tables_FacetClumps, clump_mask_array, numbers = Detect_By_SR(
    #     srs_list, file_name, origin_data_T, parameters, save_folder
    # )
    
    
    # # ============================================================
    # # Way 3: Detection using precomputed SR + S/N cube input
    # # ============================================================
    
    # # Precomputed SR list file (previous segmentation result)
    # file_name_srs_array = save_folder + "/MWISP_SRegion_Array_All.fits"
    
    # # Precomputed signal-to-noise (S/N) data cube
    # file_name_SN = save_folder.split('/')[0] + "/Detect_Data_Cube/outcat_SN_Data.fits"
    
    # # Load SRs and corresponding S/N cube from saved files
    # srs_list, origin_data_T = Read_SRs_File(file_name_srs_array, file_name_SN)
    
    # # Run SR-based detection using precomputed inputs
    # did_tables_FacetClumps, clump_mask_array, numbers = Detect_By_SR(
    #     srs_list, file_name, origin_data_T, parameters, save_folder
    # )