#usr JiangYu
import numpy as np
import astropy.io.fits as fits

# from FacetClumps.Detect_Files import Detect as DF_FacetClumps
import FacetClumps

def Convert_SN_Data(file_name,rms_file,outcat_name):
    origin_data = fits.getdata(file_name)
    origin_data = np.squeeze(origin_data)
    data_header = fits.getheader(file_name)
    rms_data = fits.getdata(rms_file)
    rms_data = np.squeeze(rms_data)
    file_name_SN = outcat_name[:-4] + '_SN_Data.fits'
    if rms_data.shape[0]!=origin_data.shape[1] or rms_data.shape[1]!=origin_data.shape[2]:
        convert_to_SN = False
        print('The shape of RMS data and original data is not matched.')
    else:
        convert_to_SN = True
        rms_data_expanded = np.repeat(rms_data[np.newaxis, :, :], origin_data.shape[0], axis=0)
        origin_data_SN = origin_data / rms_data_expanded
        data_hdu = fits.PrimaryHDU(origin_data_SN, header=data_header)
        fits.HDUList([data_hdu]).writeto(file_name_SN, overwrite=True)
    return convert_to_SN, file_name_SN

def Detect_SN(file_name,parameters,mask_name,outcat_name,outcat_wcs_name,rms_file,detect_SN_data):
    if detect_SN_data:
        convert_to_SN_flag,file_name_SN = Convert_SN_Data(file_name,rms_file,outcat_name)
        did_tables_FacetClumps = None
        if convert_to_SN_flag:
            parameters[1] = parameters[1] / parameters[0]
            parameters[0] = 1
            did_tables_FacetClumps = FacetClumps.Detect_Files.Detect(file_name_SN,parameters,mask_name,outcat_name,outcat_wcs_name)
            # from FacetClumps.Cal_Tables_From_Mask_Funs import Cal_Tables_From_Mask as DF_FacetClumps_Mask
            if did_tables_FacetClumps != None:
                did_tables_FacetClumps = FacetClumps.Cal_Tables_From_Mask_Funs.Cal_Tables_From_Mask(file_name, mask_name, outcat_name, outcat_wcs_name)
    else:
        did_tables_FacetClumps = FacetClumps.Detect_Files.Detect(file_name,parameters,mask_name,outcat_name,outcat_wcs_name)
    return did_tables_FacetClumps
