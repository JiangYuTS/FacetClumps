#usr JiangYu
import time
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from skimage import measure

import FacetClumps

def Cal_Tables_From_Mask_2D(file_name,mask_name):
    origin_data = fits.getdata(file_name)
    regions_data = fits.getdata(mask_name)
    regions_data = np.array(regions_data,dtype='int')
    regions_list = measure.regionprops(regions_data)
    peak_value = []
    peak_location = []
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_edge = []
    clump_angle = []
    detect_infor_dict = {}
    for index in range(len(regions_list)):
        clump_coords = regions_list[index].coords
    #     clump_coords = (clump_coords[:,0],clump_coords[:,1],clump_coords[:,2])
        clump_coords_x = clump_coords[:,0]
        clump_coords_y = clump_coords[:,1]
        clump_x_min, clump_x_max = clump_coords_x.min(),clump_coords_x.max()
        clump_y_min, clump_y_max = clump_coords_y.min(),clump_coords_y.max()
        clump_item = np.zeros((clump_x_max-clump_x_min+1,clump_y_max-clump_y_min+1))
        clump_item[(clump_coords_x-clump_x_min,clump_coords_y-clump_y_min)]=\
            origin_data[clump_coords_x,clump_coords_y]
        od_mass = origin_data[(clump_coords_x,clump_coords_y)]
#         od_mass = od_mass - od_mass.min()
        mass_array = np.c_[od_mass,od_mass]
        com = np.around((mass_array*clump_coords).sum(0)\
                    /od_mass.sum(),3).tolist()
        size = np.sqrt((mass_array*(np.array(clump_coords)**2)).sum(0)/od_mass.sum()- \
                       ((mass_array*np.array(clump_coords)).sum(0)/od_mass.sum())**2)
        clump_com.append(com)
        clump_size.append(size.tolist())
        com_item = [com[0]-clump_x_min,com[1]-clump_y_min]
        D,V,size_ratio,angle = FacetClumps.FacetClumps_2D_Funs.Get_DV(clump_item,com_item)
        clump_angle.append(angle)
        peak_coord = np.where(clump_item == clump_item.max())
        peak_coord = [(peak_coord[0]+clump_x_min)[0],(peak_coord[1]+clump_y_min)[0]]
        peak_value.append(origin_data[peak_coord[0],peak_coord[1]])
        peak_location.append(peak_coord)
        clump_sum.append(od_mass.sum())
        clump_volume.append(len(clump_coords_x))
        data_size = origin_data.shape
        if clump_x_min == 0 or clump_y_min == 0 or \
            clump_x_max+1 == data_size[0] or clump_y_max+1 == data_size[1]:
            clump_edge.append(1)
        else:
            clump_edge.append(0)
    detect_infor_dict['peak_value'] = np.around(peak_value,3).tolist()
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_center'] = clump_com
    detect_infor_dict['clump_size'] = np.around(clump_size,3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum,3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
    return detect_infor_dict

def Cal_Tables_From_Mask_3D(file_name,mask_name):
    origin_data = fits.getdata(file_name)
    regions_data = fits.getdata(mask_name)
    regions_data = np.array(regions_data,dtype='int')
    regions_list = measure.regionprops(regions_data)
    peak_value = []
    peak_location = []
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_edge = []
    clump_angle = []
    detect_infor_dict = {}
    for index in range(len(regions_list)):
        clump_coords = regions_list[index].coords
    #     clump_coords = (clump_coords[:,0],clump_coords[:,1],clump_coords[:,2])
        clump_coords_x = clump_coords[:,0]
        clump_coords_y = clump_coords[:,1]
        clump_coords_z = clump_coords[:,2]
        clump_x_min, clump_x_max = clump_coords_x.min(),clump_coords_x.max()
        clump_y_min, clump_y_max = clump_coords_y.min(),clump_coords_y.max()
        clump_z_min, clump_z_max = clump_coords_z.min(),clump_coords_z.max()
        clump_item = np.zeros((clump_x_max-clump_x_min+1,clump_y_max-clump_y_min+1,clump_z_max-clump_z_min+1))
        clump_item[(clump_coords_x-clump_x_min,clump_coords_y-clump_y_min,clump_coords_z-clump_z_min)]=\
            origin_data[clump_coords_x,clump_coords_y,clump_coords_z]
        od_mass = origin_data[(clump_coords_x,clump_coords_y,clump_coords_z)]
#         od_mass = od_mass - od_mass.min()
        mass_array = np.c_[od_mass,od_mass,od_mass]
        com = np.around((mass_array*clump_coords).sum(0)\
                    /od_mass.sum(),3).tolist()
        size = np.sqrt((mass_array*(np.array(clump_coords)**2)).sum(0)/od_mass.sum()- \
                       ((mass_array*np.array(clump_coords)).sum(0)/od_mass.sum())**2)
        clump_com.append(com)
        clump_size.append(size.tolist())
        com_item = [com[0]-clump_x_min,com[1]-clump_y_min,com[2]-clump_z_min]
        D,V,size_ratio,angle = FacetClumps.FacetClumps_3D_Funs.Get_DV(clump_item,com_item)
        clump_angle.append(angle)
        peak_coord = np.where(clump_item == clump_item.max())
        peak_coord = [(peak_coord[0]+clump_x_min)[0],(peak_coord[1]+clump_y_min)[0],(peak_coord[2]+clump_z_min)[0]]
        peak_value.append(origin_data[peak_coord[0],peak_coord[1],peak_coord[2]])
        peak_location.append(peak_coord)
        clump_sum.append(od_mass.sum())
        clump_volume.append(len(clump_coords_x))
        data_size = origin_data.shape
        if clump_x_min == 0 or clump_y_min == 0 or clump_z_min == 0 or \
            clump_x_max+1 == data_size[0] or clump_y_max+1 == data_size[1] or clump_z_max+1 == data_size[2]:
            clump_edge.append(1)
        else:
            clump_edge.append(0)
    detect_infor_dict['peak_value'] = np.around(peak_value,3).tolist()
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_center'] = clump_com
    detect_infor_dict['clump_size'] = np.around(clump_size,3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum,3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
    return detect_infor_dict

def Cal_Tables_From_Mask(file_name, mask_name, outcat_name, outcat_wcs_name):
    start_1 = time.time()
    start_2 = time.ctime()
    did_table, td_outcat, td_outcat_wcs = [], [], []
    origin_data = fits.getdata(file_name)
    origin_data = np.squeeze(origin_data)
    # origin_data[np.isnan(origin_data)] = -999
    ndim = origin_data.ndim
    if ndim == 2:
        did_table = Cal_Tables_From_Mask_2D(file_name,mask_name)
    elif ndim == 3:
        did_table = Cal_Tables_From_Mask_3D(file_name,mask_name)
    else:
        raise Exception('Please check the dimensionality of the data!')
    if len(did_table['peak_value']) != 0:
        # np.savez(outcat_name[:-4] + '_FacetClumps_npz', did_FacetClumps=did_table)
        # regions_data = did_table['regions_data']
        # fits.writeto(mask_name, regions_data, overwrite=True)
        data_header = fits.getheader(file_name)
        td_outcat, td_outcat_wcs, convert_to_WCS = FacetClumps.Detect_Files_Funs.Table_Interface(did_table, data_header, ndim)
        td_outcat.write(outcat_name, overwrite=True)
        td_outcat_wcs.write(outcat_wcs_name, overwrite=True)
        print('Number:', len(did_table['peak_value']))
    else:
        print('No clumps!')
        convert_to_WCS = False
    end_1 = time.time()
    end_2 = time.ctime()
    delta_time = np.around(end_1 - start_1, 2)
    par_time_record = np.hstack([[start_2, end_2, delta_time, convert_to_WCS]])
    par_time_record = Table(par_time_record, names=['Start', 'End', 'DTime', 'CToWCS'])
    par_time_record.write(outcat_name[:-4] + '_FacetClumps_Convert_Record.csv', overwrite=True)
    print('Time:', delta_time)
    did_tables = {}
    did_tables['outcat_table'] = td_outcat
    did_tables['outcat_wcs_table'] = td_outcat_wcs
    did_tables['mask'] = did_table['regions_data']
    return did_tables




