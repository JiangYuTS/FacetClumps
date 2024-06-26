#usr JiangYu
import time
import numpy as np
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.table import Table

from FacetClumps import Detect_FacetClumps

def Change_pix2word(data_header, outcat, ndim):
    """
    :param data_wcs: file header.
    :param outcat: table in pixel coordinate.
    :return:
    outcat_wcs: table in WCS coordinate.
    """
    data_header['CTYPE3'] = 'VELO'
    data_wcs = wcs.WCS(data_header)

    clump_Peak = outcat['Peak']
    clump_Volume = outcat['Volume']
    clump_Angle = outcat['Angle']
    clump_Edge = outcat['Edge']
    convert_to_WCS = False
    if 'CDELT1' in data_header and 'CDELT2' in data_header:
        size1, size2 = np.array([outcat['Size1'] * np.abs(data_header['CDELT1']) * 3600,
                                 outcat['Size2'] * np.abs(data_header['CDELT2']) * 3600])
        convert_to_WCS = True
    else:
        size1, size2 = np.array([outcat['Size1'], outcat['Size2']])
        print('The size has not converted to WCS!')
        print('You need to transform the WCS table through the Pix table by yourself.')
        convert_to_WCS = False
    if ndim == 2:
        # 2D result
        clump_Sum = outcat['Sum']
        if data_wcs.world_n_dim == 2:
            peak1, peak2 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1)
            cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
        elif data_wcs.world_n_dim == 3:
            peak1, peak2, temp_value = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1, 1)
            cen1, cen2, temp_value = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1, 1)
        else:
            print('The data_wcs.world_n_dim is unexpected!')
        clump_Peaks = np.column_stack([peak1, peak2])
        clump_Cen = np.column_stack([cen1, cen2])
        clump_Size = np.column_stack([size1, size2])
    elif ndim == 3:
        # 3D result
        if data_wcs.world_n_dim == 3:
            peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
            cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
        elif data_wcs.world_n_dim == 4:
            peak1, peak2, peak3, temp_p = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1,
                                                                 1)
            cen1, cen2, cen3, temp_c = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1, 1)
        else:
            print('The data_wcs.world_n_dim is unexpected!')
        if 'CDELT3' in data_header and 'CUNIT3' in data_header:
            if data_header['CUNIT3'] == 'm/s' or data_header['CUNIT3'] == 'm s-1':
                clump_Peaks = np.column_stack([peak1, peak2, peak3 / 1000])
                clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
                clump_Size = np.column_stack([size1, size2, outcat['Size3'] * data_header['CDELT3'] / 1000])
                clump_Sum = outcat['Sum'] * data_header['CDELT3'] / 1000
                convert_to_WCS = True
            elif data_header['CUNIT3'] == 'km/s' or data_header['CUNIT3'] == 'km s-1':
                clump_Peaks = np.column_stack([peak1, peak2, peak3])
                clump_Cen = np.column_stack([cen1, cen2, cen3])
                clump_Size = np.column_stack([size1, size2, outcat['Size3'] * data_header['CDELT3']])
                clump_Sum = outcat['Sum'] * data_header['CDELT3']
                convert_to_WCS = True
            else:
                print('Please cheek the unit of the velocity channels (str: km/s, m/s, m s-1, km s-1, or else).')
                print('You need to transform the WCS table through the Pix table by yourself.')
                clump_Peaks = np.column_stack([peak1, peak2, peak3])
                clump_Cen = np.column_stack([cen1, cen2, cen3])
                clump_Size = np.column_stack([size1, size2, outcat['Size3']])
                clump_Sum = outcat['Sum']
                convert_to_WCS = False
        else:
            print('Please cheek the key world of the velocity channel and unit (CDELT3, CUNIT3, or else).')
            print('You need to transform the WCS table through the Pix table by yourself.')
            clump_Peaks = np.column_stack([peak1, peak2, peak3])
            clump_Cen = np.column_stack([cen1, cen2, cen3])
            clump_Size = np.column_stack([size1, size2, outcat['Size3']])
            clump_Sum = outcat['Sum']
            convert_to_WCS = False
    id_clumps = np.arange(1, len(clump_Peak) + 1, 1)
    outcat_wcs = np.column_stack((id_clumps, clump_Peaks, clump_Cen, clump_Size, clump_Peak,
                                  clump_Sum, clump_Volume, clump_Angle, clump_Edge))
    return outcat_wcs,convert_to_WCS

def Table_Interface(did_table, data_header, ndim):
    """
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume', 'Angle', 'Edge'] -->3d
    ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume', 'Angle', 'Edge']-->2d
    """
    Peak = did_table['peak_value']
    Sum = did_table['clump_sum']
    Volume = did_table['clump_volume']
    Angle = did_table['clump_angle']
    Edge = did_table['clump_edge']
    if ndim == 2:
        Peak1 = list(np.array(did_table['peak_location'])[:, 1] + 1)
        Peak2 = list(np.array(did_table['peak_location'])[:, 0] + 1)
        Cen1 = list(np.array(did_table['clump_center'])[:, 1] + 1)
        Cen2 = list(np.array(did_table['clump_center'])[:, 0] + 1)
        Size1 = list(np.array(did_table['clump_size'])[:, 1])
        Size2 = list(np.array(did_table['clump_size'])[:, 0])
        index_id = np.arange(1, len(Peak1) + 1, 1)
        d_outcat = np.hstack([[index_id, Peak1, Peak2, Cen1, Cen2, Size1, Size2, Peak, Sum, Volume, Angle, Edge]]).T
        columns = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume', 'Angle', 'Edge']
        units = [None, 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', None, None, 'pix', 'deg', None]
        dtype = ['int', 'int', 'int', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'int', 'int8',
                 'int8']
        if 'BUNIT' in data_header.keys():
            units_wcs = [None, 'deg', 'deg', 'deg', 'deg', 'pix', 'pix', data_header['BUNIT'], data_header['BUNIT'], 'pix',
                     'deg', None]
        else:
            units_wcs = [None, 'deg', 'deg', 'deg', 'deg', 'pix', 'pix', 'K', 'K','pix',
                         'deg', None]
        dtype_wcs = ['int', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32',
                     'int', 'int8', 'int8']
        td_outcat = Table(d_outcat, names=columns, dtype=dtype, units=units)
        td_outcat_wcs = Change_pix2word(data_header, td_outcat, ndim)
        td_outcat_wcs = Table(td_outcat_wcs, names=columns, dtype=dtype_wcs, units=units_wcs)
        for i in range(len(dtype)):
            if dtype[i] == 'float32':
                td_outcat[columns[i]].info.format = '.3f'
            if dtype_wcs[i] == 'float32':
                td_outcat_wcs[columns[i]].info.format = '.3f'
    elif ndim == 3:
        Peak1 = list(np.array(did_table['peak_location'])[:, 2] + 1)
        Peak2 = list(np.array(did_table['peak_location'])[:, 1] + 1)
        Peak3 = list(np.array(did_table['peak_location'])[:, 0] + 1)
        Cen1 = list(np.array(did_table['clump_center'])[:, 2] + 1)
        Cen2 = list(np.array(did_table['clump_center'])[:, 1] + 1)
        Cen3 = list(np.array(did_table['clump_center'])[:, 0] + 1)
        Size1 = list(np.array(did_table['clump_size'])[:, 2])
        Size2 = list(np.array(did_table['clump_size'])[:, 1])
        Size3 = list(np.array(did_table['clump_size'])[:, 0])
        index_id = np.arange(1, len(Peak1) + 1, 1)
        d_outcat = np.hstack([[index_id, Peak1, Peak2, Peak3, Cen1, Cen2, Cen3, Size1, Size2, Size3,
                               Peak, Sum, Volume, Angle, Edge]]).T
        columns = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3',
                   'Peak', 'Sum', 'Volume', 'Angle', 'Edge']
        units = [None, 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', 'pix', None, None, 'pix', 'deg', None]
        dtype = ['int', 'int', 'int', 'int', 'float32', 'float32', 'float32', 'float32', 'float32',
                 'float32', 'float32', 'float32', 'int', 'int8', 'int8']
        if 'BUNIT' in data_header.keys():
            units_wcs = [None, 'deg', 'deg', 'km/s', 'deg', 'deg', 'km/s', 'arcmin', 'arcmin', 'km/s', \
                     data_header['BUNIT'][0], data_header['BUNIT'][0] + ' km/s', 'pix', 'deg', None]
        else:
            units_wcs = [None, 'deg', 'deg', 'km/s', 'deg', 'deg', 'km/s', 'arcmin', 'arcmin', 'km/s', \
                     'K', 'K' + ' km/s', 'pix', 'deg', None]
        dtype_wcs = ['int', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32',
                     'float32', 'float32', 'float32', 'float32', 'float32', 'int', 'int8', 'int8']
        td_outcat = Table(d_outcat, names=columns, dtype=dtype, units=units)
        td_outcat_wcs,convert_to_WCS = Change_pix2word(data_header, td_outcat, ndim)
        td_outcat_wcs = Table(td_outcat_wcs, names=columns, dtype=dtype_wcs, units=units_wcs)
        for i in range(len(dtype)):
            if dtype[i] == 'float32':
                td_outcat[columns[i]].info.format = '.3f'
            if dtype_wcs[i] == 'float32':
                td_outcat_wcs[columns[i]].info.format = '.3f'
    return td_outcat, td_outcat_wcs,convert_to_WCS


def Detect(file_name, parameters, mask_name, outcat_name, outcat_wcs_name):
    start_1 = time.time()
    start_2 = time.ctime()
    RMS = parameters[0]
    Threshold = parameters[1]
    SWindow = parameters[2]
    KBins = parameters[3]
    FwhmBeam = parameters[4]
    VeloRes = parameters[5]
    SRecursionLBV = parameters[6]
    # WeightFactor = [FwhmBeam, VeloRes]
    did_table, td_outcat, td_outcat_wcs = [], [], []
    origin_data = fits.getdata(file_name)
    origin_data = np.squeeze(origin_data)
    origin_data[np.isnan(origin_data)] = -999
    ndim = origin_data.ndim
    if ndim == 2:
        SRecursionLB = SRecursionLBV[0]
        did_table = Detect_FacetClumps.Detect_FacetClumps_2D(RMS, Threshold, SWindow, KBins, SRecursionLB, origin_data)
    elif ndim == 3:
        did_table = Detect_FacetClumps.Detect_FacetClumps_3D(RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes,
                                                             SRecursionLBV, origin_data)
    else:
        raise Exception('Please check the dimensionality of the data!')
    if len(did_table['peak_value']) != 0:
        # np.savez(outcat_name[:-4] + '_FacetClumps_npz', did_FacetClumps=did_table)
        regions_data = did_table['regions_data']
        fits.writeto(mask_name, regions_data, overwrite=True)
        data_header = fits.getheader(file_name)
        td_outcat, td_outcat_wcs, convert_to_WCS = Table_Interface(did_table, data_header, ndim)
        td_outcat.write(outcat_name, overwrite=True)
        td_outcat_wcs.write(outcat_wcs_name, overwrite=True)
        print('Number:', len(did_table['peak_value']))
    else:
        print('No clumps!')
        convert_to_WCS = False
    end_1 = time.time()
    end_2 = time.ctime()
    delta_time = np.around(end_1 - start_1, 2)
    SRecursionLBV = str(SRecursionLBV)
    par_time_record = np.hstack([[RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, \
                                  SRecursionLBV, start_2, end_2, delta_time, convert_to_WCS]])
    par_time_record = Table(par_time_record, names=['RMS', 'Threshold', 'SWindow', 'KBins', \
                    'FwhmBeam', 'VeloRes', 'SRecursionLBV', 'Start', 'End', 'DTime', 'CToWCS'])
    par_time_record.write(outcat_name[:-4] + '_FacetClumps_record.csv', overwrite=True)
    print('Time:', delta_time)
    did_tables = {}
    did_tables['outcat_table'] = td_outcat
    did_tables['outcat_wcs_table'] = td_outcat_wcs
    did_tables['mask'] = did_table['regions_data']
    return did_tables





