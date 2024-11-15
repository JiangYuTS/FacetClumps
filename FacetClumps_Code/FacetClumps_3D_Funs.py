# usr JiangYu
import numpy as np
from skimage import filters, measure, morphology
from scipy import signal
from scipy.stats import multivariate_normal
from tqdm import tqdm


def Get_LBV_Table(coords):
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    v_delta = x_max - x_min + 1
    box_data = np.zeros([y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[:, 1] - y_min + 1, coords[:, 2] - z_min + 1] = 1
    box_label = measure.label(box_data)
    box_region = measure.regionprops(box_label)
    lb_area = box_region[0].area
    coords_range = [x_min, x_max, y_min, y_max, z_min, z_max]
    return coords_range, lb_area, v_delta


def Get_Regions_FacetClumps(origin_data, RMS=0.1, threshold='otsu', temp_array=None):
    kopen_radius = 1
    if threshold == 'mean':
        threshold = origin_data.mean()
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(origin_data)
    else:
        threshold = threshold
    open_data = morphology.opening(origin_data > threshold, morphology.ball(kopen_radius))
    if temp_array.ndim != 3:
        dilation_data = morphology.dilation(open_data, morphology.ball(kopen_radius))
        dilation_data = dilation_data * (origin_data > threshold)
        dilation_label = measure.label(dilation_data, connectivity=1)
    elif temp_array.ndim == 3:
        dilation_data = morphology.dilation(temp_array > 0, morphology.ball(kopen_radius))
        dilation_data = dilation_data * (origin_data > RMS)
        xor_data = np.logical_xor(temp_array > 0, dilation_data)
        and_data = np.logical_and(origin_data < threshold, xor_data)
        dilation_data = np.logical_or(temp_array, and_data)
        # dilation_data_1 = ndimage.binary_fill_holes(dilation_data_1)
        dilation_label = measure.label(dilation_data, connectivity=3)
    regions = measure.regionprops(dilation_label)
    regions_array = np.array(dilation_label,dtype=np.int32)
    return regions, regions_array


def Convolve(origin_data, SWindow):
    s = np.int64(SWindow / 2)
    L = 0
    xres, yres, zres = np.mgrid[0:SWindow:1, 0:SWindow:1, 0:SWindow:1]
    xyz = np.column_stack([xres.flat, yres.flat, zres.flat])
    sigma = np.array([s, s, s])
    covariance = np.diag(sigma ** 2)
    center = [s, s, s]
    prob_density = multivariate_normal.pdf(xyz, mean=center, cov=covariance)
    prob_density = prob_density.reshape((SWindow, SWindow, SWindow))
    #     w = np.ones((SWindow,SWindow,SWindow))
    w = prob_density
    z = np.linspace(-s, s, SWindow)
    x = np.expand_dims(z, axis=1)
    y = np.expand_dims(z, axis=1)
    x0 = np.repeat(x, SWindow, axis=1)
    xt = np.repeat(x0, SWindow, axis=0).reshape(SWindow, SWindow, SWindow)
    xt = w * xt
    yt = w * y
    zt = w * z
    k1 = (5 * s ** 2 - (xt ** 2 + yt ** 2 + zt ** 2)) / (2 * np.sqrt(2 * np.pi) * s ** 3)
    k2 = xt * (7 * s ** 2 - (xt ** 2 + yt ** 2 + zt ** 2)) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k3 = yt * (7 * s ** 2 - (xt ** 2 + yt ** 2 + zt ** 2)) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k4 = zt * (7 * s ** 2 - (xt ** 2 + yt ** 2 + zt ** 2)) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k5 = -(s ** 2 - xt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k6 = -(s ** 2 - yt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k7 = -(s ** 2 - zt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 5)
    k8 = xt * yt / (np.sqrt(2 * np.pi) * s ** 5)
    k9 = xt * zt / (np.sqrt(2 * np.pi) * s ** 5)
    k10 = yt * zt / (np.sqrt(2 * np.pi) * s ** 5)
    k11 = xt * yt * zt / (np.sqrt(2 * np.pi) * s ** 7)
    k12 = -xt * (s ** 2 - yt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k13 = -xt * (s ** 2 - zt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k14 = -yt * (s ** 2 - xt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k15 = -yt * (s ** 2 - zt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k16 = -zt * (s ** 2 - xt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k17 = -zt * (s ** 2 - yt ** 2) / (2 * np.sqrt(2 * np.pi) * s ** 7)
    k18 = -xt * (3 * s ** 2 - xt ** 2) / (6 * np.sqrt(2 * np.pi) * s ** 7)
    k19 = -yt * (3 * s ** 2 - yt ** 2) / (6 * np.sqrt(2 * np.pi) * s ** 7)
    k20 = -zt * (3 * s ** 2 - zt ** 2) / (6 * np.sqrt(2 * np.pi) * s ** 7)
    gx = np.array(k2 + 1 / 3 * L ** 2 * k12 + 1 / 3 * L ** 2 * k13 + 1 / 3 * L ** 2 * k18, dtype=np.float32)
    gy = np.array(k3 + 1 / 3 * L ** 2 * k14 + 1 / 3 * L ** 2 * k15 + 1 / 3 * L ** 2 * k19, dtype=np.float32)
    gz = np.array(k4 + 1 / 3 * L ** 2 * k16 + 1 / 3 * L ** 2 * k17 + 1 / 3 * L ** 2 * k20, dtype=np.float32)
    gxx = np.array(2 * k5, dtype=np.float32)
    gyy = np.array(2 * k6, dtype=np.float32)
    gzz = np.array(2 * k7, dtype=np.float32)
    gxy = np.array(k8, dtype=np.float32)
    gxz = np.array(k9, dtype=np.float32)
    gyz = np.array(k10, dtype=np.float32)
    origin_data_T = np.array(origin_data, dtype=np.float32)
    conv_gx = signal.convolve(origin_data_T, gx, mode='same', method='auto')
    conv_gy = signal.convolve(origin_data_T, gy, mode='same', method='auto')
    conv_gz = signal.convolve(origin_data_T, gz, mode='same', method='auto')
    conv_gxx = signal.convolve(origin_data_T, gxx, mode='same', method='auto')
    conv_gyy = signal.convolve(origin_data_T, gyy, mode='same', method='auto')
    conv_gzz = signal.convolve(origin_data_T, gzz, mode='same', method='auto')
    conv_gxy = signal.convolve(origin_data_T, gxy, mode='same', method='auto')
    conv_gxz = signal.convolve(origin_data_T, gxz, mode='same', method='auto')
    conv_gyz = signal.convolve(origin_data_T, gyz, mode='same', method='auto')
    convs = [conv_gx, conv_gy, conv_gz, conv_gxx, conv_gyy, conv_gzz, conv_gxy, conv_gxz, conv_gyz]
    conv_K = k1
    for k in [k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20]:
        conv_K += k
    conv_K = np.array(conv_K, dtype=np.float32)
    hook_face = signal.convolve(origin_data_T, conv_K, mode='same', method='auto')
    return convs, hook_face


def Calculate_Eig(eigenvalue_arrays, convs, region):
    conv_gxx = convs[3]
    conv_gyy = convs[4]
    conv_gzz = convs[5]
    conv_gxy = convs[6]
    conv_gxz = convs[7]
    conv_gyz = convs[8]
    eigenvalue_0 = eigenvalue_arrays[0]
    eigenvalue_1 = eigenvalue_arrays[1]
    eigenvalue_2 = eigenvalue_arrays[2]
    hessian_matrix = np.array(
        [[2 * conv_gxx[region[:, 0], region[:, 1], region[:, 2]], conv_gxy[region[:, 0], region[:, 1], region[:, 2]], \
          conv_gxz[region[:, 0], region[:, 1], region[:, 2]]], \
         [conv_gxy[region[:, 0], region[:, 1], region[:, 2]], 2 * conv_gyy[region[:, 0], region[:, 1], region[:, 2]], \
          conv_gyz[region[:, 0], region[:, 1], region[:, 2]]], \
         [conv_gxz[region[:, 0], region[:, 1], region[:, 2]], conv_gyz[region[:, 0], region[:, 1], region[:, 2]], \
          2 * conv_gzz[region[:, 0], region[:, 1], region[:, 2]]]])
    reshaped_matrix = np.transpose(hessian_matrix, (2, 0, 1))
    eigenvalues_T = np.linalg.eigvals(reshaped_matrix)
    eigenvalue_0[region[:, 0], region[:, 1], region[:, 2]] = eigenvalues_T[:, 0]
    eigenvalue_1[region[:, 0], region[:, 1], region[:, 2]] = eigenvalues_T[:, 1]
    eigenvalue_2[region[:, 0], region[:, 1], region[:, 2]] = eigenvalues_T[:, 2]
    eigenvalue_arrays = [eigenvalue_0, eigenvalue_1, eigenvalue_2]
    return eigenvalue_arrays


def Get_Lable(convs, region, eigenvalue_arrays, bins, keig_bins):
    times = 2
    conv_gx = convs[0]
    conv_gy = convs[1]
    conv_gz = convs[2]
    coords = (region[:, 0], region[:, 1], region[:, 2])
    number_ex, eigs_x = np.histogram(eigenvalue_arrays[0][coords], bins=bins)
    number_ey, eigs_y = np.histogram(eigenvalue_arrays[1][coords], bins=bins)
    number_ez, eigs_z = np.histogram(eigenvalue_arrays[2][coords], bins=bins)
    eig_x = eigs_x[bins + keig_bins]
    eig_y = eigs_y[bins + keig_bins]
    eig_z = eigs_z[bins + keig_bins]
    x_line = convs[0][region[:, 0], region[:, 1], region[:, 2]]
    y_line = convs[1][region[:, 0], region[:, 1], region[:, 2]]
    z_line = convs[2][region[:, 0], region[:, 1], region[:, 2]]
    gra_x_min = -times * np.std(x_line)
    gra_x_max = times * np.std(x_line)
    gra_y_min = -times * np.std(y_line)
    gra_y_max = times * np.std(y_line)
    gra_z_min = -times * np.std(z_line)
    gra_z_max = times * np.std(z_line)
    logic_e_0 = eigenvalue_arrays[0][region[:, 0], region[:, 1], region[:, 2]] < eig_x
    logic_e_1 = eigenvalue_arrays[1][region[:, 0], region[:, 1], region[:, 2]] < eig_y
    logic_e_2 = eigenvalue_arrays[2][region[:, 0], region[:, 1], region[:, 2]] < eig_z
    logic_gx = np.logical_and(conv_gx[region[:, 0], region[:, 1], region[:, 2]] > gra_x_min,
                              conv_gx[region[:, 0], region[:, 1], region[:, 2]] < gra_x_max)
    logic_gy = np.logical_and(conv_gy[region[:, 0], region[:, 1], region[:, 2]] > gra_y_min,
                              conv_gy[region[:, 0], region[:, 1], region[:, 2]] < gra_y_max)
    logic_gz = np.logical_and(conv_gz[region[:, 0], region[:, 1], region[:, 2]] > gra_z_min,
                              conv_gz[region[:, 0], region[:, 1], region[:, 2]] < gra_z_max)
    logic_e = np.logical_and(logic_e_0, np.logical_and(logic_e_1, logic_e_2))
    logic_g = np.logical_or(logic_gx, np.logical_or(logic_gy, logic_gz))
    logic_e_and_g = np.logical_and(logic_e, logic_g)
    new_region = region[logic_e_and_g]
    return new_region


def Recursion_Lable(convs, region, regions_record, eigenvalue_arrays, lnV, logV, bins, keig_bins, SRecursionLBV,
                    recursion_time, regions_num):
    keig_bins -= 1
    if keig_bins < -lnV:
        keig_bins = -lnV
    x_min = region[:, 0].min()
    x_max = region[:, 0].max()
    y_min = region[:, 1].min()
    y_max = region[:, 1].max()
    z_min = region[:, 2].min()
    z_max = region[:, 2].max()
    valid_lable = np.zeros([x_max - x_min + 2, y_max - y_min + 2, z_max - z_min + 2])
    valid_lable[region[:, 0] - x_min, region[:, 1] - y_min, region[:, 2] - z_min] = 1
    valid_lable_measure = measure.label(valid_lable, connectivity=1)
    valid_regions_l = measure.regionprops(valid_lable_measure)
    if len(valid_regions_l) == 2:
        areas = []
        for region_l in valid_regions_l:
            area = region_l.area
            areas.append(area)
        areas_sort_index = np.argsort(areas)
        valid_regions_l_T = []
        for id_i in areas_sort_index:
            valid_regions_l_T.append(valid_regions_l[id_i])
        # valid_regions_l = valid_regions_l_T
    for region_l in valid_regions_l:
        coords_l = region_l.coords
        region = np.c_[coords_l[:, 0] + x_min, coords_l[:, 1] + y_min, coords_l[:, 2] + z_min]
        coords_range, lb_area, v_delta = Get_LBV_Table(coords_l)
        area = region_l.area
        if lb_area <= SRecursionLBV[0] and v_delta <= SRecursionLBV[1] and recursion_time > 1 and \
                (area >= logV or len(valid_regions_l) == 1):
            regions_record[0].append(region)
        elif lb_area <= SRecursionLBV[0] and v_delta <= SRecursionLBV[1] and recursion_time > 1 and \
                area > np.max([2, logV / 2]) and len(valid_regions_l) == 2:
            # regions_record[1].append([regions_num, region])
            regions_num += 1
        elif lb_area > SRecursionLBV[0] or v_delta > SRecursionLBV[1]:
            recursion_time += 1
            region = Get_Lable(convs, region, eigenvalue_arrays, bins, keig_bins)
            if len(region) != 0:
                regions_record = Recursion_Lable(convs, region, regions_record, eigenvalue_arrays, \
                                                 lnV, logV, bins, keig_bins, SRecursionLBV, recursion_time, regions_num)
    return regions_record


def Merge_Regions_Record(regions_record):
    for i in range(1, len(regions_record[1])):
        if regions_record[1][i][0] == 0:
            regions_record[0].append(regions_record[1][i - 1][1])
    if len(regions_record[1]) > 0:
        regions_record[0].append(regions_record[1][-1][1])
    return regions_record


def Get_COM(origin_data, regions_record):
    center_of_mass = []
    # label_area_min = 0
    for final_region in regions_record:
        # if len(final_region[:, 0]) > label_area_min:
        x_region = final_region[:, 0]
        y_region = final_region[:, 1]
        z_region = final_region[:, 2]
        od_mass = origin_data[x_region, y_region, z_region]
        center_of_mass.append(
            np.around((np.c_[od_mass, od_mass, od_mass] * final_region).sum(0) / od_mass.sum(), 3).tolist())
    center_of_mass = np.array(center_of_mass)
    return center_of_mass


def Get_Total_COM(origin_data, regions, convs, kbins, SRecursionLBV):
    keig_bins = -1
    recursion_time = 0
    regions_record = [[], []]
    eigenvalue_0 = np.zeros_like(origin_data)
    eigenvalue_1 = np.zeros_like(origin_data)
    eigenvalue_2 = np.zeros_like(origin_data)
    for i in tqdm(range(len(regions))):
        region_i = np.array(regions[i].coords, dtype=np.int32)
        lnV = np.int32(np.log(len(region_i)))
        logV = np.min([np.log10(len(region_i)), 7])
        # logV = np.log10(len(region_i))
        bins = kbins * lnV
        eigenvalue_arrays = [eigenvalue_0, eigenvalue_1, eigenvalue_2]
        eigenvalue_arrays = Calculate_Eig(eigenvalue_arrays, convs, region_i)
        regions_num = 0
        regions_record = Recursion_Lable(convs, region_i, regions_record, eigenvalue_arrays, \
                                         lnV, logV, bins, keig_bins, SRecursionLBV, recursion_time, regions_num)
        eigenvalue_0[region_i[:, 0], region_i[:, 1], region_i[:, 2]] = 0
        eigenvalue_1[region_i[:, 0], region_i[:, 1], region_i[:, 2]] = 0
        eigenvalue_2[region_i[:, 0], region_i[:, 1], region_i[:, 2]] = 0
    regions_record = Merge_Regions_Record(regions_record)
    com = Get_COM(origin_data, regions_record[0])
    return com


def GNC_FacetClumps(origin_data, cores_coordinate):
    xres, yres, zres = origin_data.shape
    x_center = cores_coordinate[0] + 1
    y_center = cores_coordinate[1] + 1
    z_center = cores_coordinate[2] + 1
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    z_arange = np.arange(max(0, z_center - 1), min(zres, z_center + 2))
    [x, y, z] = np.meshgrid(x_arange, y_arange, z_arange);
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    gradients = origin_data[xyz[:, 0], xyz[:, 1], xyz[:, 2]] \
                - origin_data[x_center, y_center, z_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xyz[g_step] - 1)
    return gradients, new_center


def Build_RC_Dict(com, regions_array, regions_first):
    k1 = 0
    k2 = 0
    i_record = []
    temp_rc_dict = {}
    rc_dict = {}
    new_regions = []
    temp_regions_array = np.zeros_like(regions_array)
    for i in range(1, len(regions_first) + 1):
        temp_rc_dict[i] = []
    center = np.array(np.around(com, 0), dtype = np.int32)
    for cent in center:
        if regions_array[cent[0], cent[1], cent[2]] != 0:
            temp_rc_dict[regions_array[cent[0], cent[1], cent[2]]].append(com[k1])
            i_record.append(regions_array[cent[0], cent[1], cent[2]])
        k1 += 1
    for i in range(1, len(regions_first) + 1):
        if i in i_record:
            coordinates = regions_first[i - 1].coords
            temp_regions_array[(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])] = 1
            new_regions.append(regions_first[i - 1])
            rc_dict[k2] = temp_rc_dict[i]
            k2 += 1
    return new_regions, temp_regions_array, rc_dict


def Build_MPR_Dict(origin_data, regions):
    k = 1
    reg = -1
    peak_dict = {}
    peak_dict[k] = []
    mountain_dict = {}
    mountain_dict[k] = []
    region_mp_dict = {}
    origin_data = origin_data + np.random.random(origin_data.shape) / 100000
    mountain_array = np.zeros_like(origin_data,dtype=np.int32)
    temp_origin_data = np.zeros(tuple(np.array(origin_data.shape) + 2))
    for i in range(len(regions)):
        region_mp_dict[i] = []
    for region in tqdm(regions):
        reg += 1
        coordinates = region.coords
        temp_origin_data[(coordinates[:, 0] + 1, coordinates[:, 1] + 1, coordinates[:, 2] + 1)] = \
            origin_data[(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])]
        for i in range(coordinates.shape[0]):
            temp_coords = []
            if mountain_array[coordinates[i][0], coordinates[i][1], coordinates[i][2]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0], coordinates[i][1], coordinates[i][2]] = k
                gradients, new_center = GNC_FacetClumps(temp_origin_data, coordinates[i])
                if gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                    temp_coords.append(new_center)
                while gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                    mountain_array[new_center[0], new_center[1], new_center[2]] = k
                    gradients, new_center = GNC_FacetClumps(temp_origin_data, new_center)
                    if gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                        temp_coords.append(new_center)
                mountain_array[np.stack(temp_coords)[:, 0], np.stack(temp_coords)[:, 1], np.stack(temp_coords)[:, 2]] = \
                    mountain_array[new_center[0], new_center[1], new_center[2]]
                mountain_dict[mountain_array[new_center[0], new_center[1], new_center[2]]] += temp_coords
                if gradients.max() <= 0:
                    peak_dict[k].append(new_center)
                    region_mp_dict[reg].append(k)
                    k += 1
                    mountain_dict[k] = []
                    peak_dict[k] = []
    del (mountain_dict[k])
    del (peak_dict[k])
    return mountain_array, mountain_dict, peak_dict, region_mp_dict


def Updata_Peak_Dict(origin_data, mountain_dict, peak_dict):
    for key in mountain_dict.keys():
        coords = np.array(mountain_dict[key])
        od_mass = origin_data[coords[:, 0], coords[:, 1], coords[:, 2]]
        mass_array = np.c_[od_mass, od_mass, od_mass]
        com = np.around((np.c_[mass_array] * coords).sum(0) / od_mass.sum(), 3).tolist()
        peak_dict[key] = [com]
    return peak_dict


def Dists_Array_Weighted(FwhmBeam, VeloRes, matrix_1, matrix_2):
    WeightFactor = np.array([VeloRes, FwhmBeam, FwhmBeam])
    matrix_1 = matrix_1 / WeightFactor
    matrix_2 = matrix_2 / WeightFactor
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    dists = np.sqrt(dist_1 + dist_2 + dist_3)
    return dists


def Connectivity_FacetClumps(core_dict_1, core_dict_2, i_num, j_num):
    i_region = core_dict_1[i_num]
    j_region = core_dict_2[j_num]
    x_min = np.r_[i_region[:, 0], j_region[:, 0]].min()
    x_max = np.r_[i_region[:, 0], j_region[:, 0]].max()
    y_min = np.r_[i_region[:, 1], j_region[:, 1]].min()
    y_max = np.r_[i_region[:, 1], j_region[:, 1]].max()
    z_min = np.r_[i_region[:, 2], j_region[:, 2]].min()
    z_max = np.r_[i_region[:, 2], j_region[:, 2]].max()
    box_data = np.zeros([x_max - x_min + 2, y_max - y_min + 2, z_max - z_min + 2])
    box_data[i_region[:, 0] - x_min, i_region[:, 1] - y_min, i_region[:, 2] - z_min] = 1
    box_data[j_region[:, 0] - x_min, j_region[:, 1] - y_min, j_region[:, 2] - z_min] = 1
    box_label = measure.label(box_data, connectivity=3)
    box_region = measure.regionprops(box_label)
    return len(box_region)


def Delect(FwhmBeam, VeloRes, new_peak_dict, new_core_dict, com_keys_array, com_values_array, core_dict_center,
           key_peak_record, near_k_2, origin_data):
    del_key_peak_ids = []
    key_peak_id = 0
    for key_peak in key_peak_record:
        dist = Dists_Array_Weighted(FwhmBeam, VeloRes, [new_peak_dict[key_peak]], com_values_array)
        dist_index_sort = np.argsort(dist[0])
        pdc_sort = com_keys_array[dist_index_sort]
        i_num = key_peak
        for key_center in pdc_sort[:near_k_2]:
            j_num = key_center
            connectivity_1 = Connectivity_FacetClumps(new_core_dict, core_dict_center, i_num, j_num)
            if connectivity_1 == 1:
                core_dict_center[key_center] = np.r_[core_dict_center[key_center], new_core_dict[key_peak]]
                del new_core_dict[key_peak]
                del new_peak_dict[key_peak]
                del_key_peak_ids.append(key_peak_id)
                break
        key_peak_id += 1
    return del_key_peak_ids


def Update_CP_Dict_FacetClumps(FwhmBeam, VeloRes, new_peak_dict, com_dict, new_core_dict, mountain_array, origin_data):
    com_dict_temp = {}
    core_dict_center = {}
    key_mountain_record = []
    center_mountain = {}
    new_peak_dict_2 = {}
    for key in new_peak_dict.keys():
        new_peak_dict_2[key] = new_peak_dict[key]
    for key_center in com_dict.keys():
        com_coord = com_dict[key_center]
        com_coord = np.array(np.around(com_coord, 0), dtype = np.int32)
        key_mountain = mountain_array[com_coord[0], com_coord[1], com_coord[2]]
        if key_mountain not in key_mountain_record:
            com_dict_temp[key_center] = com_dict[key_center]
            core_dict_center[key_center] = new_core_dict[key_mountain]
            center_mountain[key_mountain] = key_center
            key_mountain_record.append(key_mountain)
            del new_core_dict[key_mountain]
            del new_peak_dict[key_mountain]
        else:
            old_center = com_dict_temp[center_mountain[key_mountain]]
            new_center = com_dict[key_center]
            dist_1 = Dists_Array_Weighted(FwhmBeam, VeloRes, [old_center], [new_peak_dict_2[key_mountain]])[0][0]
            dist_2 = Dists_Array_Weighted(FwhmBeam, VeloRes, [new_center], [new_peak_dict_2[key_mountain]])[0][0]
            if dist_2 < dist_1:
                com_dict_temp[center_mountain[key_mountain]] = com_dict[key_center]
    com_dict = com_dict_temp
    com_keys_array = np.array(list(com_dict.keys())).copy()
    com_values_array = np.array(list(com_dict.values())).copy()
    np_keys_array = np.array(list(new_peak_dict.keys())).copy()
    np_values_array = np.array(list(new_peak_dict.values())).copy()
    near_k_1 = 1
    near_k_2 = 1
    while len(new_core_dict) > 0:
        if len(new_core_dict) > len(com_dict.keys()):
            for key_center in com_dict.keys():
                if len(new_core_dict) > 0:
                    distance = Dists_Array_Weighted(FwhmBeam, VeloRes, [com_dict[key_center]], np_values_array)
                    distance_index_sort = np.argsort(distance[0])
                    npd_sort = np_keys_array[distance_index_sort]
                    key_peak_record = npd_sort[:near_k_2]
                    del_key_peak_ids = Delect(FwhmBeam, VeloRes, new_peak_dict, new_core_dict, com_keys_array,
                                              com_values_array, \
                                              core_dict_center, key_peak_record, near_k_2, origin_data)
                    real_del_ids = distance_index_sort[del_key_peak_ids]
                    real_del_ids = sorted(real_del_ids, reverse=True)
                    for real_del_id in real_del_ids:
                        np_keys_array = np.delete(np_keys_array, real_del_id, axis=0)
                        np_values_array = np.delete(np_values_array, real_del_id, axis=0)
        else:
            key_peak_record = np.array(list(new_peak_dict.keys()))
            del_key_peak_ids = Delect(FwhmBeam, VeloRes, new_peak_dict, new_core_dict, com_keys_array, com_values_array, \
                                      core_dict_center, key_peak_record, near_k_1, origin_data)
        temp_near_k = near_k_2
        near_k_2 = near_k_1
        near_k_1 += temp_near_k
        if near_k_2 > 8:
            near_k_2 = temp_near_k
    return com_dict_temp, core_dict_center


def Get_CP_Dict(FwhmBeam, VeloRes, rc_dict, mountain_array, mountain_dict, peak_dict, region_mp_dict, origin_data):
    core_id = 0
    com_dict_record = {}
    core_dict_record = {}
    for key in tqdm(np.array(list(rc_dict.keys()))):
        new_peak_dict = {}
        peak_dict_center = {}
        new_core_dict = {}
        mountain_keys = region_mp_dict[key]
        com_FacetClumpss = rc_dict[key]
        for mountain_key in mountain_keys:
            new_peak_dict[mountain_key] = peak_dict[mountain_key][0]
            new_core_dict[mountain_key] = np.array(mountain_dict[mountain_key])
        for com_FacetClumps in com_FacetClumpss:
            peak_dict_center[core_id] = com_FacetClumps.tolist()
            core_id += 1
        com_dict_temp, core_dict_center = \
            Update_CP_Dict_FacetClumps(FwhmBeam, VeloRes, new_peak_dict, peak_dict_center, new_core_dict,
                                       mountain_array, origin_data)
        for key_center in core_dict_center.keys():
            com_dict_record[key_center] = com_dict_temp[key_center]
            core_dict_record[key_center] = core_dict_center[key_center]
    return com_dict_record, core_dict_record


def Get_DV(box_data, box_center):
    box_data_sum = box_data.sum(0)
    box_region = np.where(box_data_sum != 0)
    A11 = np.sum((box_region[0] - box_center[1]) ** 2 * \
                 box_data_sum[box_region])
    A12 = -np.sum((box_region[0] - box_center[1]) * \
                  (box_region[1] - box_center[2]) * \
                  box_data_sum[box_region])
    A21 = A12
    A22 = np.sum((box_region[1] - box_center[2]) ** 2 * \
                 box_data_sum[box_region])
    A = np.array([[A11, A12], [A21, A22]]) / len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1, 0]]
        V = V[[1, 0]]
    if V[1][0] < 0 and V[0][0] > 0 and V[1][1] > 0:
        V = -V
    size_ratio = np.sqrt(D[0] / D[1])
    angle = np.around(np.arccos(V[0][0]) * 180 / np.pi - 90, 3)
    return D, V, size_ratio, angle


def DID_FacetClumps(SRecursionLBV, center_dict, core_dict, origin_data):
    peak_value = []
    peak_location = []
    clump_center = []
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_edge = []
    clump_angle = []
    detect_infor_dict = {}
    k = 0
    regions_data = np.zeros_like(origin_data,dtype=np.int32)
    data_size = origin_data.shape
    for key in center_dict.keys():
        core_x = np.array(core_dict[key])[:, 0]
        core_y = np.array(core_dict[key])[:, 1]
        core_z = np.array(core_dict[key])[:, 2]
        coords_range, lb_area, v_delta = Get_LBV_Table(np.array(core_dict[key]))
        core_x_min, core_x_max = coords_range[0], coords_range[1]
        core_y_min, core_y_max = coords_range[2], coords_range[3]
        core_z_min, core_z_max = coords_range[4], coords_range[5]
        if lb_area > SRecursionLBV[0] and v_delta > SRecursionLBV[1]:
            k += 1
            temp_core = np.zeros(
                (core_x_max - core_x_min + 1, core_y_max - core_y_min + 1, core_z_max - core_z_min + 1))
            temp_core[core_x - core_x_min, core_y - core_y_min, core_z - core_z_min] = origin_data[
                core_x, core_y, core_z]
            temp_center = [center_dict[key][0] - core_x_min, center_dict[key][1] - core_y_min,
                           center_dict[key][2] - core_z_min]
            D, V, size_ratio, angle = Get_DV(temp_core, temp_center)
            peak_coord = np.where(temp_core == temp_core.max())
            peak_coord = [(peak_coord[0] + core_x_min)[0], (peak_coord[1] + core_y_min)[0],
                          (peak_coord[2] + core_z_min)[0]]
            peak_value.append(origin_data[peak_coord[0], peak_coord[1], peak_coord[2]])
            peak_location.append(peak_coord)
            clump_center.append(center_dict[key])
            od_mass = origin_data[core_x, core_y, core_z]
            # od_mass = od_mass - od_mass.min()
            mass_array = np.c_[od_mass, od_mass, od_mass]
            clump_com.append(np.around((np.c_[mass_array] * core_dict[key]).sum(0) \
                                       / od_mass.sum(), 3).tolist())
            size = np.sqrt((mass_array * (np.array(core_dict[key]) ** 2)).sum(0) / od_mass.sum() - \
                           ((mass_array * np.array(core_dict[key])).sum(0) / od_mass.sum()) ** 2)
            clump_size.append(size.tolist())
            clump_sum.append(origin_data[core_x, core_y, core_z].sum())
            clump_volume.append(len(core_dict[key]))
            clump_angle.append(angle)
            regions_data[core_x, core_y, core_z] = k
            if core_x_min == 0 or core_y_min == 0 or core_z_min == 0 or \
                    core_x_max + 1 == data_size[0] or core_y_max + 1 == data_size[1] or core_z_max + 1 == data_size[2]:
                clump_edge.append(1)
            else:
                clump_edge.append(0)
    detect_infor_dict['peak_value'] = np.around(peak_value, 3).tolist()
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_center'] = np.array(clump_center).tolist()
    detect_infor_dict['clump_com'] = clump_com
    detect_infor_dict['clump_size'] = np.around(clump_size, 3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum, 3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
    #     if len(detect_infor_dict['peak_value']) == 0:
    #         raise Exception('No clumps!')
    return detect_infor_dict


def Detect_FacetClumps(RMS, Threshold, SWindow, KBins, FwhmBeam, VeloRes, SRecursionLBV, origin_data):
    if RMS > Threshold:
        raise Exception("RMS needs less than Threshold!")
    regions_1,regions_array_1 = Get_Regions_FacetClumps(origin_data, RMS, Threshold, np.array([0]))
    convs, hook_face = Convolve(origin_data * (regions_array_1 > 0), SWindow)
    com = Get_Total_COM(hook_face, regions_1, convs, KBins, SRecursionLBV)
    convs, hook_face = None, None
    new_regions_1, regions_array_1, rc_dict = Build_RC_Dict(com, regions_array_1, regions_1)
    regions_1 = None
    regions_2, regions_array_2 = Get_Regions_FacetClumps(origin_data, RMS, Threshold, regions_array_1)
    regions_array_1 = None
    new_regions_2, regions_array_2, rc_dict = Build_RC_Dict(com, regions_array_2, regions_2)
    com, regions_array_2, regions_2 = None, None, None
    mountain_array, mountain_dict, peak_dict, region_mp_dict = Build_MPR_Dict(origin_data, new_regions_2)
    new_regions_2 = None
    peak_dict = Updata_Peak_Dict(origin_data, mountain_dict, peak_dict)
    com_dict_record, core_dict_record = Get_CP_Dict(FwhmBeam, VeloRes, rc_dict, mountain_array, mountain_dict,peak_dict, region_mp_dict, origin_data)
    rc_dict, mountain_array, mountain_dict, peak_dict, region_mp_dict = None, None, None, None, None
    detect_infor_dict = DID_FacetClumps(SRecursionLBV, com_dict_record, core_dict_record, origin_data)
    return detect_infor_dict