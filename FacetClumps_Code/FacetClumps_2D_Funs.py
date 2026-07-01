#usr JiangYu
import numpy as np
from skimage import filters,measure,morphology
from scipy import signal
from scipy.stats import multivariate_normal
from tqdm import tqdm

def Get_Regions_FacetClumps(origin_data,RMS=0.1,threshold='otsu',temp_array=None):
    kopen_radius=1
    if threshold == 'mean':
        threshold = origin_data.mean()
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(origin_data)
    else:
        threshold = threshold
    open_data = morphology.opening(origin_data > threshold,morphology.disk(kopen_radius))
    if temp_array.ndim != 2:
        dilation_data = morphology.dilation(open_data,morphology.disk(kopen_radius))
        dilation_data = dilation_data*(origin_data > threshold)
        dilation_label = measure.label(dilation_data,connectivity=1)
    elif temp_array.ndim == 2:
        dilation_data = morphology.dilation(temp_array>0,morphology.disk(kopen_radius))
        dilation_data = dilation_data*(origin_data > RMS)
        xor_data = np.logical_xor(temp_array>0, dilation_data)
        and_data = np.logical_and(origin_data < threshold, xor_data)
        dilation_data = np.logical_or(temp_array,and_data)
#     dilation_data_1 = ndimage.binary_fill_holes(dilation_data_1)
        dilation_label = measure.label(dilation_data,connectivity=2)
    regions = measure.regionprops(dilation_label)
    regions_array = dilation_label
    return regions,regions_array

def Convolve(origin_data,SWindow):
    s = SWindow/2
    t = 17/5
    xres, yres = np.mgrid[0:SWindow:1,0:SWindow:1]
    xy = np.column_stack([xres.flat, yres.flat])
    sigma = np.array([s,s])
    covariance = np.diag(sigma**2)
    center = [np.int64(s),np.int64(s)]
    prob_density = multivariate_normal.pdf(xy, mean=center, cov=covariance)
    prob_density = prob_density.reshape((SWindow,SWindow))
    w = np.ones((SWindow,SWindow))
#     w = prob_density
    y = np.linspace(-np.int64(s),np.int64(s),SWindow)
    x = np.expand_dims(y,axis = 1)
    k1 = w/np.sum(w**2)
    k2 = w*x/np.sum((w*x)**2)
    k3 = w*y/np.sum((w*y)**2)
    k4 = w*(x**2-s)/np.sum((w*(x**2-s))**2)
    k5 = w*x*y/np.sum((w*x*y)**2)
    k6 = w*(y**2-s)/np.sum((y**2-s)**2)
    k7 = w*(x**3-t*x)/np.sum((x**3-t*x)**2)
    k8 = w*(x**2-s)*y/np.sum(((x**2-s)*y)**2)
    k9 = np.array(np.matrix(k8).T)
    k10 = np.array(np.matrix(k7).T)
    gx = k2 - t*k7 - 2*k9
    gy = k3 - t*k10 - 2*k8
    conv4 = signal.convolve(origin_data, k4, mode='same', method='direct')
    conv5 = signal.convolve(origin_data, k5, mode='same', method='direct')
    conv6 = signal.convolve(origin_data, k6, mode='same', method='direct')
    conv_gx = signal.convolve(origin_data, gx, mode='same', method='direct')
    conv_gy = signal.convolve(origin_data, gy, mode='same', method='direct')
    hook_face = np.zeros_like(origin_data)
    for k in [k1,k2,k3,k4,k5,k6,k7,k8,k9,k10]:
        hook_face += signal.convolve(origin_data, k, mode='same', method='direct')
    convs = [conv_gx,conv_gy,conv4,conv5,conv6]
    return convs,hook_face


def Calculate_Eig(eigenvalue_arrays, convs, region):
    coords = (region[:, 0], region[:, 1])
    H = np.array([
        [2*convs[2][coords], convs[3][coords]],
        [convs[3][coords], 2*convs[4][coords]]
    ])
    vals = np.linalg.eigvals(np.transpose(H, (2, 0, 1)))
    eigenvalue_arrays[0][coords], eigenvalue_arrays[1][coords] = vals[:, 0], vals[:, 1]
    return eigenvalue_arrays


def Get_Lable(convs, region, eigenvalue_arrays, bins, keig_bins):
    coords = (region[:, 0], region[:, 1])
    ex = np.histogram(eigenvalue_arrays[0][coords], bins=bins)[1][bins + keig_bins]
    ey = np.histogram(eigenvalue_arrays[1][coords], bins=bins)[1][bins + keig_bins]

    gx, gy = convs[0][coords], convs[1][coords]
    sx, sy = 2*np.std(gx), 2*np.std(gy)

    mask = (
        (eigenvalue_arrays[0][coords] < ex) &
        (eigenvalue_arrays[1][coords] < ey) &
        ((np.abs(gx) < sx) | (np.abs(gy) < sy))
    )
    return region[mask]


def Recursion_Lable(convs, region, regions_record, eigenvalue_arrays, \
                    lnV, logV, bins, keig_bins, SRecursionLB, recursion_time, regions_num):
    keig_bins = max(keig_bins-1, -lnV)
    
    b_min, l_min = region.min(0)
    box = np.zeros(region.max(0) - [b_min, l_min] + 2)
    box[region[:, 0]-b_min, region[:, 1]-l_min] = 1
    for r in measure.regionprops(measure.label(box, connectivity=1)):
        coords = r.coords + [b_min, l_min]
        area = r.area
        
        if area <= SRecursionLB and recursion_time > 1:
            if area >= logV or len(regions_record[0]) == 0:
                regions_record[0].append(coords)
            elif area > max(2, logV/2):
                regions_num += 1
        elif  area > SRecursionLB:
            region_new = Get_Lable(convs, coords, eigenvalue_arrays, bins, keig_bins)
            if len(region_new):
                regions_record = Recursion_Lable(convs, region_new, regions_record, eigenvalue_arrays, \
                                                 lnV, logV, bins, keig_bins, SRecursionLB, recursion_time+1, regions_num)
    
    return regions_record


def Merge_Regions_Record(regions_record):
    if regions_record[1]:
        regions_record[0] += [r[1] for r in regions_record[1]]
    return regions_record


def Get_COM(origin_data, regions_record):
    return np.array([
        np.around((origin_data[r[:,0], r[:,1]][:, None]*r).sum(0)/origin_data[r[:,0], r[:,1]].sum(), 3)
        for r in regions_record
    ])


def Get_Total_COM(origin_data, regions, convs, kbins, SRecursionLBV):
    eigenvalue_0, eigenvalue_1 = np.zeros_like(origin_data), np.zeros_like(origin_data)
    regions_record = [[], []]

    for reg in tqdm(range(len(regions))):
        coords = np.array(regions[reg].coords, dtype=int)
        lnV = int(np.log(len(coords)))
        logV = min(np.log10(len(coords)), 7)
        bins = kbins*lnV

        eigenvalue_arrays = [eigenvalue_0, eigenvalue_1]
        eigenvalue_arrays = Calculate_Eig(eigenvalue_arrays, convs, coords)
        
        regions_record = Recursion_Lable(convs, coords, regions_record, eigenvalue_arrays, lnV, logV, bins, -1, SRecursionLBV, 0, 0)

        eigenvalue_0[coords[:,0], coords[:,1]] = 0
        eigenvalue_1[coords[:,0], coords[:,1]] = 0

    regions_record = Merge_Regions_Record(regions_record)
    return Get_COM(origin_data, regions_record[0])


def GNC_FacetClumps(core_data,cores_coordinate):
    xres,yres = core_data.shape
    x_center = cores_coordinate[0]+1
    y_center = cores_coordinate[1]+1
    x_arange = np.arange(max(0,x_center-1),min(xres,x_center+2))
    y_arange = np.arange(max(0,y_center-1),min(yres,y_center+2))
    [x, y] = np.meshgrid(x_arange, y_arange);
    xy = np.column_stack([x.flat, y.flat])
    gradients = core_data[xy[:,0],xy[:,1]]\
                - core_data[x_center,y_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xy[g_step]-1)
    return gradients,new_center

def Build_RC_Dict(com,regions_array,regions_first):
    k1 = 0
    k2 = 0
    i_record = []
    temp_rc_dict = {}
    rc_dict = {}
    new_regions = []
    temp_regions_array = np.zeros_like(regions_array)
    for i in range(1,np.int64(regions_array.max()+1)):
        temp_rc_dict[i] = []
    center = np.array(np.around(com),dtype = 'uint32')
    for cent in center:
        if regions_array[cent[0],cent[1]] != 0 :
            temp_rc_dict[regions_array[cent[0],cent[1]]].append(com[k1])
            i_record.append(regions_array[cent[0],cent[1]])
        k1 += 1
    for i in range(1,np.int64(regions_array.max())+1):
        if i in i_record:
            coordinates = regions_first[i-1].coords
            temp_regions_array[(coordinates[:,0],coordinates[:,1])] = 1
            new_regions.append(regions_first[i-1])
            rc_dict[k2] = temp_rc_dict[i]
            k2 += 1
    return new_regions,temp_regions_array,rc_dict

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
        temp_origin_data[(coordinates[:, 0] + 1, coordinates[:, 1] + 1)] = \
            origin_data[(coordinates[:, 0], coordinates[:, 1])]
        for i in range(coordinates.shape[0]):
            temp_coords = []
            if mountain_array[coordinates[i][0], coordinates[i][1]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0], coordinates[i][1]] = k
                gradients, new_center = GNC_FacetClumps(temp_origin_data, coordinates[i])
                if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                    temp_coords.append(new_center)
                while gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                    mountain_array[new_center[0], new_center[1]] = k
                    gradients, new_center = GNC_FacetClumps(temp_origin_data, new_center)
                    if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                        temp_coords.append(new_center)
                mountain_array[np.stack(temp_coords)[:, 0], np.stack(temp_coords)[:, 1]] = \
                    mountain_array[new_center[0], new_center[1]]
                mountain_dict[mountain_array[new_center[0], new_center[1]]] += temp_coords
                if gradients.max() <= 0:
                    peak_dict[k].append(new_center)
                    region_mp_dict[reg].append(k)
                    k += 1
                    mountain_dict[k] = []
                    peak_dict[k] = []
    del (mountain_dict[k])
    del (peak_dict[k])
    return mountain_array, mountain_dict, peak_dict, region_mp_dict


def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    dists = np.sqrt(dist_1 + dist_2 + dist_3)
    return dists

def Connectivity_FacetClumps(core_dict_1,core_dict_2,i_num,j_num):
    i_region = np.array(core_dict_1[i_num])
    j_region = np.array(core_dict_2[j_num])
    x_min = np.r_[i_region[:,0],j_region[:,0]].min()
    x_max = np.r_[i_region[:,0],j_region[:,0]].max()
    y_min = np.r_[i_region[:,1],j_region[:,1]].min()
    y_max = np.r_[i_region[:,1],j_region[:,1]].max()
    box_data = np.zeros([x_max-x_min+2,y_max-y_min+2])
    box_data[i_region[:,0]-x_min,i_region[:,1]-y_min] = 1
    box_data[j_region[:,0]-x_min,j_region[:,1]-y_min] = 1
    box_label = measure.label(box_data,connectivity=2)
    box_region = measure.regionprops(box_label)
    return len(box_region)

def Delect(new_peak_dict, new_core_dict, com_keys_array, com_values_array, core_dict_center,
           key_peak_record, near_k_2, origin_data):
    del_key_peak_ids = []
    key_peak_id = 0
    for key_peak in key_peak_record:
        dist = Dists_Array([new_peak_dict[key_peak]], com_values_array)
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


def Update_CP_Dict_FacetClumps(new_peak_dict, com_dict, new_core_dict, mountain_array, origin_data):
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
        key_mountain = mountain_array[com_coord[0], com_coord[1]]
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
            dist_1 = Dists_Array([old_center], [new_peak_dict_2[key_mountain]])[0][0]
            dist_2 = Dists_Array([new_center], [new_peak_dict_2[key_mountain]])[0][0]
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
                    distance = Dists_Array([com_dict[key_center]], np_values_array)
                    distance_index_sort = np.argsort(distance[0])
                    npd_sort = np_keys_array[distance_index_sort]
                    key_peak_record = npd_sort[:near_k_2]
                    del_key_peak_ids = Delect(new_peak_dict, new_core_dict, com_keys_array,
                                              com_values_array, \
                                              core_dict_center, key_peak_record, near_k_2, origin_data)
                    real_del_ids = distance_index_sort[del_key_peak_ids]
                    real_del_ids = sorted(real_del_ids, reverse=True)
                    for real_del_id in real_del_ids:
                        np_keys_array = np.delete(np_keys_array, real_del_id, axis=0)
                        np_values_array = np.delete(np_values_array, real_del_id, axis=0)
        else:
            key_peak_record = np.array(list(new_peak_dict.keys()))
            del_key_peak_ids = Delect(new_peak_dict, new_core_dict, com_keys_array, com_values_array, \
                                      core_dict_center, key_peak_record, near_k_1, origin_data)
        temp_near_k = near_k_2
        near_k_2 = near_k_1
        near_k_1 += temp_near_k
        if near_k_2 > 8:
            near_k_2 = temp_near_k
    return com_dict_temp, core_dict_center


def Get_CP_Dict(rc_dict,mountain_array,mountain_dict,peak_dict,region_mp_dict,origin_data):
    core_id = 0
    com_dict_record = {}
    core_dict_record = {}
    for key in tqdm(rc_dict.keys()):
        new_peak_dict = {}
        com_dict = {}
        new_core_dict = {}
        mountain_keys = region_mp_dict[key]
        com_FacetClumpss = rc_dict[key]
        for mountain_key in mountain_keys:
            new_peak_dict[mountain_key] = peak_dict[mountain_key][0]
            new_core_dict[mountain_key] = mountain_dict[mountain_key]
        for com_FacetClumps in com_FacetClumpss:
            com_dict[core_id] = com_FacetClumps.tolist()
            core_id += 1
        com_dict_temp,core_dict_center = Update_CP_Dict_FacetClumps(new_peak_dict,com_dict,new_core_dict,mountain_array,origin_data)
        for key_center in core_dict_center.keys():
            com_dict_record[key_center] = com_dict_temp[key_center]
            core_dict_record[key_center] = core_dict_center[key_center]
    return com_dict_record,core_dict_record


def Get_DV(box_data,box_center):
    #2D
    box_region = np.where(box_data!= 0)
    A11 = np.sum((box_region[0]-box_center[0])**2*\
        box_data[box_region])
    A12 = -np.sum((box_region[0]-box_center[0])*\
        (box_region[1]-box_center[1])*\
        box_data[box_region])
    A21 = A12
    A22 = np.sum((box_region[1]-box_center[1])**2*\
        box_data[box_region])
    A = np.array([[A11,A12],[A21,A22]])/len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1,0]]
        V = V[[1,0]]

    if V[1][0]<0 and V[0][0]>0 and V[1][1]>0:
        V = -V
    size_ratio = np.sqrt(D[0]/D[1])
    angle = np.around(np.arccos(V[0][0])*180/np.pi-90,2)
    return D,V,size_ratio,angle

def DID_FacetClumps(SRecursionLB,center_dict,core_dict,origin_data):
    peak_value = []
    peak_location = []
    clump_center=[]
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_angle = []
    clump_regions = []
    clump_edge = []
    detect_infor_dict = {}
    k = 0
    regions_data = np.zeros_like(origin_data,dtype=np.int16)
    data_size = origin_data.shape
    for key in center_dict.keys():
        core_x = np.array(core_dict[key])[:,0]
        core_y = np.array(core_dict[key])[:,1]
        core_x_min = core_x.min()
        core_x_max = core_x.max()
        core_y_min = core_y.min()
        core_y_max = core_y.max()
        if len(core_dict[key])>SRecursionLB:
            k += 1
            temp_core = np.zeros((core_x_max-core_x_min+1,core_y_max-core_y_min+1))
            temp_core[core_x-core_x_min,core_y-core_y_min]=origin_data[core_x,core_y]
            temp_center = [center_dict[key][0]-core_x_min,center_dict[key][1]-core_y_min]
            D,V,size_ratio,angle = Get_DV(temp_core,temp_center)
            peak_coord = np.where(temp_core == temp_core.max())
            peak_coord = [(peak_coord[0]+core_x_min)[0],(peak_coord[1]+core_y_min)[0]]
            peak_value.append(origin_data[peak_coord[0],peak_coord[1]])
            peak_location.append(peak_coord)
            clump_center.append(center_dict[key])
            od_mass = origin_data[core_x,core_y]
    #         od_mass = od_mass - od_mass.min()
            mass_array = np.c_[od_mass,od_mass]
            clump_com.append(np.around((np.c_[mass_array]*core_dict[key]).sum(0)\
                        /od_mass.sum(),3).tolist())
            size = np.sqrt((mass_array*(np.array(core_dict[key])**2)).sum(0)/od_mass.sum()-\
                           ((mass_array*np.array(core_dict[key])).sum(0)/od_mass.sum())**2)
            clump_size.append(size.tolist())
            clump_sum.append(origin_data[core_x,core_y].sum())
            clump_volume.append(len(core_dict[key]))
            clump_angle.append(angle)
            regions_data[core_x,core_y] = k
            clump_regions.append([np.array(core_dict[key])[:,0],np.array(core_dict[key])[:,1]])
            if core_x_min == 0 or core_y_min == 0 or \
                core_x_max+1 == data_size[0] or core_y_max+1 == data_size[1]:
                clump_edge.append(1)
            else:
                clump_edge.append(0)
    detect_infor_dict['peak_value'] = np.around(peak_value,3).tolist()
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_center'] = np.array(clump_center).tolist()
    detect_infor_dict['clump_com'] = clump_com
    detect_infor_dict['clump_size'] = np.around(clump_size,3).tolist()
    detect_infor_dict['clump_sum'] = np.around(clump_sum,3).tolist()
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
    return detect_infor_dict

def Detect_FacetClumps(RMS,Threshold,SWindow,KBins,SRecursionLB,origin_data):
    if RMS > Threshold:
        raise Exception("RMS need less than Threshold!")
    regions_1,regions_array_1 = Get_Regions_FacetClumps(origin_data,RMS,Threshold,np.array([0]))
    convs,hook_face = Convolve(origin_data*regions_array_1,SWindow)
    com = Get_Total_COM(hook_face,regions_1,convs,KBins,SRecursionLB)
    new_regions_1,regions_array_1,rc_dict = Build_RC_Dict(com,regions_array_1,regions_1)
    regions_2,regions_array_2 = Get_Regions_FacetClumps(origin_data,RMS,Threshold,regions_array_1)
    new_regions_2,regions_array_2,rc_dict = Build_RC_Dict(com,regions_array_2,regions_2)
    mountain_array,mountain_dict,peak_dict,region_mp_dict = Build_MPR_Dict(origin_data,new_regions_2)
    com_dict_record,core_dict_record = Get_CP_Dict(rc_dict,mountain_array,mountain_dict,peak_dict,region_mp_dict,origin_data)
    detect_infor_dict = DID_FacetClumps(SRecursionLB,com_dict_record,core_dict_record,origin_data)
    return detect_infor_dict
