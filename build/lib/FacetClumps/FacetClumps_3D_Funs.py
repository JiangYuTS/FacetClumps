#usr JiangYu
import numpy as np
from skimage import filters,measure,morphology
from scipy import signal
from scipy.stats import multivariate_normal
from tqdm import tqdm

def Get_Regions_Facet(origin_data,RMS=0.1,threshold='otsu',temp_array=None):
    kopen_radius=1
    if threshold == 'mean':
        threshold = origin_data.mean()
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(origin_data)
    else:
        threshold = threshold
    open_data = morphology.opening(origin_data > threshold,morphology.ball(kopen_radius))
    if temp_array.ndim != 3:
        dilation_data = morphology.dilation(open_data,morphology.ball(kopen_radius))
        dilation_data = dilation_data*(origin_data > threshold)
#         dilation_label = measure.label(dilation_data,connectivity=1)
    elif temp_array.ndim == 3:
        dilation_data = morphology.dilation(temp_array>0,morphology.ball(kopen_radius))
        dilation_data = dilation_data*(origin_data > RMS)
        xor_data = np.logical_xor(temp_array>0, dilation_data)
        and_data = np.logical_and(origin_data < threshold, xor_data)
        dilation_data = np.logical_or(temp_array,and_data)
#     dilation_data_1 = ndimage.binary_fill_holes(dilation_data_1)
    dilation_label = measure.label(dilation_data,connectivity=1)
    regions = measure.regionprops(dilation_label)
    regions_array = dilation_label
    return regions,regions_array

def Convolve(origin_data):
    axis = 1
    L = 2*axis+1
    s = 2*axis+1
    xres, yres, zres = np.mgrid[0:s:1,0:s:1,0:s:1]
    xyz = np.column_stack([xres.flat, yres.flat, zres.flat])
    sigma = np.array([1,1,1])
    covariance = np.diag(sigma**2)
    center = [int(s/2),int(s/2),int(s/2)]
    prob_density = multivariate_normal.pdf(xyz, mean=center, cov=covariance)
    prob_density = prob_density.reshape((s,s,s))
    w = np.ones((s,s,s))
    w = prob_density
    z = np.linspace(-axis,axis,s)
    x = np.expand_dims(z,axis = 1)
    y = np.expand_dims(z,axis = 1)
    x0 = np.repeat(x,s,axis =1)
    xt = np.repeat(x0,s,axis =0).reshape(s,s,s)
    xt = w*xt
    yt = w*y
    zt = w*z
    k1 = (5*s**2-(xt**2+yt**2+zt**2))/(2*np.sqrt(2*np.pi)*s**3)
    k2 = xt*(7*s**2-(xt**2+yt**2+zt**2))/(2*np.sqrt(2*np.pi)*s**5)
    k3 = yt*(7*s**2-(xt**2+yt**2+zt**2))/(2*np.sqrt(2*np.pi)*s**5)
    k4 = zt*(7*s**2-(xt**2+yt**2+zt**2))/(2*np.sqrt(2*np.pi)*s**5)
    k5 = -(s**2-xt**2)/(2*np.sqrt(2*np.pi)*s**5)
    k6 = -(s**2-yt**2)/(2*np.sqrt(2*np.pi)*s**5)
    k7 = -(s**2-zt**2)/(2*np.sqrt(2*np.pi)*s**5)
    k8 = xt*yt/(np.sqrt(2*np.pi)*s**5)
    k9 = xt*zt/(np.sqrt(2*np.pi)*s**5)
    k10 = yt*zt/(np.sqrt(2*np.pi)*s**5)
    k11 = xt*yt*zt/(np.sqrt(2*np.pi)*s**7)
    k12 = -xt*(s**2-yt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k13 = -xt*(s**2-zt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k14 = -yt*(s**2-xt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k15 = -yt*(s**2-zt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k16 = -zt*(s**2-xt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k17 = -zt*(s**2-yt**2)/(2*np.sqrt(2*np.pi)*s**7)
    k18 = -xt*(3*s**2-xt**2)/(6*np.sqrt(2*np.pi)*s**7)
    k19 = -yt*(3*s**2-yt**2)/(6*np.sqrt(2*np.pi)*s**7)
    k20 = -zt*(3*s**2-zt**2)/(6*np.sqrt(2*np.pi)*s**7)
    gx = k2+1/3*L**2*k12+1/3*L**2*k13+1/3*L**2*k18
    gy = k3+1/3*L**2*k14+1/3*L**2*k15+1/3*L**2*k19
    gz = k4+1/3*L**2*k16+1/3*L**2*k17+1/3*L**2*k20
    gxx = 2*k5
    gyy = 2*k6
    gzz = 2*k7
    gxy = k8
    gxz = k9
    gyz = k10
    conv_gx = signal.convolve(origin_data, gx, mode='same', method='auto')
    conv_gy = signal.convolve(origin_data, gy, mode='same', method='auto')
    conv_gz = signal.convolve(origin_data, gz, mode='same', method='auto')
    conv_gxx = signal.convolve(origin_data, gxx, mode='same', method='auto')
    conv_gyy = signal.convolve(origin_data, gyy, mode='same', method='auto')
    conv_gzz = signal.convolve(origin_data, gzz, mode='same', method='auto')
    conv_gxy = signal.convolve(origin_data, gxy, mode='same', method='auto')
    conv_gxz = signal.convolve(origin_data, gxz, mode='same', method='auto')
    conv_gyz = signal.convolve(origin_data, gyz, mode='same', method='auto')
    convs = [conv_gx,conv_gy,conv_gz,conv_gxx,conv_gyy,conv_gzz,conv_gxy,conv_gxz,conv_gyz]
    return convs

def Calculate_Eig(origin_data,convs,region):
    conv_gxx = convs[3]
    conv_gyy = convs[4]
    conv_gzz = convs[5]
    conv_gxy = convs[6]
    conv_gxz = convs[7]
    conv_gyz = convs[8]
    coords = (region[:,0],region[:,1],region[:,2])
    eigenvalue_0 = np.zeros_like(origin_data)
    eigenvalue_1 = np.zeros_like(origin_data)
    eigenvalue_2 = np.zeros_like(origin_data)
    for i,j,k in zip(coords[0],coords[1],coords[2]):
        A = np.array([[2*conv_gxx[i,j,k],conv_gxy[i,j,k],conv_gxz[i,j,k]],\
                       [conv_gxy[i,j,k],2*conv_gyy[i,j,k],conv_gyz[i,j,k]],\
                       [conv_gxz[i,j,k],conv_gyz[i,j,k],2*conv_gzz[i,j,k]]])
        a,b = np.linalg.eig(A)
        eigenvalue_0[i,j,k] = a[0]
        eigenvalue_1[i,j,k] = a[1]
        eigenvalue_2[i,j,k] = a[2]
    eigenvalue = [eigenvalue_0,eigenvalue_1,eigenvalue_2]
    return eigenvalue

def Get_Lable(origin_data,convs,region,eigenvalue,bins,keig_bins):
    times = 1
    conv_gx = convs[0]
    conv_gy = convs[1]
    conv_gz = convs[2]
    coords = (region[:,0],region[:,1],region[:,2])
    temp_data = np.zeros_like(origin_data,dtype = 'uint16')
    temp_data[coords] = 1
    label_data = np.zeros_like(origin_data,dtype = 'uint16')
    number_ex,eigs_x = np.histogram(eigenvalue[0][coords],bins = bins)
    number_ey,eigs_y = np.histogram(eigenvalue[1][coords],bins = bins)
    number_ez,eigs_z = np.histogram(eigenvalue[2][coords],bins = bins)
    eig_x = eigs_x[bins + keig_bins]
    eig_y = eigs_y[bins + keig_bins]
    eig_z = eigs_z[bins + keig_bins]
    x_line = convs[0][region[:,0],region[:,1],region[:,2]]
    y_line = convs[1][region[:,0],region[:,1],region[:,2]]
    z_line = convs[2][region[:,0],region[:,1],region[:,2]]
    gra_x_min = -times*np.std(x_line)
    gra_x_max = times*np.std(x_line)
    gra_y_min = -times*np.std(y_line)
    gra_y_max = times*np.std(y_line)
    gra_z_min = -times*np.std(z_line)
    gra_z_max = times*np.std(z_line)
    new_region = []
    for i,j,k in zip(coords[0],coords[1],coords[2]):
        if (eigenvalue[0][i,j,k]<eig_x and eigenvalue[1][i,j,k]<eig_y and eigenvalue[2][i,j,k]<eig_z)\
            and ((conv_gx[i,j,k]>gra_x_min and conv_gx[i,j,k]< gra_x_max)\
            or (conv_gy[i,j,k]>gra_y_min and conv_gy[i,j,k]< gra_y_max) \
            or (conv_gz[i,j,k]>gra_z_min and conv_gz[i,j,k]< gra_z_max)):
            label_data[i,j,k] = 1
            new_region.append([i,j,k])
    new_region = np.array(new_region)
    return new_region

def Recursion_Lable(origin_data,convs,region,regions_record,eigenvalue,logV,bins,keig_bins,Vmm,recursion_time):
    keig_bins -= 1
#     if keig_bins<-bins/10:
#         keig_bins = np.int(-bins/10)
    x_min = region[:,0].min()
    x_max = region[:,0].max()
    y_min = region[:,1].min()
    y_max = region[:,1].max()
    z_min = region[:,2].min()
    z_max = region[:,2].max()
    valid_lable = np.zeros([x_max-x_min+2,y_max-y_min+2,z_max-z_min+2])
    valid_lable[region[:,0]-x_min,region[:,1]-y_min,region[:,2]-z_min] = 1
    erosion_label = measure.label(valid_lable,connectivity=1)
    ero_regions_l = measure.regionprops(erosion_label)
    for region_l in ero_regions_l:
        coords_l = region_l.coords
        region = np.c_[coords_l[:,0]+x_min,coords_l[:,1]+y_min,coords_l[:,2]+z_min]
        if region_l.area <= Vmm and recursion_time*region_l.area >= Vmm and region_l.area >= logV:
            regions_record.append(region)
        elif recursion_time==0 and region_l.area > Vmm/2 and region_l.area <= Vmm:
            regions_record.append(region)
        elif  region_l.area > Vmm:
            recursion_time += 1
            region = Get_Lable(origin_data,convs,region,eigenvalue,bins,keig_bins)
            regions_record = Recursion_Lable(origin_data,convs,region,regions_record,eigenvalue,logV,bins,keig_bins,Vmm,recursion_time)
    return regions_record

def Get_COM(origin_data,regions_record):
    center_of_mass = []
    label_area_min = 0
    for final_region in regions_record:
        if len(final_region[:,0])> label_area_min :
            x_region = final_region[:,0]
            y_region = final_region[:,1]
            z_region = final_region[:,2]
            od_mass = origin_data[x_region,y_region,z_region]
            center_of_mass.append(np.around((np.c_[od_mass,od_mass,od_mass]*final_region).sum(0)\
                /od_mass.sum(),3).tolist())
    return center_of_mass

def Get_Total_COM(origin_data,regions,convs,kbins,Vmm):
    keig_bins = -1
    recursion_time = 0
    regions_record = []
    for i in tqdm(range(len(regions))):
        coords = regions[i].coords
        region = np.c_[coords[:,0],coords[:,1],coords[:,2]]
        lnV = np.int(np.log(len(coords)))
        logV = np.int(np.log10(len(coords)))
        bins = kbins*lnV
        eigenvalue = Calculate_Eig(origin_data,convs,region)
        regions_record = Recursion_Lable(origin_data,convs,region,regions_record,eigenvalue,logV,bins,keig_bins,Vmm,recursion_time)
    com = Get_COM(origin_data,regions_record)
    sorted_id_com = sorted(range(len(com)), key=lambda k: com[k], reverse=False)
    com = (np.array(com)[sorted_id_com])
    return com

def GNC_Facet(origin_data,cores_coordinate):
    xres,yres,zres = origin_data.shape
    x_center = cores_coordinate[0]+1
    y_center = cores_coordinate[1]+1
    z_center = cores_coordinate[2]+1
    x_arange = np.arange(max(0,x_center-1),min(xres,x_center+2))
    y_arange = np.arange(max(0,y_center-1),min(yres,y_center+2))
    z_arange = np.arange(max(0,z_center-1),min(zres,z_center+2))
    [x, y, z] = np.meshgrid(x_arange, y_arange, z_arange);
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    gradients = origin_data[xyz[:,0],xyz[:,1],xyz[:,2]]\
                - origin_data[x_center,y_center,z_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xyz[g_step]-1)
    return gradients,new_center

def Build_RC_Dict(com,regions_array,regions_first):
    k1 = 0
    k2 = 0
    i_record = []
    temp_rc_dict = {}
    rc_dict = {}
    new_regions = []
    temp_regions_array = np.zeros_like(regions_array)
    for i in range(1,np.int(regions_array.max()+1)):
        temp_rc_dict[i] = []
    center = np.array(com,dtype = 'uint16')
    for cent in center:
        if regions_array[cent[0],cent[1],cent[2]] != 0 :
            temp_rc_dict[regions_array[cent[0],cent[1],cent[2]]].append(com[k1])
            i_record.append(regions_array[cent[0],cent[1],cent[2]])
        k1 += 1
    for i in range(1,np.int(regions_array.max())+1):
        if i in i_record:
            coordinates = regions_first[i-1].coords
            temp_regions_array[(coordinates[:,0],coordinates[:,1],coordinates[:,2])] = 1
            new_regions.append(regions_first[i-1])
            rc_dict[k2] = temp_rc_dict[i]
            k2 += 1
    return new_regions,temp_regions_array,rc_dict

def Build_MPR_Dict(origin_data,regions):
    k = 1
    reg = -1
    peak_dict = {}
    peak_dict[k] = []
    mountain_dict = {}
    mountain_dict[k] = []
    region_mp_dict = {}
    mountain_array = np.zeros_like(origin_data)
    temp_origin_data = np.zeros(tuple(np.array(origin_data.shape)+2))
    for i in range(len(regions)):
        region_mp_dict[i] = []
    for region in tqdm(regions):
        reg += 1
        coordinates = region.coords
        temp_origin_data[(coordinates[:,0]+1,coordinates[:,1]+1,coordinates[:,2]+1)] = \
                    origin_data[(coordinates[:,0],coordinates[:,1],coordinates[:,2])]
        for i in range(coordinates.shape[0]):
            temp_coords = []
            if mountain_array[coordinates[i][0],coordinates[i][1],coordinates[i][2]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0],coordinates[i][1],coordinates[i][2]] = k
                gradients,new_center = GNC_Facet(temp_origin_data,coordinates[i])
                if gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                    temp_coords.append(new_center)
                while gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                    mountain_array[new_center[0],new_center[1],new_center[2]] = k
                    gradients,new_center = GNC_Facet(temp_origin_data,new_center)
                    if gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                        temp_coords.append(new_center)
                mountain_array[np.stack(temp_coords)[:,0],np.stack(temp_coords)[:,1],np.stack(temp_coords)[:,2]]=\
                    mountain_array[new_center[0],new_center[1],new_center[2]]
                mountain_dict[mountain_array[new_center[0],new_center[1],new_center[2]]] += temp_coords
                if gradients.max() <= 0:
                    peak_dict[k].append(new_center)
                    region_mp_dict[reg].append(k)
                    k += 1
                    mountain_dict[k] = []
                    peak_dict[k] = []
    del(mountain_dict[k])
    del(peak_dict[k])
    return mountain_array,mountain_dict,peak_dict,region_mp_dict

def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    dists = np.sqrt(dist_1 + dist_2 + dist_3)
    return dists

def Connectivity_Facet(core_dict_1,core_dict_2,i_num,j_num):
    i_region = np.array(core_dict_1[i_num])
    j_region = np.array(core_dict_2[j_num])
    x_min = np.r_[i_region[:,0],j_region[:,0]].min()
    x_max = np.r_[i_region[:,0],j_region[:,0]].max()
    y_min = np.r_[i_region[:,1],j_region[:,1]].min()
    y_max = np.r_[i_region[:,1],j_region[:,1]].max()
    z_min = np.r_[i_region[:,2],j_region[:,2]].min()
    z_max = np.r_[i_region[:,2],j_region[:,2]].max()
    box_data = np.zeros([x_max-x_min+2,y_max-y_min+2,z_max-z_min+2])
    box_data[i_region[:,0]-x_min,i_region[:,1]-y_min,i_region[:,2]-z_min] = 1
    box_data[j_region[:,0]-x_min,j_region[:,1]-y_min,j_region[:,2]-z_min] = 1
    box_label = measure.label(box_data,connectivity=3)
    box_region = measure.regionprops(box_label)
    return len(box_region)

def Delect(new_peak_dict,new_core_dict,peak_dict_center,core_dict_center,key_peak_record,near_k_2,origin_data):
    for key_peak in key_peak_record:
        dist = Dists_Array([new_peak_dict[key_peak]], list(peak_dict_center.values()))
        dist_index_sort = np.argsort(dist[0])
        pdc_sort = np.array(list(peak_dict_center.keys()))[dist_index_sort]
        i_num = key_peak
        for key_center in pdc_sort[:near_k_2]:
            j_num = key_center
            connectivity_1 = Connectivity_Facet(new_core_dict,core_dict_center,i_num,j_num)
            if connectivity_1 == 1:
                core_dict_center[key_center] += new_core_dict[key_peak]
                del new_core_dict[key_peak]
                del new_peak_dict[key_peak]
                break

def Update_CP_Dict_Facet(Vmm,new_peak_dict,peak_dict_center,new_core_dict,mountain_array,origin_data):
    peak_dict_temp = {}
    core_dict_center = {}
    key_mountain_record = []
    center_mountain = {}
    new_peak_dict_2 = {}
    for key in new_peak_dict.keys():
        new_peak_dict_2[key] = new_peak_dict[key]
    for key_center in peak_dict_center.keys():
        peak_coord = peak_dict_center[key_center]
        peak_coord = np.array(peak_coord,dtype = 'uint16')
        if mountain_array[peak_coord[0],peak_coord[1],peak_coord[2]]!=0:
            key_mountain = mountain_array[peak_coord[0],peak_coord[1],peak_coord[2]]
        if key_mountain not in key_mountain_record:
            peak_dict_temp[key_center] = peak_dict_center[key_center]
            core_dict_center[key_center] = new_core_dict[key_mountain]
            center_mountain[key_mountain] = key_center
            key_mountain_record.append(key_mountain)
            del new_core_dict[key_mountain]
            del new_peak_dict[key_mountain]
        else:
            old_center = peak_dict_temp[center_mountain[key_mountain]]
            new_center = peak_dict_center[key_center]
            dist_1 = ((old_center[0]-new_peak_dict_2[key_mountain][0])**2\
                    +(old_center[1]-new_peak_dict_2[key_mountain][1])**2\
                    +(old_center[2]-new_peak_dict_2[key_mountain][2])**2)**(1/2)
            dist_2 = ((new_center[0]-new_peak_dict_2[key_mountain][0])**2\
                    +(new_center[1]-new_peak_dict_2[key_mountain][1])**2\
                    +(new_center[2]-new_peak_dict_2[key_mountain][2])**2)**(1/2)
            if dist_2<dist_1:
                peak_dict_temp[center_mountain[key_mountain]] = peak_dict_center[key_center]
    peak_dict_center = peak_dict_temp
    near_k_1 = 0
    near_k_2 = 1
    near_number = 26
    while len(new_core_dict)>0:
        near_k_1 += 1
        if len(new_core_dict)>len(peak_dict_center.keys()):
            for key_center in peak_dict_center.keys():
                if len(new_core_dict)>0:
                    distance = Dists_Array([peak_dict_center[key_center]], list(new_peak_dict.values()))
                    distance_index_sort = np.argsort(distance[0])
                    npd_sort = np.array(list(new_peak_dict.keys()))[distance_index_sort]
                    key_peak_record = npd_sort[:near_number]
                    Delect(new_peak_dict,new_core_dict,peak_dict_center,core_dict_center,key_peak_record,near_k_2,origin_data)
        else:
            key_peak_record = np.array(list(new_peak_dict.keys()))
            Delect(new_peak_dict,new_core_dict,peak_dict_center,core_dict_center,key_peak_record,near_k_1,origin_data)
        if near_k_2 > len(peak_dict_center.keys())*8:
            break
        near_k_2 += near_k_1
        near_number += near_k_2
#     key_cdc = np.array(list(core_dict_center.keys()))
#     for key_center in key_cdc:
#         new_peak_dict = {}
#         new_core_dict = {}
#         key_peak_record = []
#         if len(core_dict_center[key_center])<Vmm and len(peak_dict_temp)==1:
#             del peak_dict_temp[key_center]
#             del core_dict_center[key_center]
#         elif len(core_dict_center[key_center])<Vmm and len(peak_dict_temp)>1:
#             key_peak_record.append(key_center)
#             new_peak_dict[key_center] = peak_dict_temp[key_center]
#             new_core_dict[key_center] = core_dict_center[key_center]
#             near_k_1 = 0
#             near_k_2 = 2
#             del peak_dict_temp[key_center]
#             while len(new_core_dict)>0:
#                 near_k_1 += 1
#                 Delect(new_peak_dict,new_core_dict,peak_dict_temp,core_dict_center,key_peak_record,near_k_2,origin_data)
#                 near_k_2 += near_k_1
#             del core_dict_center[key_center]
    return peak_dict_temp,core_dict_center

def Get_CP_Dict(Vmm,rc_dict,mountain_array,mountain_dict,peak_dict,region_mp_dict,origin_data):
    core_id = 0
    peak_dict_record = {}
    core_dict_record = {}
    for key in tqdm(rc_dict.keys()):
        new_peak_dict = {}
        peak_dict_center = {}
        new_core_dict = {}
        mountain_keys = region_mp_dict[key]
        com_Facets = rc_dict[key]
        for mountain_key in mountain_keys:
            new_peak_dict[mountain_key] = peak_dict[mountain_key][0]
            new_core_dict[mountain_key] = mountain_dict[mountain_key]
        for com_Facet in com_Facets:
            peak_dict_center[core_id] = com_Facet.tolist()
            core_id += 1
        peak_dict_temp,core_dict_center = \
            Update_CP_Dict_Facet(Vmm,new_peak_dict,peak_dict_center,new_core_dict,mountain_array,origin_data)
        for key_center in core_dict_center.keys():
            peak_dict_record[key_center] = peak_dict_temp[key_center]
            core_dict_record[key_center] = core_dict_center[key_center]
    return peak_dict_record,core_dict_record

def Get_DV(box_data,box_center):
    #3D
    box_data_sum = box_data.sum(0)
    box_region = np.where(box_data_sum!= 0)
    A11 = np.sum((box_region[0]-box_center[1])**2*\
        box_data_sum[box_region])
    A12 = -np.sum((box_region[0]-box_center[1])*\
        (box_region[1]-box_center[2])*\
        box_data_sum[box_region])
    A21 = A12
    A22 = np.sum((box_region[1]-box_center[2])**2*\
        box_data_sum[box_region])
    A = np.array([[A11,A12],[A21,A22]])/len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1,0]]
        V = V[[1,0]]
    if V[1][0]<0 and V[0][0]>0 and V[1][1]>0:
        V = -V
    size_ratio = np.sqrt(D[0]/D[1])
    angle = np.around(np.arccos(V[0][0])*180/np.pi-90,0)
    return D,V,size_ratio,angle

def DID_Facet(center_dict,core_dict,origin_data):
    peak_value = []
    peak_location = []
    clump_center=[]
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_regions = []
    clump_edge = []
    clump_angle = []
    single_clump = []
    detect_infor_dict = {}
    k = 0
    regions_data = np.zeros_like(origin_data)
    for key in center_dict.keys():
        k += 1
        core_x = np.array(core_dict[key])[:,0]
        core_y = np.array(core_dict[key])[:,1]
        core_z = np.array(core_dict[key])[:,2]
        core_x_min = core_x.min()
        core_x_max = core_x.max()
        core_y_min = core_y.min()
        core_y_max = core_y.max()
        core_z_min = core_z.min()
        core_z_max = core_z.max()
        temp_core = np.zeros((core_x_max-core_x_min+1,core_y_max-core_y_min+1,core_z_max-core_z_min+1))
        temp_core[core_x-core_x_min,core_y-core_y_min,core_z-core_z_min]=origin_data[core_x,core_y,core_z]
        temp_center = [center_dict[key][0]-core_x_min,center_dict[key][1]-core_y_min,center_dict[key][2]-core_z_min]
        D,V,size_ratio,angle = Get_DV(temp_core,temp_center)
        peak_coord = np.where(temp_core == temp_core.max())
        peak_coord = [(peak_coord[0]+core_x_min)[0],(peak_coord[1]+core_y_min)[0],(peak_coord[2]+core_z_min)[0]]
        peak_value.append(origin_data[peak_coord[0],peak_coord[1],peak_coord[2]])
        peak_location.append(peak_coord)
        clump_center.append(center_dict[key])
        od_mass = origin_data[core_x,core_y,core_z]
        mass_array = np.c_[od_mass,od_mass,od_mass]
        clump_com.append(np.around((np.c_[mass_array]*core_dict[key]).sum(0)\
                    /od_mass.sum(),3).tolist())
        size = np.sqrt((mass_array*(np.array(core_dict[key])**2)).sum(0)/od_mass.sum()-\
                       ((mass_array*np.array(core_dict[key])).sum(0)/od_mass.sum())**2)
        clump_size.append(size.tolist())
        clump_sum.append(origin_data[core_x,core_y,core_z].sum())
        clump_volume.append(len(core_dict[key]))
        clump_angle.append(angle)
        regions_data[core_x,core_y,core_z] = k
#         clump_regions.append([np.array(core_dict[key])[:,0],np.array(core_dict[key])[:,1],np.array(core_dict[key])[:,2]])
        data_size = origin_data.shape
        if core_x_min == 0 or core_y_min == 0 or core_z_min == 0 or \
            core_x_max+1 == data_size[0] or core_y_max+1 == data_size[1] or core_z_max+1 == data_size[2]:
            clump_edge.append(1)
        else:
            clump_edge.append(0)
        # single_clump.append(temp_core)
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
#     detect_infor_dict['single_clump'] = single_clump
#     detect_infor_dict['clump_regions'] = clump_regions
    return detect_infor_dict

def Detect_Facet(RMS,Threshold,KBins,Vmm,origin_data):
    if RMS > Threshold:
        raise Exception("RMS needs less than Threshold!")
    regions_1,regions_array_1 = Get_Regions_Facet(origin_data,RMS,Threshold,np.array([0]))
    convs = Convolve(origin_data)
    com = Get_Total_COM(origin_data,regions_1,convs,KBins,Vmm)
    new_regions_1,regions_array_1,rc_dict = Build_RC_Dict(com,regions_array_1,regions_1)
    regions_2,regions_array_2 = Get_Regions_Facet(origin_data,RMS,Threshold,regions_array_1)
    new_regions_2,regions_array_2,rc_dict = Build_RC_Dict(com,regions_array_2,regions_2)
    mountain_array,mountain_dict,peak_dict,region_mp_dict = Build_MPR_Dict(origin_data,new_regions_2)
    peak_dict_record,core_dict_record = Get_CP_Dict(Vmm,rc_dict,mountain_array,mountain_dict,peak_dict,region_mp_dict,origin_data)
    detect_infor_dict = DID_Facet(peak_dict_record,core_dict_record,origin_data)
    return detect_infor_dict

