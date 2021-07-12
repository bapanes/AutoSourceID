import pandas as pd 
import numpy as np
import math as mt
import random as rd

from scipy.stats import poisson

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

from keras.models import load_model

from tqdm import tqdm_notebook, tnrange, tqdm
from sklearn.cluster import KMeans

import glob

#geometric utilities
def RotMatrixY(psi, isdeg = True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))],[0.0, 1.0, 0.0], \
                         [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])
    else:
        return np.array([[np.cos(psi), 0.0, -np.sin(psi)],[0.0, 1.0, 0.0],[np.sin(psi), 0.0, np.cos(psi)]]) 
    
def RotMatrixZ(psi, isdeg = True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0], \
                         [-np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0.0],[0.0, 0.0, 1.0]])
    else:
        return np.array([[np.cos(psi), np.sin(psi), 0.0],[-np.sin(psi), np.cos(psi), 0.0],[0.0, 0.0, 1.0]])
    
def sph2xyz(r, theta, phi, isdeg = True):
    if isdeg:
        return np.array([r*np.sin(np.radians(theta))*np.cos(np.radians(phi)), \
                         r*np.sin(np.radians(theta))*np.sin(np.radians(phi)), r*np.cos(np.radians(theta))])
    else:
        return np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
    
def xyz2sph(x, y, z, isdeg= True, is_lat = False):
    r = np.sqrt(x*x + y*y + z*z)
    if isdeg:
        phi = np.degrees(np.arctan2(y,x))
        lat = np.degrees(np.arctan2(z, np.sqrt(x*x + y*y)))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, 90. - lat, phi])
    else:
        phi = np.arctan2(y,x)
        lat = np.arctan2(z, np.sqrt(x*x + y*y))
        if is_lat:
            return np.array([r, lat, phi])
        else:
            return np.array([r, np.pi/2.0 - lat, phi])

def get_lb_from_pixel(pixel_id, lb_centre, xsize=128, isdeg = True, is_lat = True):
##if input angles are in degree use 'isdeg = True'
######### Generate (l,b) coordinate map of 10x10deg patch ######
    
    #Following the suggestions of CA mail
    if (xsize == 100):
        coord_range = np.linspace(-4.95, 4.95, xsize)
    
    if (xsize == 128):
        coord_range = np.linspace(-4.9609375, 4.9609375, xsize)
    
    X, Y = np.meshgrid(coord_range, coord_range)
    lonlat_patch = list(zip(np.flip(X.flatten()), Y.flatten()))
    ######### Get rotation matrix used to rotate the original centre to (0., 0.) #########
    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    ######### 
    
    lon_PS_rotated, lat_PS_rotated = lonlat_patch[pixel_id]
     
    #x_pixel_normal = int(pixel_id % xsize)
    #y_pixel_normal = int(pixel_id / xsize)
        
    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r.T, xyz_PS_rotated), dtype = 'float32')
    #xyz_centre = sph2xyz(1., 90. - b_centre, l_centre)
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, isdeg = isdeg, is_lat = is_lat)
    
    if l_PS < 0:
        l_PS = 360 + l_PS
    
    return l_PS, b_PS

def pixel_id(row, col, xsize_patch):
    
    return xsize_patch*row + col

#########################
#unet kmeans routines
#########################

def try_kmeans_with_error_method(grid2D_pred, outer_radius, inner_radius, nmin, nmax):
    
    max_intensity = 0
    max_predicted_centers = []
    max_nsources = 0
    
    #we have to work on these numbers, although they seem to be working fine 
    penalty_factor = intensity_penalty_points_uk
    
    xsize = grid2D_pred.shape[0]
    
    #print("Number of sources, Penalized intensity")
    
    for nsources in range(nmin, nmax):
          
        predicted_centers = list_of_centers_kmeans(grid2D_pred, nsources)

        if len(predicted_centers)>0:
        
            grid2D_kmeans = create_disks_from_list_of_centers(xsize, outer_radius, predicted_centers)
        
            intensity = penalized_intensity(predicted_centers, grid2D_pred, inner_radius, penalty_factor)
              
            if (intensity > max_intensity):
                max_intensity = intensity
            
                max_predicted_centers = predicted_centers
                max_nsources = nsources
            
    return max_nsources, max_predicted_centers

def list_of_centers_kmeans(grid2D, ncenters):
    X = []

    xsize = grid2D.shape[0]
    
    for i in range(xsize):
        for j in range(xsize):
            if (grid2D[i,j] > label_score_uk):
                X.append((i,j))

    positions = []
                
    X = np.array(X)

    if (len(X)>0):
        
        ncenters = min(len(X), ncenters)
    
        kmeansModel = KMeans(n_clusters = ncenters).fit(X)
        kmeansModel.fit(X)
    
        positions = kmeansModel.cluster_centers_
    
    
    return positions

def create_disks_from_list_of_centers(xsize, radius, list_of_centers):
    
    grid2D = np.zeros((xsize, xsize))
    
    for center in list_of_centers:
        grid2D = PSF(center[0], center[1], radius, grid2D)
        
    return grid2D

def PSF(x0, y0, s0, grid2D):
    nrow = grid2D.shape[0]
    ncol = grid2D.shape[1]
    
    grid2D_new = grid2D.copy()
    
    for x in range(ncol):
        for y in range(nrow):
                
            s1 = distance(x0,y0,x,y)
               
            if s1<=s0: 
                grid2D_new[x,y] = 1
                    
    return grid2D_new 

def distance(x0,y0,x1,y1):
    return mt.sqrt((mt.pow(x1-x0,2) + mt.pow(y1-y0,2)))

def penalized_intensity(predicted_centers, grid2D_pred, inner_radius, penalty_factor):
    
    intensity = 0
    
    xsize = grid2D_pred.shape[0]
    grid2D_copy = grid2D_pred.copy()
    
    for n in range(predicted_centers.shape[0]):
        row = int(predicted_centers[n][0])
        col = int(predicted_centers[n][1])
        
        for sub_row in range(row-inner_radius, row+inner_radius):
            for sub_col in range(col-inner_radius, col+inner_radius):
                if ((sub_row > 0 and sub_row < xsize) and (sub_col > 0 and sub_col < xsize)):
                    intensity = intensity + grid2D_copy[sub_row][sub_col]
                    grid2D_copy[sub_row][sub_col] = penalty_factor
            
    return intensity

def cut_and_order_list(list):
    
    new_list = []
    
    for n in range(len(list)):
        
        if (list[n,0] > 0 and list[n,1] > 0):
            new_list.append([round(list[n,0],2), round(list[n,1],2)])
        else:
            break
      
    return np.array(sorted(new_list))

##########################
#Main program
##########################

nrow = 64
ncol = 64
xsize = 64

input_bins = 5
output_bins = 2

#kmeans optimization
label_score_uk = 0.2
internal_radius_uk = 2
external_radius_uk = 3

intensity_penalty_points_uk = -10

#min and max number of sources to search for
nmin = 1
nmax = 50
#we may increase the max number of sources to analyze agn faint sample

def main():

    ######################################################################################
    #INPUT and OUTPUT files
    #####################################################################################
    
    #path to results
    path_to_data = f"/home/bapanes/Research-Now/Gamma-Ray-Point-Source-Detector"
    path_to_results_file = f"{path_to_data}/outputs-csv/unek_prediction_test.csv"
    
    #path to test file
    path_to_test = f"{path_to_data}/../local/dm-local/data-test/data-768-F0-B1"
    test_file = f"{path_to_test}/test.csv"

    #unet model
    unet_model_file = f"{path_to_data}/models/unet_model.h5"

    ########################################################################################
    
    #load Unet model weights
    unet_model = load_model(unet_model_file)
    
    #load test file
    test_data = pd.read_csv(test_file)

    #list of file names in test data
    list_of_names = test_data["filename"].unique()

    #Generation of predictions
    number_of_test_files = len(list_of_names)

    list_file_number = []
    list_catalog = []
    list_lon_c = []
    list_lat_c = []

    list_xc = []
    list_yc = []
    list_lon_ps = []
    list_lat_ps = []
    list_class_id = []
    list_probability = []

    con_50_plus = 0

    print("evaluation of %d input images"%(number_of_test_files))

    for con_test_file in range(number_of_test_files):
            
        #read the files
        image_file = list_of_names[con_test_file]

        #get the information about this image_file in particular
        sub_df = test_data[test_data["filename"]==image_file]

        #for each file in test data there is one lon_c, lat_C
        lon_c_file = sub_df["lon_c"].iloc[0]
        lat_c_file = sub_df["lat_c"].iloc[0]
        catalog_file = sub_df["catalog"].iloc[0]
        
        print(image_file)
        
        masks_file = image_file.replace("image", "masks")
        
        print("image file: ", image_file)
        print("masks file: ", masks_file)
        
        X1 = np.zeros((1, 3, xsize, xsize, input_bins), dtype=float)

        Xa = np.zeros((1, xsize, xsize, input_bins), dtype=float)
        Xp = np.zeros((1, xsize, xsize, input_bins), dtype=float)
        Y = np.zeros((1, xsize, xsize, output_bins), dtype=float)

        #read new format X data and transform to usual shapes
        X1[0,:,:,:,:] = np.load(f'{path_to_test}/{image_file}')
        Y[0,:,:,:] = np.load(f'{path_to_test}/{masks_file}')

        Xa[0,:,:,:] = X1[0,0,:,:,:] + X1[0,1,:,:,:] + X1[0,2,:,:,:]
            
        #Poisson effect !!
        Xp[0,:,:,:] = poisson.rvs(Xa[0,:,:,:])
        
        #unet prediction
        unet_pred = unet_model.predict(Xp, verbose=1)
        
        #layer with the source probs
        grid2D_pred = unet_pred[0,:,:,0]

        #disk radius to evaluate the intensity, not necessarily the same as 
        #masks radius, which is 5, but normally equal or smaller
        outer_radius = external_radius_uk
        inner_radius = internal_radius_uk
        
        pred_nsources, pred_centers = try_kmeans_with_error_method(grid2D_pred, outer_radius, inner_radius, nmin, nmax)

        #cut and order the list
        pred_centers     = cut_and_order_list(pred_centers)
        ordered_nsources = len(pred_centers)
        
        for con_ctd in range(ordered_nsources):
            
            list_file_number.append(con_test_file)    
            list_lon_c.append(lon_c_file)
            list_lat_c.append(lat_c_file)
            
            ycentroid = int(pred_centers[con_ctd][0])
            xcentroid = int(pred_centers[con_ctd][1])

            #notice that here we send the pixel id considering a patch of 128 pixels
            #so we multiply the ycentroid, xcentroid and xsize times 2
            l_ps, b_ps = get_lb_from_pixel(pixel_id(ycentroid*2,xcentroid*2,xsize*2), [lon_c_file, lat_c_file])
            
            prob_at_centroid = grid2D_pred[ycentroid, xcentroid]
            class_id = 0
            
            #append results to list
        
            list_yc.append(ycentroid*1.0)
            list_xc.append(xcentroid*1.0)
            
            list_class_id.append(class_id*1.0)
            list_probability.append(prob_at_centroid)
            
            list_lon_ps.append(l_ps)
            list_lat_ps.append(b_ps)
            
            list_catalog.append(catalog_file)
            
            print(con_test_file, ycentroid, xcentroid, class_id, prob_at_centroid, pred_nsources)
            
            if (pred_nsources >= nmax):
                print("ATTENTION HERE!!")
                con_50_plus = con_50_plus + 1
                
            #output data generation in centroid format
            output_data_frame = pd.DataFrame(data={"image_nr": list_file_number,\
                                                   "centroid_y": list_yc, "centroid_x": list_xc,\
                                                   "class_id": list_class_id, "probability": list_probability, \
                                                   "lon_c": list_lon_c, "lat_c":list_lat_c,\
                                                   "lon_ps": list_lon_ps, "lat_ps": list_lat_ps, \
                                                   "catalog": list_catalog})
            
            #change the sep to , for the next round of outputs
            output_data_frame.to_csv(f"{path_to_results_file}", sep=',', index=False)
            

    #number of patches with 50+ predicted sources
    print(con_50_plus)
    
    return 0

if __name__ == "__main__":
    main()
