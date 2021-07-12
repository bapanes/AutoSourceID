import sys, os
import numpy as np
import pandas as pd

from numpy import genfromtxt
import math as mt
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

#astropy physical distance in degrees
def distance_degrees(lon1, lat1, lon2, lat2):

    ra_1 = lon1*u.degree
    de_1 = lat1*u.degree
                    
    ra_2 = lon2*u.degree
    de_2 = lat2*u.degree
                    
    c1 = SkyCoord(ra = ra_1, dec = de_1)
    c2 = SkyCoord(ra = ra_2, dec = de_2)
        
    distance_c1c2 = c1.separation(c2).degree

    return distance_c1c2 

#cartesian distance in pixels
def distance_pixel(pred_yc, pred_xc, test_yc, test_xc):

    return mt.sqrt(pow(pred_yc - test_yc, 2) + pow(pred_xc - test_xc, 2))

#stats about tp, fp and fn for the pair test and pred
def stats_tp_fp_fn(test, pred, probability_threshold = 0.5, distance_degrees_threshold = 0.3, bl_alg=False):

    #we copy the test array in order to delete elements without changing
    #the original test array
    test_aux = np.copy(test)

    #in the following list we want to save relevant info for plots
    list_source_tp_fp_fn = []

    #These indices correspond to the last version of our data generation
    #and unet-kmeans predictions, which are the same after removal of source duplicates which
    #appear in overlapping patches
    test_patch_idx = 0
    
    test_xmin_idx = 1
    test_xmax_idx = 2
    test_ymin_idx = 3
    test_ymax_idx = 4

    test_lon_patch_idx = 6
    test_lat_patch_idx = 7
    
    #without snr
    test_flux_1000_ps_idx = 8
    test_lon_ps_idx = 9
    test_lat_ps_idx = 10
    test_cat_idx = 11

    test_flux_10000_ps_idx = 12

    #SNR column, created by this algorithm
    test_snr_box_idx = 13
    test_sbr_box_idx = 14
    
    #class extras
    test_class_ps_idx = 5
    
    #with snr
    #test_flux_ps_idx = 9
    #test_lon_ps_idx = 10
    #test_lat_ps_idx = 11
    #test_cat_idx = 12

    #prediction indices in the format uk-full
    #we have to figure out how to introduce this format in centroidnet prediction output
    pred_patch_idx = 0

    pred_yc_idx = 1
    pred_xc_idx = 2

    pred_classId_idx = 3

    pred_probability_idx = 4
    
    pred_lon_ps_idx = 7
    pred_lat_ps_idx = 8
    pred_cat_idx = 9

    pred_snr_box_idx = 10
    pred_sbr_box_idx = 11
    
    #distance threshold in degrees
    #distance_degrees_threshold = 0.3
    #distance_degrees_threshold = 0.6

    #global counters for tp, fp and fn
    tp_con = 0
    fp_con = 0
    fn_con = 0

    #list of border lines in the test are defined as those without a proper box around
    #such that we can evaluate them with the classification algorithm
    #these points are eliminated in advance in order to avoid false negatives
    #from border lines too
    list_of_bl = []

    xsize = 64
    r_b = 7
    box_inf = 3
    box_sup = 4
    for aux_con in range(len(test_aux)):

        test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])//2)
        test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])//2)

        xmin_b, xmax_b, ymin_b, ymax_b = max(0,test_xc-box_inf),min(xsize,test_xc+box_sup),\
                                         max(0,test_yc-box_inf),min(xsize,test_yc+box_sup)
        
        #we get rid of test border line stars since we do not want to count them as false negatives
        if (((xmax_b-xmin_b) != r_b) or ((ymax_b-ymin_b) != r_b)):
            list_of_bl.append(aux_con)

    if bl_alg:
        test_aux = np.delete(test_aux, list_of_bl, 0)

    #we loop over each one of the predictions in the ps basis
    #pred_con is not the patch number
    for pred_con in range(len(pred)):

        pred_cat = pred[pred_con, pred_cat_idx]
        pred_patch = pred[pred_con, pred_patch_idx]
        
        pred_lon = pred[pred_con, pred_lon_ps_idx]
        pred_lat = pred[pred_con, pred_lat_ps_idx]

        pred_id = pred[pred_con, pred_classId_idx] 
        
        pred_yc = pred[pred_con, pred_yc_idx]
        pred_xc = pred[pred_con, pred_xc_idx]

        pred_snr_box = pred[pred_con, pred_snr_box_idx]
        pred_sbr_box = pred[pred_con, pred_sbr_box_idx]
        
        #borderline in pred is just skipped
        xmin_b, xmax_b, ymin_b, ymax_b = max(0,int(pred_xc)-box_inf),min(xsize,int(pred_xc)+box_sup),\
                                         max(0,int(pred_yc)-box_inf),min(xsize,int(pred_yc)+box_sup)
        
        #we get rid of test border line stars since we do not want to count them as false negatives
        if bl_alg:
            if (((xmax_b-xmin_b) != r_b) or ((ymax_b-ymin_b) != r_b)):
                continue
        
        pred_probability = pred[pred_con, pred_probability_idx]
                    
        #we initialize the true positive flag as 1 in order
        #to indicate that there is not a tp yet
        tp_ban = 1

        #here we define the variables that we want to record for TP MAXIMUM flux                 
        pred_yc_max = pred_yc
        pred_xc_max = pred_xc
        pred_probability_max = pred_probability
        pred_snr_box_max = pred_snr_box
        pred_sbr_box_max = pred_sbr_box
        
        test_cat_max = pred_cat
        test_patch_max = pred_patch
        
        #the following variables only make sense when we find a potential TP
        pred_test_dg_max = 0
        pred_test_px_max = 0

        test_lon_max = 0
        test_lat_max = 0
                        
        test_flux_1000_max = 0
        test_flux_10000_max = 0

        test_yc_max = 0
        test_xc_max = 0

        test_lon_patch_max = 0
        test_lat_patch_max = 0

        test_id_max = -1

        test_snr_box_max = 0
        test_sbr_box_max = 0
        
        aux_con_flux_max = 0

        #here we define the variables for FP matches with MINIMUM distance variables
        pred_yc_min = pred_yc
        pred_xc_min = pred_xc
        pred_probability_min = pred_probability
        pred_snr_box_min = pred_snr_box
        pred_sbr_box_min = pred_sbr_box

        test_cat_min = pred_cat
        test_patch_min = pred_patch

        #the variables below only make sense when we find the nearest FP
        pred_test_dg_min = 1000
        pred_test_px_min = 1000

        test_lon_min = 0
        test_lat_min = 0
                        
        test_flux_1000_min = 0   
        test_flux_10000_min = 0       
                        
        test_yc_min = 0
        test_xc_min = 0

        test_lon_patch_min = 0
        test_lat_patch_min = 0

        test_id_min = -1
        
        test_snr_box_min = 0
        test_sbr_box_min = 0
        
        aux_con_flux_min = 0

        #for each prediction we loop over each of the true sources in test
        #until we find a true positive

        for aux_con in range(len(test_aux)):

            test_cat = test_aux[aux_con, test_cat_idx]
            test_patch = test_aux[aux_con, test_patch_idx]
            
            test_lon = test_aux[aux_con, test_lon_ps_idx]
            test_lat = test_aux[aux_con, test_lat_ps_idx]
            
            test_flux_1000  = test_aux[aux_con, test_flux_1000_ps_idx]
            test_flux_10000 = test_aux[aux_con, test_flux_10000_ps_idx]
            
            test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])/2)
            test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])/2)
            
            test_id = test_aux[aux_con, test_class_ps_idx]

            test_lon_patch = test_aux[aux_con, test_lon_patch_idx]
            test_lat_patch = test_aux[aux_con, test_lat_patch_idx]

            test_snr_box = test_aux[aux_con, test_snr_box_idx]
            test_sbr_box = test_aux[aux_con, test_sbr_box_idx]
            
            #print(pred_cat == test_cat)
                    
            #first, we have to ensure that we are comparing between pred and true sources
            #in the same catalogs
            #if (pred_cat == test_cat):
            if (pred_patch == test_patch): 

            #if True:
                #here we compute the geometrical distance between true and predicted sources
                pred_test_dg = distance_degrees(pred_lon, pred_lat, test_lon, test_lat)

                #distance in pixels
                pred_test_px = distance_pixel(pred_yc, pred_xc, test_yc, test_xc)

                #print(pred_con, pred_test_dg, pred_probability)
                if (pred_test_dg <= distance_degrees_threshold and pred_probability >= probability_threshold and pred_id >= 0):

                    #swith the true positive banner to tp = 0
                    tp_ban = 0

                    #for more than one TP match we choose the source with the highest flux
                    if test_flux_1000 > test_flux_1000_max:
                        pred_test_dg_max = pred_test_dg
                        pred_test_px_max = pred_test_px

                        test_lon_max = test_lon
                        test_lat_max = test_lat
                        
                        test_flux_1000_max = test_flux_1000
                        test_flux_10000_max = test_flux_10000

                        test_cat_max = test_cat
                        test_patch_max = test_patch
                        
                        pred_yc_max = pred_yc
                        pred_xc_max = pred_xc

                        test_yc_max = test_yc
                        test_xc_max = test_xc

                        test_lon_patch_max = test_lon_patch
                        test_lat_patch_max = test_lat_patch

                        test_id_max = test_id

                        test_snr_box_max = test_snr_box
                        pred_snr_box_max = pred_snr_box

                        test_sbr_box_max = test_sbr_box
                        pred_sbr_box_max = pred_sbr_box
                        
                        pred_probability_max = pred_probability

                        aux_con_flux_max = aux_con
                else:
                    #for FP matches (dg>0.3) we consider the nearest one
                    if (pred_test_dg < pred_test_dg_min):

                        pred_test_dg_min = pred_test_dg
                        pred_test_px_min = pred_test_px

                        test_lon_min = test_lon
                        test_lat_min = test_lat
                        
                        test_flux_1000_min = test_flux_1000   
                        test_flux_10000_min = test_flux_10000
                        
                        test_cat_min = test_cat
                        test_patch_min = test_patch
                        
                        pred_yc_min = pred_yc
                        pred_xc_min = pred_xc

                        test_yc_min = test_yc
                        test_xc_min = test_xc

                        test_lon_patch_min = test_lon_patch
                        test_lat_patch_min = test_lat_patch

                        test_id_min = test_id
                        
                        test_snr_box_min = test_snr_box
                        pred_snr_box_min = pred_snr_box

                        test_sbr_box_min = test_sbr_box
                        pred_sbr_box_min = pred_sbr_box
                      
                        pred_probability_min = pred_probability
            
        #if the tp_ban does not change from 1, the predicted source is a false positive
        #anyway, we save the information of the closest source from the true
        #be careful since the distance in pixel can be computed through different patches
        if (tp_ban == 1):

            test_lon = test_lon_min
            test_lat = test_lat_min
            
            test_flux_1000  = test_flux_1000_min
            test_flux_10000 = test_flux_10000_min
            
            pred_test_dg = pred_test_dg_min
            pred_test_px = pred_test_px_min

            test_cat = test_cat_min
            test_patch = test_patch_min
            
            pred_yc = pred_yc_min
            pred_xc = pred_xc_min

            test_yc = test_yc_min
            test_xc = test_xc_min

            test_lon_patch = test_lon_patch_min
            test_lat_patch = test_lat_patch_min

            test_snr_box = test_snr_box_min
            pred_snr_box = pred_snr_box_min
            
            test_sbr_box = test_sbr_box_min
            pred_sbr_box = pred_sbr_box_min
            
            pred_probability_min = pred_probability
            
            #false positive should be classified as fake
            test_id = test_id_min
            test_id_ps_fake = 2

            #add 1 to false possitives
            fp_con = fp_con + 1
            
            list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                         pred_test_dg, pred_test_px, test_flux_1000,\
                                         test_cat, test_patch, pred_yc, pred_xc, test_id_ps_fake, test_yc, test_xc,\
                                         test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, pred_snr_box,\
                                         test_id, test_snr_box, pred_sbr_box, test_sbr_box])

            print("false positive: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
                   pred_lon, pred_lat, test_lon, test_lat,\
                   pred_test_dg, pred_test_px, test_flux_1000, pred_snr_box)) 

        #true positives matched to the highest flux
        if (tp_ban == 0):
            pred_test_dg = pred_test_dg_max
            pred_test_px = pred_test_px_max

            test_lon = test_lon_max
            test_lat = test_lat_max
                        
            test_flux_1000  = test_flux_1000_max
            test_flux_10000 = test_flux_10000_max

            test_cat = test_cat_max
            test_patch = test_patch_max
                        
            pred_yc = pred_yc_max
            pred_xc = pred_xc_max

            test_yc = test_yc_max
            test_xc = test_xc_max

            test_lon_patch = test_lon_patch_max
            test_lat_patch = test_lat_patch_max

            test_snr_box = test_snr_box_max
            pred_snr_box = pred_snr_box_max

            test_sbr_box = test_sbr_box_max
            pred_sbr_box = pred_sbr_box_max

            pred_probability = pred_probability_max

            aux_con_flux = aux_con_flux_max

            #false positive should be classified as fake
            test_id = test_id_max
            test_id_ps_fake = test_id

            #add 1 to false possitives
            tp_con = tp_con + 1
            
            list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                         pred_test_dg, pred_test_px, test_flux_1000,\
                                         test_cat, test_patch, pred_yc, pred_xc, test_id_ps_fake, test_yc, test_xc,\
                                         test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, pred_snr_box,\
                                         test_id, test_snr_box, pred_sbr_box, test_sbr_box])

            print("true positive: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
                   pred_lon, pred_lat, test_lon, test_lat,\
                   pred_test_dg, pred_test_px, test_flux_1000, pred_snr_box)) 

            #We get rid of true positives in order to assume that the rest are false negatives
            test_aux = np.delete(test_aux, aux_con_flux, 0)

    #finally, we count as false negatives all the surviving sources from the true 
    fn_con = len(test_aux)

    tp_ban = 2
    pred_con = 0
    pred_lon = 0
    pred_lat = 0
    pred_test_dg = 0
    pred_test_px = 0

    pred_yc = 0
    pred_xc = 0
    
    for aux_con in range(len(test_aux)):

        test_lon = test_aux[aux_con, test_lon_ps_idx]
        test_lat = test_aux[aux_con, test_lat_ps_idx]
        
        test_flux_1000  = test_aux[aux_con, test_flux_1000_ps_idx]                 
        test_flux_10000 = test_aux[aux_con, test_flux_10000_ps_idx]
        
        test_id = test_aux[aux_con, test_class_ps_idx]
        test_cat =  test_aux[aux_con, test_cat_idx]
        test_patch = test_aux[aux_con, test_patch_idx]

        test_yc = int((test_aux[aux_con, test_ymin_idx] + test_aux[aux_con, test_ymax_idx])/2)
        test_xc = int((test_aux[aux_con, test_xmin_idx] + test_aux[aux_con, test_xmax_idx])/2)

        test_lon_patch = test_aux[aux_con, test_lon_patch_idx]
        test_lat_patch = test_aux[aux_con, test_lat_patch_idx]

        test_snr_box = test_aux[aux_con, test_snr_box_idx]
        test_sbr_box = test_aux[aux_con, test_sbr_box_idx]

        pred_probability = 1
        
        list_source_tp_fp_fn.append([tp_ban, pred_con, pred_lon, pred_lat, test_lon, test_lat,\
                                     pred_test_dg, pred_test_px, test_flux_1000,\
                                     test_cat, test_patch, pred_yc, pred_xc, test_id, test_yc, test_xc,\
                                     test_lon_patch, test_lat_patch, pred_probability, test_flux_10000, test_snr_box,\
                                     test_id, test_snr_box, test_sbr_box, test_sbr_box])

        print("false negative: %d %d (%1.2f, %1.2f) (%1.2f, %1.2f) %1.2f %1.2f %1.2e %1.2e"%(tp_ban, pred_con,\
               pred_lon, pred_lat, test_lon, test_lat,\
               pred_test_dg, pred_test_px, test_flux_1000, test_snr_box))
             
    return np.array(list_source_tp_fp_fn), tp_con, fp_con, fn_con

def main():
    path_to_data = f"/home/bapanes/Research-Now/Gamma-Ray-Point-Source-Detector"
    path_to_pred = f"{path_to_data}/outputs-csv"

    #summary file
    global_stats_file = "global_stats.csv"
    header_line = "dataset,precision,recall\n"
    
    f1 = open(os.path.join(path_to_pred, global_stats_file), "a+")
    f1.writelines(header_line)
    f1.close()

    #this is common for UNEK 
    path_to_test = f"{path_to_data}/data-test/data-768-F0-B1"
    test_file_name = f"{path_to_test}/test.csv"

    #PS basis
    #path_to_test = f"{path_to_data}/data-test/data_800_uniform_sky_cat_F0_B4_Id_FS/dataset_gamma_ray/test"
    #test_file_name = f"{path_to_test}/test_ps_basis.csv"

    #UNEK like file folders
    pred_file_name = f"{path_to_pred}/unek_prediction_test.csv"
    stat_file_name = f"{path_to_pred}/unek_evaluation_test.csv"
    
    print("test catalog: ", test_file_name[26:])
    print("inference file: ", pred_file_name[26:])
    print("evaluation file: ", stat_file_name[26:])

    #print("numbers before bl algorihm")
    #############################################################
    #open csv files and transform to arrays
    #############################################################

    test_data = pd.read_csv(test_file_name) 
    test_data_aux = np.array(test_data.iloc[:,:])

    #now we add the SNR column
    test_data_array = np.empty((test_data_aux.shape[0], test_data_aux.shape[1] + 2), dtype=object)        
    test_data_array[:,:-2] = test_data_aux[:,:]
        
    for con_test in range(len(test_data_array)):

        X_input = np.load(f"{path_to_test}/{test_data_array[con_test,0]}")
            
        xmin = test_data_array[con_test,1]
        xmax = test_data_array[con_test,2]
        ymin = test_data_array[con_test,3]
        ymax = test_data_array[con_test,4]

        ps_class = test_data_array[con_test,5]
            
        IEM_ph = X_input[0,(ymin+ymax)//2,(xmin+xmax)//2,:].sum()
        AGN_ph = X_input[1,(ymin+ymax)//2,(xmin+xmax)//2,:].sum()
        PSR_ph = X_input[2,(ymin+ymax)//2,(xmin+xmax)//2,:].sum()
            
        SNR_ph = 0
        SBR_ph = 0
        if IEM_ph > 0:
            #AGN point source
            if ps_class==0:
                SNR_ph = AGN_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph)
                SBR_ph = AGN_ph/(PSR_ph + IEM_ph)
                    
            #PSR point source
            if ps_class==1:
                SNR_ph = PSR_ph/np.sqrt(AGN_ph + IEM_ph + PSR_ph)
                SBR_ph = PSR_ph/(AGN_ph + IEM_ph)

        test_data_array[con_test,-2] = SNR_ph
        test_data_array[con_test,-1] = SBR_ph
    
    #pred data read
    pred_data = pd.read_csv(pred_file_name)

    #select only the 0 and 1 class as potential sources
    pred_data = pred_data[pred_data["class_id"]>=0]          
    pred_data_aux = np.array(pred_data.iloc[:,:])

    #now we add the SNR column
    pred_data_array = np.empty((pred_data_aux.shape[0], pred_data_aux.shape[1] + 2), dtype=object)        
    pred_data_array[:,:-2] = pred_data_aux[:,:]
        
    for con_pred in range(len(pred_data_array)):

        X_input = np.load(f"{path_to_test}/test_image_{int(pred_data_array[con_pred,0])}.npy")
        #print(X_input.shape)
            
        yc = int(pred_data_array[con_pred,1])
        xc = int(pred_data_array[con_pred,2])
            
        IEM_ph = X_input[0,yc,xc,:].sum()
        AGN_ph = X_input[1,yc,xc,:].sum()
        PSR_ph = X_input[2,yc,xc,:].sum()
            
        #notice that predicted positions are not directly associated with a given class
        SNR_ph = 0
        SBR_ph = 0
            
        if IEM_ph > 0:
               
            if AGN_ph >= PSR_ph:
                #The following line was corrected on June 25, 2021
                #Saptashawa found the bug
                SNR_ph = AGN_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph)
                SBR_ph = AGN_ph/(PSR_ph + IEM_ph)
                    
            if PSR_ph > AGN_ph:
                SNR_ph = PSR_ph/np.sqrt(PSR_ph + IEM_ph + AGN_ph)
                SBR_ph = PSR_ph/(AGN_ph + IEM_ph)
                
        pred_data_array[con_pred,-2] = SNR_ph
        pred_data_array[con_pred,-1] = SBR_ph

    print("test and pred: ",len(test_data), len(pred_data))
        
    #here we could transform the filename to numeric code and then ask
    #for the same patch comparison

    for con in range(test_data_array.shape[0]):
        filename = test_data_array[con,0]
        test_data_array[con,0] = int((filename.split('_')[2]).split('.')[0])
            
    print("length of test: ", len(test_data_array))
    print("length of pred: ", len(pred_data_array))
    
    ############################################################################
    #calling external function that compute the metrics TP-FP-FN in the ps basis
    ############################################################################

    stat_array, tp_tot, fp_tot, fn_tot = stats_tp_fp_fn(test_data_array, pred_data_array, \
                                                        probability_threshold = 0.0, distance_degrees_threshold = 0.3,\
                                                        bl_alg=False)

    ###############################################
    #Consistency check, both lines should be equal
    ###############################################
    
    print(tp_tot, fp_tot, fn_tot)
    print(stat_array.shape)
    print(len(np.where(stat_array[:,0]==0)[0]), len(np.where(stat_array[:,0]==1)[0]),len(np.where(stat_array[:,0]==2)[0]))
    
    #################################
    #global metrics 	
    #################################
    
    print("Global metric values")
    
    precision = round(tp_tot/(tp_tot + fp_tot)*100,2)
    recall = round(tp_tot/(tp_tot + fn_tot)*100,2)

    print('precision: ', precision)
    print('recall:', recall)       
        
    f1 = open(os.path.join(path_to_pred, global_stats_file), "a+")
    value_line = f"test_f0_b1,{precision},{recall}\n"
    f1.writelines(value_line)
    f1.close()
        
    #############################################################
    #file for plots with from-performance-file-to-figures.ipynb
    #############################################################

    #stat_code
    #0=tp
    #1=fp
    #2=fn

    #class
    #0=AGN, only for tp
    #1=PSR, only fot tp
    #2=FAKE, only for fp

    #class_ps
    #0=AGN, for tp max flux, fp min distance or fn
    #1=PSR, for tp max flux, fp min distance or fn
        
    stat_col_names = ["stat_code","pred_con","pred_lon","pred_lat","test_lon","test_lat",\
                      "distance_degree","distance_pixel","test_flux_1000","catalog","patch_number",\
                      "pred_y","pred_x","class","test_y","test_x","test_lon_patch","test_lat_patch",\
                      "probability","test_flux_10000","snr_center_pred","class_ps","snr_center_test",\
                      "sbr_center_pred","sbr_center_test"]

    stat_data_output = {stat_col_names[0]:stat_array[:,0], stat_col_names[1]:stat_array[:,1],\
                        stat_col_names[2]:stat_array[:,2], stat_col_names[3]:stat_array[:,3],\
                        stat_col_names[4]:stat_array[:,4], stat_col_names[5]:stat_array[:,5],\
                        stat_col_names[6]:stat_array[:,6], stat_col_names[7]:stat_array[:,7],\
                        stat_col_names[8]:stat_array[:,8], stat_col_names[9]:stat_array[:,9],\
                        stat_col_names[10]:stat_array[:,10], stat_col_names[11]:stat_array[:,11],\
                        stat_col_names[12]:stat_array[:,12], stat_col_names[13]:stat_array[:,13],\
                        stat_col_names[14]:stat_array[:,14], stat_col_names[15]:stat_array[:,15],\
                        stat_col_names[16]:stat_array[:,16], stat_col_names[17]:stat_array[:,17],\
                        stat_col_names[18]:stat_array[:,18], stat_col_names[19]:stat_array[:,19],\
                        stat_col_names[20]:stat_array[:,20], stat_col_names[21]:stat_array[:,21],\
                        stat_col_names[22]:stat_array[:,22], stat_col_names[23]:stat_array[:,23],\
                        stat_col_names[24]:stat_array[:,24]}

    stat_data_frame = pd.DataFrame(data=stat_data_output)
        
    stat_data_frame.to_csv(stat_file_name, sep=',', index=False)

    print("stats output file: ", stat_file_name)
     
if __name__ == '__main__':
    main()

 
