#author: Boris Panes, February 4, 2021
#this code contains several contributions from Christopher Eckner, Gulli and Roberto 
#specially concerning patch generation and photon flux

import random
import os

from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5

import numpy as np
import healpy as hp
import sys
import matplotlib.pyplot as plt
import matplotlib
from xml.dom import minidom
import math as mt

import h5py
from tqdm import tqdm_notebook, tnrange
from itertools import chain

from bisect import bisect
import cv2

import sys, re
from scipy.integrate import quad

np.random.seed()

def PLSuperExpCutoff2(E, K0, E0, gamma1, a, gamma2):
    return K0 * np.power(E/E0, -gamma1) * np.exp(-a*np.power(E, gamma2))

def LogParabola(E, N0, E0, alpha, beta):
    return N0 * np.power(E/E0, -(alpha+beta*np.log(E/E0)))
    
def get_photon_flux(catalog, is_psr = True):
    source_list = []    
    with open(catalog, 'r') as infl:
        if is_psr:
            for line in infl:
                if '</source>' in line:
                    source_list.append((l, b, photon_flux_1000, photon_flux_10000))
                    #list_of_pos.append((lon, lat))
                elif "name='Prefactor'" in line:
                    tmp = re.search("scale=\'([0-9a-z\-\.]*)\'", line)
                    K0 = float(tmp.group(1))
                elif "name='Index1'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    gamma1 = -1.0*float(tmp.group(1))
                elif "name='Scale'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    E0 = float(tmp.group(1))
                elif "name='Expfactor'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    a = float(tmp.group(1))
                elif "name='Index2'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    gamma2 = float(tmp.group(1))
                elif "name='RA'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    ra = float(tmp.group(1))
                elif "name='DEC'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    dec = float(tmp.group(1))
                    
                    photon_flux_1000  = quad(lambda x: PLSuperExpCutoff2(x, K0, E0, gamma1, a, gamma2), 1e3, 1e5)[0]
                    photon_flux_10000 = quad(lambda x: PLSuperExpCutoff2(x, K0, E0, gamma1, a, gamma2), 1e4, 1e6)[0]

                    coords = SkyCoord(ra = ra, dec = dec, unit = 'deg', frame = 'fk5')
                    b, l = coords.galactic.b.to_value(), coords.galactic.l.to_value()
        else:
            for line in infl:
                if '</source>' in line:
                    source_list.append((l, b, photon_flux_1000, photon_flux_10000))
                    #list_of_pos.append((lon, lat))
                elif "name='norm'" in line:
                    tmp = re.search("scale=\'([0-9a-z\-\.]*)\'", line)
                    N0 = float(tmp.group(1))
                elif "name='alpha'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    alpha = -1.0*float(tmp.group(1))
                elif "name='beta'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    beta = -1.0*float(tmp.group(1))
                elif "name='Eb'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    E0 = float(tmp.group(1))
                elif "name='RA'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    ra = float(tmp.group(1))
                elif "name='DEC'" in line:
                    tmp = re.search("value=\'([0-9a-z\-\.]*)\'", line)
                    dec = float(tmp.group(1))
                  
                    photon_flux_1000  = quad(lambda x: LogParabola(x, N0, E0, alpha, beta), 1e3, 1e5)[0]
                    photon_flux_10000 = quad(lambda x: LogParabola(x, N0, E0, alpha, beta), 1e4, 1e6)[0]

                    coords = SkyCoord(ra = ra, dec = dec, unit = 'deg', frame = 'fk5')
                    b, l = coords.galactic.b.to_value(), coords.galactic.l.to_value()
    return source_list

def DeclRaToIndex(NSIDE,decl,RA):
    return hp.pixelfunc.ang2pix(NSIDE,np.radians(-decl+90.),np.radians(360.-RA))

def get_patch_latitude():
    while True:
        b = np.random.uniform(0.,180)
        tmp = 0.5*np.sin(np.radians(b))
        x = np.random.uniform(0, 0.5)
        if tmp < x:
            continue
        else:
            return b-90.
            break

def get_sky_maps(map_IEM, map_AGN, map_PSR):
    hdu_atlas_IEM = hp.read_map(map_IEM, field = None, verbose = False)
    hdu_atlas_AGN = hp.read_map(map_AGN, field = None, verbose = False)
    hdu_atlas_PSR = hp.read_map(map_PSR, field = None, verbose = False)
    return (hdu_atlas_IEM, hdu_atlas_AGN, hdu_atlas_PSR)

#new approach
def get_ps_info_128(patch_centre, list_agn_xml, list_psr_xml, xsize_patch):
    
    list_row_agn, list_col_agn, list_flux_agn_1000, list_flux_agn_10000, list_ltrue_agn, list_btrue_agn = get_pixel_rc_list_from_xml_lb_list(xsize_patch, patch_centre, list_agn_xml)

    list_row_psr, list_col_psr, list_flux_psr_1000, list_flux_psr_10000, list_ltrue_psr, list_btrue_psr = get_pixel_rc_list_from_xml_lb_list(xsize_patch, patch_centre, list_psr_xml)
        
    agn_pos_list = np.zeros((len(list_row_agn),6)) 
    psr_pos_list = np.zeros((len(list_row_psr),6))  

    #agn sources
    #notice that we are using the 0 position for y0=list_row_agn and the first position for  x0=list_col_agn
    for i in range(len(list_row_agn)):
        
        y0 = list_row_agn[i]
        x0 = list_col_agn[i]

        flux_1000  = list_flux_agn_1000[i]
        flux_10000 = list_flux_agn_10000[i]
        
        ltrue = list_ltrue_agn[i]
        btrue = list_btrue_agn[i]
               
        agn_pos_list[i][0] = y0
        agn_pos_list[i][1] = x0
        agn_pos_list[i][2] = ltrue
        agn_pos_list[i][3] = btrue
        agn_pos_list[i][4] = flux_1000
        agn_pos_list[i][5] = flux_10000
              
    #psr sources
    for i in range(len(list_row_psr)):
        
        y0 = list_row_psr[i]
        x0 = list_col_psr[i]

        flux_1000  = list_flux_psr_1000[i]
        flux_10000 = list_flux_psr_10000[i]
        
        ltrue = list_ltrue_psr[i]
        btrue = list_btrue_psr[i]
        
        psr_pos_list[i][0] = y0
        psr_pos_list[i][1] = x0
        psr_pos_list[i][2] = ltrue
        psr_pos_list[i][3] = btrue
        psr_pos_list[i][4] = flux_1000
        psr_pos_list[i][5] = flux_10000
                  
    return len(list_row_agn), len(list_row_psr), agn_pos_list, psr_pos_list
    
def distance(y0,x0,y1,x1):
    return mt.sqrt((mt.pow(y1-y0,2) + mt.pow(x1-x0,2)))

def psf_bck_mask(y0, x0, radius, psf_mask):
    nrow = psf_mask.shape[0]
    ncol = psf_mask.shape[1]
    
    grid2D_psf = psf_mask.copy()
    grid2D_bck = np.ones((nrow, ncol)) - grid2D_psf
    
    for y in range(nrow):
        for x in range(ncol):
            
            #distance to the center
            s1 = distance(y0,x0,y,x)
           
            if (s1 < radius):
                grid2D_psf[y,x] = 1.0
                grid2D_bck[y,x] = 0.0
      
    return grid2D_psf, grid2D_bck

def RotMatrixY(psi, isdeg = True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), 0.0, -np.sin(np.radians(psi))],[0.0, 1.0, 0.0],\
                         [np.sin(np.radians(psi)), 0.0, np.cos(np.radians(psi))]])
    else:
        return np.array([[np.cos(psi), 0.0, -np.sin(psi)],[0.0, 1.0, 0.0],[np.sin(psi), 0.0, np.cos(psi)]])

def RotMatrixZ(psi, isdeg = True):
    if isdeg:
        return np.array([[np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0.0],[-np.sin(np.radians(psi)),\
                                                                                  np.cos(np.radians(psi)), 0.0],[0.0, 0.0, 1.0]])
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
        
#list of pixel positions in row, col format
def get_pixel_rc_list_from_xml_lb_list(xsize, lb_centre, list_of_pos_xml):

    lb_std_list = get_xml_lb_list_in_std_patch_coord(xsize, lb_centre, list_of_pos_xml)

    coord_range_x = np.linspace(-4.9609375, 4.9609375, xsize)
    coord_range_y = np.linspace(-4.9609375, 4.9609375, xsize)
    
    list_of_pixel_row = []
    list_of_pixel_col = []

    list_of_pixel_flux_1000 = []
    list_of_pixel_flux_10000 = []
     
    list_of_pixel_ltrue = []
    list_of_pixel_btrue = []
    #l = longitude
    #b = latitude

    for i in range(len(lb_std_list)):
        l, b, ltrue, btrue, flux_1000, flux_10000 = lb_std_list[i][0], lb_std_list[i][1], lb_std_list[i][2],\
                                                    lb_std_list[i][3], lb_std_list[i][4], lb_std_list[i][5]
        
        #print(l,b)
        
        pixel_l = xsize - bisect(list(coord_range_x),l)
        pixel_b = bisect(list(coord_range_y),b)
        
        list_of_pixel_row.append(pixel_b)
        list_of_pixel_col.append(pixel_l)

        list_of_pixel_flux_1000.append(flux_1000)
        list_of_pixel_flux_10000.append(flux_10000)
        
        list_of_pixel_ltrue.append(ltrue)
        list_of_pixel_btrue.append(btrue)
       
    return list_of_pixel_row, list_of_pixel_col, list_of_pixel_flux_1000, list_of_pixel_flux_10000, list_of_pixel_ltrue, list_of_pixel_btrue


#Trying to implement something more complicated
def get_xml_lb_list_in_std_patch_coord(xsize_patch, patch_centre, list_of_pos_xml):
    
    # corners around the patch_centre_position
    #l_p_min, b_p_min = get_lb_from_pixel(id_pixel(0,xsize_patch-1), patch_centre, xsize = xsize_patch)
    #l_p_max, b_p_max = get_lb_from_pixel(id_pixel(xsize_patch-1,0), patch_centre, xsize = xsize_patch)   
    
    l_a, b_a = get_lb_from_pixel(id_pixel(0,0,xsize_patch), patch_centre, xsize = xsize_patch)
    l_b, b_b = get_lb_from_pixel(id_pixel(xsize_patch-1,0,xsize_patch), patch_centre, xsize = xsize_patch)   
    
    l_c, b_c = get_lb_from_pixel(id_pixel(0,xsize_patch-1,xsize_patch), patch_centre, xsize = xsize_patch)
    l_d, b_d = get_lb_from_pixel(id_pixel(xsize_patch-1,xsize_patch-1,xsize_patch), patch_centre, xsize = xsize_patch)  
    
    #print(l_p_min, l_p_max, b_p_min, b_p_max)
    #print(l_a, l_b, l_c, l_d)
    #print(b_a, b_b, b_c, b_d)
    
    # corners around the center position
    #l_c_min, b_c_min = get_lb_from_pixel(id_pixel(0,xsize_patch-1), (0,0), xsize = xsize_patch)
    #l_c_max, b_c_max = get_lb_from_pixel(id_pixel(xsize_patch-1,0), (0,0), xsize = xsize_patch)
    
    l_c_a, b_c_a = get_lb_ps_centered((l_a, b_a), patch_centre)
    l_c_b, b_c_b = get_lb_ps_centered((l_b, b_b), patch_centre)
    l_c_c, b_c_c = get_lb_ps_centered((l_c, b_c), patch_centre)
    l_c_d, b_c_d = get_lb_ps_centered((l_d, b_d), patch_centre)
    
    #print(l_c_a, l_c_b, l_c_c, l_c_d)
    #print(b_c_a, b_c_b, b_c_c, b_c_d)
    
    l_arr = np.array([l_c_a, l_c_b, l_c_c, l_c_d])
    b_arr = np.array([b_c_a, b_c_b, b_c_c, b_c_d])
    
    #print(np.amin(l_arr), np.amax(l_arr))
    #print(np.amin(b_arr), np.amax(b_arr))
    
    l_c_min = np.amin(l_arr)
    l_c_max = np.amax(l_arr)
    b_c_min = np.amin(b_arr)
    b_c_max = np.amax(b_arr)
    
    list_of_lb_in_std_patch_coord = []
    
    for pos_con in range(len(list_of_pos_xml)):
        
        #positions in global lon lat coordinates
        l_pos = list_of_pos_xml[pos_con][0]
        b_pos = list_of_pos_xml[pos_con][1]
        
        #positions in standard patch coordinates
        l_c, b_c = get_lb_ps_centered((l_pos, b_pos), patch_centre)
        
        if ((l_c >= l_c_min) & (l_c <= l_c_max) & (b_c >= b_c_min) & (b_c <= b_c_max)):
            #print(pos_con, l_c, b_c, l_pos, b_pos)
            #we add the flux
            flux_pos_1000  = list_of_pos_xml[pos_con][2]
            flux_pos_10000 = list_of_pos_xml[pos_con][3] 
            
            list_of_lb_in_std_patch_coord.append((l_c, b_c, l_pos, b_pos, flux_pos_1000, flux_pos_10000))
            
    return list_of_lb_in_std_patch_coord

def get_lb_from_pixel(pixel_id, lb_centre, xsize=128, isdeg = True, is_lat = True): ##if input angles are in degree use 'isdeg = True'
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
    return l_PS, b_PS

#Function to implement inverse rotation
#to add in the predictions?
def get_lb_ps_centered(lb_ps, lb_centre, xsize=128, isdeg = True, is_lat = True): ##if input angles are in degree use 'isdeg = True'

    l_centre, b_centre = lb_centre
    r = np.dot(RotMatrixY(-b_centre), RotMatrixZ(l_centre))
    ######### 
    
    lon_PS_rotated, lat_PS_rotated = lb_ps
        
    xyz_PS_rotated = sph2xyz(1., 90. - lat_PS_rotated, lon_PS_rotated)
    x_PS, y_PS, z_PS = np.array(np.dot(r, xyz_PS_rotated), dtype = 'float32')
    #xyz_centre = sph2xyz(1., 90. - b_centre, l_centre)
    r, b_PS, l_PS = xyz2sph(x_PS, y_PS, z_PS, isdeg = isdeg, is_lat = is_lat)
    return l_PS, b_PS
    
def get_lb_from_rd(ra, dec):
    
    # From galactic to equatorial coordinates
    # https://docs.astropy.org/en/stable/coordinates/transforming.html
    
    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame='fk5')
    lon, lat = sc.galactic.l.degree, sc.galactic.b.degree  
    
    if (lon > 180):
        lon = lon - 360
    if (lon < -180):
        lon = 360 + lon
    
    return lon, lat
    

def id_pixel(row, col, xsize_patch):
    
    return xsize_patch*row + col 
    
def create_dataset(folder, file = "training.csv", n = 50, height = 128, width = 128, prefix="test",
                   n_classes=2, faint="F0", init_con = 0):

    #xsize_location=128 is tested for location identification,
    #so we start from this to later reduce to 64 coordinates
    #on the other hand patches are generated in 64x64 directly
    
    xsize_location = 128

    ##################################################
    #global 64x64 correction
    ##################################################
    xsize_patch_generation = 64   
    NSIDE = 256
    
    Npix = 12 * NSIDE * NSIDE
    pix_sr = 4.0 * np.pi / Npix

    lb_range = [-5, 5]
    proj = hp.projector.CartesianProj(lonra=lb_range, latra=lb_range, xsize=xsize_patch_generation)

    I,J = np.meshgrid(np.arange(xsize_patch_generation), np.arange(xsize_patch_generation))
    x,y = proj.ij2xy(I,J)

    dl = np.radians((lb_range[1]-lb_range[0])/(xsize_patch_generation-1))
    solid_area_ratio = dl*dl*np.cos(np.radians(y))

    #################################################
    
    print("\nDataset: ", prefix)
    print("Total number of patches: %d \n"%n)
    
    radius = 5

    #list for the csv files
    source_lines = []
    header_line="filename,xmin,xmax,ymin,ymax,class,lon_c,lat_c,flux_1000,lon_p,lat_p,catalog,flux_10000\n"
    
    f1 = open(os.path.join(folder, file), "w+")
    f1.writelines(header_line)
    f1.close()

    r = radius
    catalog_list_used = []

    n = n - init_con
    for cat_number in range(int(n/max_patches_per_catalog)):

        source_lines = []

        #save and delete the catalog number
        catalog_id = catalog_list[0]
        catalog_list.pop(0)
        
        patches = max_patches_per_catalog

        if (n < (cat_number+1)*max_patches_per_catalog):
            patches =  n - cat_number*max_patches_per_catalog
            if patches == 0:
                break

        #catalog paths are generated inside the loop in order to run over more than one sky instance
        if faint=="F0":              
            #normal xml and fits catalogs
            path_to_agn_test_cat_xml = f"{path_to_data}/data-raw/MOCK_4FGL_agn_{catalog_id}.xml"
            path_to_psr_test_cat_xml = f"{path_to_data}/data-raw/MOCK_4FGL_psr_{catalog_id}.xml"
        
            path_to_agn_test_cat_fits = f"{path_to_data}/data-raw/MOCK_4FGL_agn_{catalog_id}_healpix.fits"
            path_to_psr_test_cat_fits = f"{path_to_data}/data-raw/MOCK_4FGL_psr_{catalog_id}_healpix.fits"
            
        print("Backgroud file: ",path_to_background_fits) 

        #catalog_list_used.append(catalog_id)
        print("Catalogue: %d, %d"%(cat_number, catalog_id))

        print("AGN catalogue xml: ",  path_to_agn_test_cat_xml)
        print("PSR catalogue xml: ",  path_to_psr_test_cat_xml)

        print("AGN calalogue fits: ", path_to_agn_test_cat_fits)
        print("AGN calalogue fits: ", path_to_psr_test_cat_fits)
        
        print("Patches: %d of %d"%(patches,n))
        
        print("Obtaining the list of positions, photon flux_1000 and flux_10000 from xml catalogs")

        list_of_agn_lb_from_xml = get_photon_flux(path_to_agn_test_cat_xml, is_psr = False)
        list_of_psr_lb_from_xml = get_photon_flux(path_to_psr_test_cat_xml, is_psr = True)
        
        print("Loading fits catalogue")
        ###############################################################
        iem, agn, psr = get_sky_maps(path_to_background_fits, path_to_agn_test_cat_fits, path_to_psr_test_cat_fits)

        #part of the angle area correction
        iem /= pix_sr
        agn /= pix_sr
        psr /= pix_sr
        ##############################################################
        
        Nbins = len(iem)
        print('Image bins: ', Nbins)
        print('Generation of patches')
       
        for k in range(patches):
            
            centre_coordinate = []
            
            patch_iem = []
            patch_agn = []
            patch_psr = []

            #transformation from 0-360 to -180-180
            lon = (longitude[k]+180)%360 - 180

            lat = latitude[k]
            
            for i in range(Nbins):
                #notice that now we multiply each bin array by solid_area_ratio
                
                patch_agn_tmp = hp.visufunc.cartview(agn[i], rot = (lon, lat, 0.), coord = 'G', xsize = xsize_patch_generation, \
                                                     lonra = lb_range, latra = lb_range, return_projected_map = True)
                patch_agn.append(np.array(patch_agn_tmp) * solid_area_ratio)
                #plt.close()

                patch_psr_tmp = hp.visufunc.cartview(psr[i], rot = (lon, lat, 0.), coord = 'G', xsize = xsize_patch_generation, \
                                                     lonra = lb_range, latra = lb_range, return_projected_map = True)
                patch_psr.append(np.array(patch_psr_tmp) * solid_area_ratio)
                #plt.close()

                patch_iem_tmp = hp.visufunc.cartview(iem[i], rot = (lon, lat, 0.), coord = 'G', xsize = xsize_patch_generation, \
                                                     lonra = lb_range, latra = lb_range, return_projected_map = True)
                patch_iem.append(np.array(patch_iem_tmp) * solid_area_ratio) 
                #plt.close()
                
                #just to be sure we clear all the plot elements
                plt.cla()
                plt.clf()
                plt.close("all")
            
            IEM_element = np.array(patch_iem)
            AGN_element = np.array(patch_agn)
            PSR_element = np.array(patch_psr)
            centre_coordinate = np.array((lon,lat))
            
            #we recover the info using 128x128 patch dimensions
            #this remains from our initial approach. it does not affect at all the image and mask generation
            nagn, npsr, agn_pos_list, psr_pos_list = get_ps_info_128(centre_coordinate, list_of_agn_lb_from_xml, 
                                                                     list_of_psr_lb_from_xml, xsize_location)

            print('patch number: %d,  coordinate center: (%.2f, %.2f) '%(k, lon, lat))
            
            #output generation
            IEM_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)
            AGN_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)
            PSR_64 = np.zeros((xsize_patch_generation, xsize_patch_generation, Nbins), dtype=np.float32)

            for bin_k in range(Nbins):
                IEM_64[:,:,bin_k] = IEM_element[bin_k,:,:]
                AGN_64[:,:,bin_k] = AGN_element[bin_k,:,:]
                PSR_64[:,:,bin_k] = PSR_element[bin_k,:,:]

            #generation of the input image
            #we save IEM, AGN and PSR info separetely because it is useful for evaluations
            X_64 = np.zeros((3,64,64,5))

            X_64[0,:,:,:] = IEM_64
            X_64[1,:,:,:] = AGN_64
            X_64[2,:,:,:] = PSR_64
            
            #generation of patch total (sum of components) in image (tensor) like files
            out_fn = f"{prefix}_image_{init_con + (cat_number*max_patches_per_catalog + k)}.npy"
            np.save(os.path.join(folder, out_fn), X_64)
          
            #generation of csv file content
            grid2D_psf = np.zeros((xsize_patch_generation, xsize_patch_generation))
            grid2D_bck = np.ones((xsize_patch_generation, xsize_patch_generation))
            radius_64 = 2.5

            #agns
            id = 0          
            for i in range(nagn):
                
                y = min(agn_pos_list[i][0], xsize_location-1)
                x = min(agn_pos_list[i][1], xsize_location-1)

                ltrue = agn_pos_list[i][2]
                btrue = agn_pos_list[i][3]

                #true photon flux from xml
                flux_1000  = agn_pos_list[i][4]
                flux_10000 = agn_pos_list[i][5]
                
                xmin,xmax,ymin,ymax = max(0,x-r),min(xsize_location-1,x+r),max(0,y-r),min(xsize_location-1,y+r)

                xmin = xmin//2
                xmax = xmax//2
                ymin = ymin//2
                ymax = ymax//2

                #generation of the masks 
                #print("agn: ", nagn, xmin,xmax,ymin,ymax)
                grid2D_psf, grid2D_bck = psf_bck_mask(y//2, x//2, radius_64, grid2D_psf)
                
                source_lines.append(f"{out_fn},{int(xmin)},{int(xmax)},{int(ymin)},{int(ymax)},"+
                                    f"{int(id)},{float(lon)},{float(lat)},{float(flux_1000)}," +
                                    f"{float(ltrue)},{float(btrue)},{int(catalog_id)},{float(flux_10000)}\n")

            #pulsars
            id = 1
            for i in range(npsr):
                
                y = min(psr_pos_list[i][0], xsize_location-1)
                x = min(psr_pos_list[i][1], xsize_location-1)
                
                ltrue = psr_pos_list[i][2]
                btrue = psr_pos_list[i][3]

                #true photon flux from xml 
                flux_1000 = psr_pos_list[i][4]
                flux_10000 = psr_pos_list[i][5]
                                            
                xmin,xmax,ymin,ymax = max(0,x-r),min(xsize_location-1,x+r),max(0,y-r),min(xsize_location-1,y+r)

                xmin = xmin//2
                xmax = xmax//2
                ymin = ymin//2
                ymax = ymax//2

                #print("psr: ", npsr, xmin,xmax,ymin,ymax)

                grid2D_psf, grid2D_bck = psf_bck_mask(y//2, x//2, radius_64, grid2D_psf)
                
                source_lines.append(f"{out_fn},{int(xmin)},{int(xmax)},{int(ymin)},{int(ymax)}," +
                                    f"{int(id)},{float(lon)},{float(lat)},{float(flux_1000)},"+
                                    f"{float(ltrue)},{float(btrue)},{int(catalog_id)},{float(flux_10000)}\n")

            Y_p = np.zeros((xsize_patch_generation, xsize_patch_generation, 2))
            Y_p[:,:,0] = grid2D_psf
            Y_p[:,:,1] = grid2D_bck

            out_mk = f"{prefix}_masks_{init_con + cat_number*max_patches_per_catalog+k}.npy"
            np.save(os.path.join(folder, out_mk), Y_p)

            #we write just at the end of the process to avoid the repeated opening of the file

        f1 = open(os.path.join(folder, file), "a")
        f1.writelines(source_lines)
        f1.close()
    
    return 0

#February 4, 2020
#given AGN, PSR and Background fits file, generate 768 patches per sky instance 

#path to repo
path_to_data = '/home/bapanes/Research-Now/Gamma-Ray-Point-Source-Detector'

#path to backgroud B1
path_to_background_fits = f"{path_to_data}/data-raw/lat_alldata_9.5years_4fgl_P8R3_ULTRACLEANVETO_V2_FRONT_zmax105_gti_HEALPIX_256_IEM_V07_IGRB_P8R3_ULTRACLEANVETO_V2_FRONT+BACK.fits"

#path to output data
test_folder = f"{path_to_data}/../local/dm-local/data-test/data-768-F0-B1"

#Gulli's approach to generate a more uniform coverage of the sky
longitude, latitude = hp.pix2ang(8, np.arange(hp.nside2npix(8)),lonlat=True)

#patches per catalog
max_patches_per_catalog = len(longitude)

#catalog list
catalog_list = []
catalog_list.append(400)

#create output folder
os.makedirs(test_folder, exist_ok=True)
print("test folder: ", test_folder)

#generate patch catalog
previous_plot_backend=matplotlib.get_backend()
matplotlib.use('Agg')
cats_test = create_dataset(test_folder,  file="test.csv", prefix="test", n=len(longitude), faint="F0", init_con=0)
#cats_test = create_dataset(test_folder,  file="test.csv", prefix="test", n=5, faint="F0", init_con=0)
matplotlib.use(previous_plot_backend)
