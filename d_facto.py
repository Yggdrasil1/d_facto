#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:20:06 2017

@author: Kai Winkler
"""

from __future__ import print_function
import numpy as np
import h5py
from matplotlib import pyplot as plt
import Applying_filter as arf
import Kmeans as kc
import Meanshift as ms
import os
import time
import math



filename = str(input("Please put the absolut filepath of your data here: "))

#/////////////////////////////////////////////////////////////

show_filter = False
show_filter_q = str(input("Do you wish to show the sumspectrum + RGB Filter?"
                          +"\n(y,n): "))

if show_filter_q in ['y','Y','Yes','yes']:
     show_filter = True
     
#//////////////////////////////////////////////////////////////

save_filter = False
save_filter_q = str(input("Do you wish to save the sumspectrum + RGB Filter?"
                          +"\n(y,n): "))

if save_filter_q in ['y','Y','Yes','yes']:
     save_filter = True     
     
#//////////////////////////////////////////////////////////////

show_filtered_images = False
show_images_q = str(input("Do you wish to see the resulting images from the RGB Filters?"
                          +"\n(y,n): "))

if show_images_q in ['y','Y','Yes','yes']:
     show_filtered_images = True 
     
#//////////////////////////////////////////////////////////////

show_clustered_images = False
show_clustered_images_q = str(input("Do you wish to cluster the data (KMeans)"
                                    + "\n(y,n): "))

if show_clustered_images_q in ['y','Y','Yes','yes']:
     show_clustered_images = True 
     cluster_amount = int(input("How many clusters shall be used?: "))
     
#//////////////////////////////////////////////////////////////

show_meanshifted_images = False
show_meanshifted_images_q = str(input("Do you wish to cluster the data (Meanshift)"
                                    + "\n(y,n): "))

if show_meanshifted_images_q in ['y','Y','Yes','yes']:
     show_meanshifted_images = True   
     
#//////////////////////////////////////////////////////////////

show_pca = False
show_pca_q = str(input("Do you wish to apply PCA to your data?"
                          +"\n(y,n): "))

if show_pca_q in ['y','Y','Yes','yes']:
     show_pca = True   

#//////////////////////////////////////////////////////////////

save_images = False
save_images_q = str(input("Do you wish to save all generated images?"
                          +"\n(y,n): "))

if save_images_q in ['y','Y','Yes','yes']:
     save_images = True   

#//////////////////////////////////////////////////////////////

#start of tracking the time d_facto needs to evaluate and visualize the data
start_d_facto=time.time()

#//////////////////////////////////////////////////////////////
#creating the filename and a directory from the given filepath
savename=filename[:-3]+"/"
savedirectory=os.path.dirname(savename)
if not os.path.exists(savedirectory):
        os.makedirs(savedirectory)

#//////////////////////////////////////////////////////////////
#reading the hdf5 file
f = h5py.File(filename, 'r')

#converting the hdf5 data into an array
data=f['Raw'][:]

f.close()

#//////////////////////////////////////////////////////////////
#creates and applies the RGB filter to the data
data_and_info=arf.RGBfilter.makeRGBfilter(data, show_filter,save_filter,savedirectory)

#list of all RGB images 
image_list=data_and_info[0]

#//////////////////////////////////////////////////////////////
#list of all Laplacian variances corresponding to the RGB images
laplacian=[]

for i in range(len(image_list)):
    laplacian.append(arf.variance_of_laplacian(image_list[i]))

#//////////////////////////////////////////////////////////////
#Printing all RGB images if wanted
if(show_filtered_images==True):
    
    for i in range(len(data_and_info[0])):
        print(data_and_info[1][i])
        print("time aquired = ",data_and_info[2][i])
        plt.show(block=False)
        plt.imshow(data_and_info[0][i])
        plt.show
        plt.show(block=False)

#//////////////////////////////////////////////////////////////
#Clustering the imaes with Meanshift and printing the images if wanted
if(show_meanshifted_images==True):
    
     meanshifted_list=[ms.Meanshift(data_and_info[0][kappa]) for kappa in range(len(image_list))]  
     
     for epsilon in range(len(meanshifted_list)):
         for i in range(len(meanshifted_list[epsilon][0])):
             
             plt.show(block=False)
             fig = plt.figure()
             plt.imshow(meanshifted_list[epsilon][0][i])
             plt.show
             
             print("Quality: ",meanshifted_list[epsilon][3][i])
             
         print("time aquired =",meanshifted_list[epsilon][2])
         
#//////////////////////////////////////////////////////////////         
#Clustering the imaes with Kmeans and printing the images if wanted       
if(show_clustered_images==True):
    
    clustered_list=[kc.Kmeans_filter(data_and_info[0][i],cluster_amount) for i in range(len(image_list))]
    
    laplacian_var_clustered=[]
    
    for i in range(len(clustered_list)):
        
        plt.show(block=False)
        time.sleep(0.5)
        plt.imshow(clustered_list[i][0])
        plt.show
        time.sleep(0.5)
        print("Interdistance/Intradistance = ",clustered_list[i][2])
        print("time aquired =",clustered_list[i][1])
        laplacian_var_clustered.append(arf.variance_of_laplacian(clustered_list[i][0]))

#//////////////////////////////////////////////////////////////
#apply the PCA to the data and printing the results
if(show_pca):
    
    start_pca=time.time()
    pca_image=kc.p_c_a(data,3)
    print("PCA result")
    plt.imshow(pca_image)
    plt.imsave(savedirectory+"/"+"PCA"+".png",pca_image[:,:,0-2])
    pca_time=math.floor(time.time()-start_pca)

#//////////////////////////////////////////////////////////////
#save all images that were printed and write all infos in a log-file
if(save_images==True):
    
    file=open(savedirectory+"/info.txt","w")
    
    file.write("info file of "+filename.split("/")[-1]+"\n")
    file.write("-----------------------------------------"+"\n")
    
    for i in range(len(data_and_info[0])):
        
        file.write(data_and_info[1][i]+"\n")
        
        if(show_filtered_images==True):
            
            plt.imsave(savedirectory+"/"+data_and_info[1][i]+".png",data_and_info[0][i])

            file.write("\t"+"RGB-filtering"+"\n")
            file.write("\t"+"\t"+"--laplacian variance: "+str(laplacian[i])+" e-3"+"\n")
            file.write("\t"+"\t"+"--time needed: "+str(data_and_info[2][i])+" sec"+"\n")
        
        if(show_meanshifted_images==True):
            
            file.write("\t"+"Meanshift"+"\n")
            
            for j in range(len(meanshifted_list[i][0])):
                
                 plt.imsave(savedirectory+"/"+data_and_info[1][i]+meanshifted_list[i][1][j]+".png",meanshifted_list[i][0][j]) #    

                 file.write("\t-"+meanshifted_list[i][1][j]+"\n")
                 file.write("\t"+"\t"+"--Distance-value: "+str(meanshifted_list[i][3][j])+"\n")
            file.write("\t"+"\t"+"--total time needed: "+str(meanshifted_list[epsilon][2])+" sec"+"\n")
        
        
        if(show_clustered_images==True):
            
            plt.imsave(savedirectory+"/"+data_and_info[1][i]+"_clustered.png",clustered_list[i][0])

            file.write("\t"+"Kmeans: "+"k = "+str(cluster_amount)+"\n")
            file.write("\t"+"\t"+"--laplacian variance: "+str(laplacian_var_clustered[i])+" e-3"+"\n")
            file.write("\t"+"\t"+"--Distance-value: "+str(clustered_list[i][2])+"\n")
            file.write("\t"+"\t"+"--time needed: "+str(clustered_list[i][1])+" sec"+"\n")
        
    if(show_pca):
        
        file.write("\t"+"PCA: "+"\n")
        file.write("\t"+"\t"+"--time needed: "+str(pca_time)+" sec"+"\n")
      
    file.write("\n")  
    
    file.write("overall time needed: "+str(math.floor(time.time()-start_d_facto)))
        
    file.close()
       
        

        
        
        