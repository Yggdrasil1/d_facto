# -*- coding: utf-8 -*-


from __future__ import print_function
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance 
import time
import math



def Meanshift(data):
    
    start=time.time()
    
    #create a vector from the original data for the clustering 
    position=np.indices((264,264))
    position=np.swapaxes(position,0,2)
    position=np.swapaxes(position,0,1)
    Xpos=position.reshape(264*264,2)/2000
    
    Xrgb=data.reshape(264*264,3)
    
    X=np.concatenate((Xrgb,Xpos),axis=1)
    
    #define the range of the meanshift parameter 'quantile'
    Quant=range(55,2000,500)
    Quant_arr=np.zeros((len(Quant)))

    #create an array to iterate over the different values for quantile    
    for theta in range(len(Quant)):
        
        Quant_arr[theta]=Quant[theta]/10000 
        
    returnlist=[]
    
    msedlist=[]
    cluster_name_list=[]
    cluster_amount_list=[]  
    quality_list=[]
    n_cluster_Arr=np.zeros((len(Quant)))
    
    #cluster the data with meanshift with different quantile values 
    for kappa in range(len(Quant)):
        bandwidth = estimate_bandwidth(X, quantile=Quant_arr[kappa], n_samples=2500)
        ms = MeanShift(bandwidth=bandwidth/1.0, bin_seeding=True)
        ms.fit(X)
        labels = ms.labels_
        centroids=ms.cluster_centers_
        
        interdist=interdistance(centroids)
        
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        
        n_cluster_Arr[kappa]=n_clusters_
                
        temp=labels.reshape(264,264)
        
        intradist=intradistance(data,n_clusters_, centroids,temp)
        
        quality_list.append(interdist/intradist)
        
        Crgb=np.array(Xrgb, copy=True)
        Crgb[:,:]=0
        
        #add additional information to the outputlist
        if not(n_clusters_ in cluster_amount_list):
            msedlist.append(temp)
            cluster_name_list.append(str(n_clusters_)+"_clusters")
            cluster_amount_list.append(n_clusters_)
    
    
    returnlist.append(msedlist)
    returnlist.append(cluster_name_list)
    returnlist.append(math.floor(time.time()-start))    
    returnlist.append(quality_list)

    return returnlist   

def interdistance(centroids):
    #calculate the intercluster dinstance
    
    interdist_list=distance.cdist(centroids,centroids, 'euclidean')
    
    dist_list=[]
    
    for j in range(len(interdist_list)):
        for k in range(len(interdist_list[0])):
            if (interdist_list[j,k]!=0):
                dist_list.append(interdist_list[j,k])
    
    interdist=np.mean(dist_list)
     
    return interdist            


def intradistance(org_data,n_cluster, centroids,cluster):
    #calculate the average intracluster distance
    
    intra_list=[]
    
    for i in range(n_cluster):
        
        sub_intra_list=[]
        
        position=np.where(cluster==i)
        
        if (len(position[0])>0):
            
            for ln in range(len(position[0])):
                sub_intra_list.append(distance_new(org_data[position[0][ln],position[1][ln]],centroids[i]))

            intra_list.append(np.mean(sub_intra_list))
        
    return np.mean(intra_list)
         

def distance_new(a,b):
    #calculating the euclidian metric 
    
    sum_quad=0
    
    for i in range(len(a)):
        
        sum_quad=sum_quad+((a[i]-b[i])**2)
    
    dist=np.sqrt(sum_quad)
    
    return dist        
        
        
        
        
        
        
        