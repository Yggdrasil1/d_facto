#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import time
from matplotlib import pyplot as plt
import math
from scipy.cluster.vq import whiten, kmeans2
from scipy.spatial import distance 
from sklearn.decomposition import PCA


def Kmeans_filter(data,K_cluster):
    
    start=time.time() 
    
    #reshape the data to single vector
    clustervector=data.reshape(69696,3)
    
    #apply a whiten function to the data
    whitenarray=whiten(clustervector)
    
    #cluster the data with kmeans
    clustering=kmeans2(whitenarray,K_cluster,iter=40)
    
    clusterarray=clustering[1]
    
    #creating an array for the recoloring of the data
    Crgb=np.zeros((264,264,3))
    
    #reshape the data to an image again
    clusterdata=clusterarray.reshape(264,264)
    
    #calculating the intercluster distance
    interdist=interdistance(clustering[0])
    
    #calculating the intracluster distance
    intradist=intradistance(data,K_cluster,clustering[0],clusterdata)
    
    value=interdist/intradist
    
    #recolor the pixels of the clustered data to the mean color of the RGB data
    for i in range(0,K_cluster):
        
        pos=np.where(clusterdata == i) 
        
        if(len(pos[0])>0):
            
            for k in range(len(data[0,0])):
                
                sum=0
            
                for j in range(len(pos[0])):
            
                    sum=sum+(data[pos[0][j],pos[1][j],k])
                
                mean=sum/len(pos[0])
                                 
                for l in range(len(pos[0])):
                    
                    Crgb[pos[0][l],pos[1][l],k]=mean    
    
    #normalize the data to values between 0 and 1
    Crgb=rgbnorm(Crgb)
       
    #calculate the time passed since this script started
    timey=math.floor(time.time()-start) 
    
    #add all return values to a list and return it
    returnlist=[]
    returnlist.append(Crgb)
    returnlist.append(timey)
    returnlist.append(value)

    return returnlist    
    
def rgbnorm(temp):
        #normalizes each channel of the given RGB-image, so that every value is between 0 and 1
        
        if (temp[:,:,0].max()!=0):
            temp[:,:,0]=temp[:,:,0]/temp[:,:,0].max()
        if (temp[:,:,1].max()!=0):
            temp[:,:,1]=temp[:,:,1]/temp[:,:,1].max()
        if (temp[:,:,2].max()!=0):
            temp[:,:,2]=temp[:,:,2]/temp[:,:,2].max()
        return temp
    
def interdistance(centroids):
    #calculates the intracluster distcance between all centroids
    
    interdist_list=distance.cdist(centroids,centroids, 'euclidean')
    
    dist_list=[]
    
    for j in range(len(interdist_list)):
        for k in range(len(interdist_list[0])):
            if (interdist_list[j,k]!=0):
                dist_list.append(interdist_list[j,k])
    
    interdist=np.mean(dist_list)
     
    return interdist


def intradistance(clusterarray,n_cluster, centroids,cluster):
    #calculating the intradistance
    
    intra_list=[]
    
    for i in range(n_cluster):
        
        sub_intra_list=[]
        
        position=np.where(cluster==i)
        
        if (len(position[0])>0):
            
            for ln in range(len(position[0])):
                sub_intra_list.append(distance_new(clusterarray[position[0][ln],position[1][ln]],centroids[i]))

            intra_list.append(np.mean(sub_intra_list))
        
    return np.mean(intra_list)
 
       
def distance_new(a,b):
    #calculates the euclidian distance between two points a and b
    sum_quad=0
    
    for i in range(len(a)):
        
        sum_quad=sum_quad+((a[i]-b[i])**2)
    
    dist=np.sqrt(sum_quad)
    
    return dist

def p_c_a(data,cluster):
    #evaluating the data with the Principal Component Analysis
    
    data1=np.log10(data+1)
    
    vec=data1.reshape(69696,1024)
    
    pca=PCA(n_components=cluster, whiten=False)
    pca.fit(vec,cluster)
    fitted=pca.transform(vec)
    
    fitted=fitted.reshape(264,264,cluster)
    
    minimum=np.amin(fitted)
    
    fitted=fitted+abs(minimum)
    fitted=fitted/np.amax(fitted)
    
    return fitted

    
    
    











