#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:30:50 2017
    
@author: recon2
    
this script is the first attempt to make a single function that return a list with different filtered 
images of the same image with different filters
    
"""
    
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats.stats import pearsonr
from scipy.optimize import minimize
import time
import cv2


   
class RGBfilter:
    
    #Initialisation of all needed class variables, if something isnt set properly, the pre-set values are used
    data_array=np.zeros((264,264,1024))
    data_norm=np.zeros((264,264,1024)) #normalized raw data 
    n_channel=1024
    show_pics=False
    start=0         #start of the smallest area that includes 99% of all counts
    end=1024        #end of the smallest area that includes 99% of all counts
    time_list=[]
    
    def makeRGBfilter(array, Bool,saves,filepath):
        """ 
        Create a list of RGB-filtered images
        
        Keyword arguments:
        
        array -- the n-dimensional data array
        Bool -- boolean value, if true the filters get shown
        saves -- boolean value, if true the images of the filters get saved
        filepath -- filepath to the directory where the images shall be saved to
        
        returns:
            
        List [image, filter names, calculation time]
        
        """
        
        #setting the class variable to the input argument 'array'
        RGBfilter.data_array=array
        
        #setting the class variable to the input argument 'Bool'
        RGBfilter.show_pics=Bool
        
        #the logarithm enhances small peaks compared to otherwise overdominant peaks 
        array=np.log10(array+1)
        
        #normalizes all channels with the highest value in all spectra
        array=array/np.amax(array[:,:])
        
        #overwriting 2 lines with artifacts
        array[132,:,:]=array[133,:,:]
        array[131,:,:]=array[130,:,:]
        
        
        RGBfilter.data_norm=np.copy(array)
        
        #number of channels in the spectrum
        RGBfilter.n_channel=len(array[0,0,:])
        
        #calculate the sumspectrum of all pixels and normalizes it 
        sumspectrum=np.zeros((RGBfilter.n_channel))
        
        for i in range(len(sumspectrum)):
            sumspectrum[i]=np.sum(array[:,:,i]) 
            
        #initializing of the list that gets returned and contains all the filtered images   
        output_list=[]
        
        #selection of filters used for visualisation
        filter_numbers=range(1,2)
        #filter_numbers=[1,2,3,4,5,6,7,10]
        
        #adds different filtered images to the list
        filtered_list=[RGBfilter.rgbnorm(RGBfilter.applyFilter(array,RGBfilter.rgbfilter(i,saves,filepath))) for i in filter_numbers]
       
        #adding up all information to the output list: filtered images, the names of the used filters and the time that the 
        #program needed to create the filter and apply them to the data 
        output_list.append(filtered_list)
        output_list.append(getNames(filter_numbers))
        output_list.append(RGBfilter.time_list)
        
        return output_list
      
        
    
    def rgbfilter(filter_nr,save_filter,savedirectory):  
        """
        Create an RGB filter
        
        Keyword arguments:
        
        filter_nr --
        save_filter --
        savedirectory --
            
        returns:
        
        
        """
        
        
        start=time.time()
        
        #initialising the values for every color over the spectrum
        red = np.zeros((RGBfilter.n_channel)) 
        green=np.zeros((RGBfilter.n_channel))
        blue= np.zeros((RGBfilter.n_channel))
        
        #create an axis to plot the filter
        scale=np.zeros((RGBfilter.n_channel))
        
        for i in range(0,RGBfilter.n_channel):
            scale[i]=i
        
        #calculate the sumspectrum of all pixels and normalizes it 
        sumspectrum=np.zeros((RGBfilter.n_channel))
        
        for i in range(len(sumspectrum)):
            sumspectrum[i]=np.sum(RGBfilter.data_array[:,:,i]) 
        
        cop_array=np.copy(RGBfilter.data_array)
        
        cop_array=np.log10(cop_array+1)
        
        normalized_sum=np.zeros((RGBfilter.n_channel))
        
        for h in range(RGBfilter.n_channel):
            normalized_sum[h]=np.sum(cop_array[:,:,h])
         
        maxvalue=np.amax(normalized_sum)
        
        for i in range(len(normalized_sum)):
            normalized_sum[i]=normalized_sum[i]/maxvalue        
        
    ############ Creation of the filters ############
    
    ##################### 1 #########################
                #Human Eye Filter
        
        if(filter_nr==1):
        #filter according to the human eye
        
            #defining the areas for the red, green and blue filter
            pos1=int(np.round(RGBfilter.n_channel*0.166))
            pos2=int(np.round(RGBfilter.n_channel*0.333))
            pos3=int(np.round(RGBfilter.n_channel*0.5))
            pos4=int(np.round(RGBfilter.n_channel*0.666))
            pos5=int(np.round(RGBfilter.n_channel*0.833))
            delta=(RGBfilter.n_channel-pos5)*2
            
            #define the colorvalues in each area
            for wavelength in range(0,pos1):
                red[wavelength]= -(wavelength-pos1)/pos1
                green[wavelength]=0
                blue[wavelength]=1.0
                
            for wavelength in range(pos1,pos2):
                red[wavelength]   = 0.0
                green[wavelength]= (wavelength - pos1) / (pos2 - pos1)
                blue[wavelength] = 1.0
                    
            for wavelength in range(pos2,pos3):
                red[wavelength]   = 0.0
                green[wavelength]= 1.0
                blue[wavelength] = -(wavelength - pos3) / (pos3 - pos2)
                        
            for wavelength in range(pos3,pos4):
                red[wavelength]   = (wavelength - pos3) / (pos4 - pos3)
                green[wavelength]= 1.0
                blue[wavelength] = 0
            
            for wavelength in range(pos4,pos5):
                red[wavelength]   = 1.0
                green[wavelength]= -(wavelength - pos5) / (pos5 - pos4)
                blue[wavelength] = 0.0
            
            for wavelength in range(pos5,RGBfilter.n_channel):
                red[wavelength]  = -(wavelength-(RGBfilter.n_channel+delta))/((RGBfilter.n_channel+delta)-pos5)
                green[wavelength]= 0
                blue[wavelength] = 0 
         
            filter=np.matrix((blue,green,red)).transpose()
         
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                #plt.plot(scale,blue,'r')
                #plt.plot(scale,green,'g')
                #plt.plot(scale,red,'b')
                #plt.xlabel('Channel [#]')
                #plt.ylabel('Intensity [I] arb. units')
                #plt.title('Human Eye Filter')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/spectrum_scan.png',dpi=600)            
    
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter
       
    ##################### 2 #########################    
            #Human Eye Filter scaled
        if(filter_nr==2):
            
            #finds the first channel with value greater than 10 
            lowerbarrier=0
            for i in range(RGBfilter.n_channel):
                if sumspectrum[i]>500:
                    lowerbarrier=i
                    break
        
            upperbarrier=RGBfilter.n_channel
            for i in range(RGBfilter.n_channel-1,-1,-1): 
                if sumspectrum[i]>250:
                    upperbarrier=i
                    break
        
            #setting the borders for each colors areas
            area=upperbarrier-lowerbarrier
        
            pos1=int(np.round(area*0.166)+lowerbarrier)
            pos2=int(np.round(area*0.333)+lowerbarrier)
            pos3=int(np.round(area*0.5)+lowerbarrier)
            pos4=int(np.round(area*0.666)+lowerbarrier)
            pos5=int(np.round(area*0.833)+lowerbarrier)
            delta=(RGBfilter.n_channel-pos5)*2
        
            #define the colorvalues in each area
            for wavelength in range(0,pos1):
                red[wavelength]= -(wavelength-pos1)/pos1
                green[wavelength]=0
                blue[wavelength]=1.0
        
            for wavelength in range(pos1,pos2):
                red[wavelength]   = 0.0
                green[wavelength]= (wavelength - pos1) / (pos2 - pos1)
                blue[wavelength] = 1.0
            
            for wavelength in range(pos2,pos3):
                red[wavelength]   = 0.0
                green[wavelength]= 1.0
                blue[wavelength] = -(wavelength - pos3) / (pos3 - pos2)
                
            for wavelength in range(pos3,pos4):
                red[wavelength]   = (wavelength - pos3) / (pos4 - pos3)
                green[wavelength]= 1.0
                blue[wavelength] = 0
            
            for wavelength in range(pos4,pos5):
                red[wavelength]   = 1.0
                green[wavelength]= -(wavelength - pos5) / (pos5 - pos4)
                blue[wavelength] = 0.0
            
            for wavelength in range(pos5,RGBfilter.n_channel):
                red[wavelength]  = -(wavelength-(RGBfilter.n_channel+delta))/((RGBfilter.n_channel+delta)-pos5)
                green[wavelength]= 0
                blue[wavelength] = 0
        
            filter=np.matrix((blue,green,red)).transpose()
         
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'r')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'b')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Human Eye Filter scaled')
                
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/human_eye_scaled.png',dpi=600)
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter
                
    ##################### 3 #########################
                #Gaussian Filter simple        
    
        if(filter_nr==3):
            
            #calculates the gaussian function
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
        
            red=RGBfilter.gaussian(red,200,200)#200,200)
            green=RGBfilter.gaussian(green,500,200)#500,200)
            blue=RGBfilter.gaussian(blue,800,200)#800,200)
            
            #combines the colorfunctions to a 3x1024 matrix and transposes it to 1024x3
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian Filter simple')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_Filter_simple.png',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter
            
    ##################### 4 #########################
             #Gaussian equal counts distribution
                
        if(filter_nr==4):
            
            #gives the value that includes 1/4 of all photon counts
            sum_third=np.sum(sumspectrum)/4
        
            #calculates the 3 points that divides the spectrum in such a way that all
            #3 areas include aproximately 1/3 of all photoncounts
            sum=0
            x1=0    #first third
            x2=0    #second third
            x3=0    #third third
        
            for i in range(len(sumspectrum)):
                sum=sum+sumspectrum[i]
                if sum<sum_third and sum>x1:
                    x1=i
                if sum<2*sum_third and sum>x2:
                    x2=i
                if sum<3*sum_third and sum>x3:
                    x3=i
            
            #calculates the standard deviation for the gaussean function
            std1=(x2-x1)/2
        
            std2=(x3-x2)/2
        
            if (x2-x1)>(x3-x2):
                std2=(x2-x1)/2
        
            std3=(x3-x2)/2
        
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            #print(x1,x2,x3,std1,std2,std3)
        
            red=RGBfilter.gaussian(red,x1,std1)
            green=RGBfilter.gaussian(green,x2,std2)
            blue=RGBfilter.gaussian(blue,x3,std3)
        
            #combines the colorfunctions to a 3x1024 matrix and transposes it to 1024x3
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian equal counts distribution')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_equal_counts_distribution.png',dpi=600)
                
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter 
            
    ##################### 5 #########################
                #Gaussian three highest maxima
    
        if(filter_nr==5):        
            
            #look for the 3 greatest maxima in the sumspectrum
            maxima=[]
        
            for i in range(len(sumspectrum)-1):
                if(i>0 and i<len(sumspectrum)-1 and sumspectrum[i]>sumspectrum[i-1] and sumspectrum[i]>sumspectrum[i+1]):
                    maxima.append((sumspectrum[i],i))
            
            y1=max(maxima)      #first maximum
        
            maxima.remove(y1)
        
            y2=max(maxima)      #second maximum, atleast 50 channel apart from the first
            maxima.remove(y2)
            
            while (math.fabs(y2[1]-y1[1])<50):
                y2=max(maxima)
                maxima.remove(y2)
        
            y3=max(maxima)      #third maximum, atleast 50 channel apart from the first and second
            maxima.remove(y3)
        
            while (math.fabs(y3[1]-y1[1])<50 or math.fabs(y3[1]-y2[1])<50):
                y3=max(maxima)
                maxima.remove(y3)
            
            #sorts the maxima so they are in the following order: red, green, blue
            if (y1[1]>y3[1]):
                buffer1=y1
                y1=y3
                y3=buffer1
            
            if (y1[1]>y2[1]):
                buffer1=y1
                y1=y2
                y2=buffer1
            
            if (y2[1]>y3[1]):
                buffer1=y2
                y2=y3
                y3=buffer1
                
            #calculates the standard deviation for the gaussean function
            std1=(y2[1]-y1[1])/2
        
            std2=(y3[1]-y2[1])/2
        
            if (y2[1]-y1[1])<(y3[1]-y2[1]):
                std2=(y2[1]-y1[1])/2
        
            std3=(y3[1]-y2[1])/2
            
            #calculates the gaussian function
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            red=RGBfilter.gaussian(red,y1[1],std1)
            green=RGBfilter.gaussian(green,y2[1],std2)
            blue=RGBfilter.gaussian(blue,y3[1],std3)
            
            #combines the colorfunctions to a 3x1024 matrix and transposes it to 1024x3
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                dumspectrum=sumspectrum/np.amax(sumspectrum)
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian three highest maxima')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_three_highest_maxima.png',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter
            
    ##################### 6 #########################
                #Gaussian Filter smallest area
    
        if(filter_nr==6):
            
            threshold=0.97*np.sum(sumspectrum)
            
            area=RGBfilter.n_channel
            
            area_start=3
            
            for j in range(len(sumspectrum)):
                counts=0
                test_area=0
                for i in range(j,len(sumspectrum)):
                    counts=counts+sumspectrum[i]
                    test_area=i-j
                    if(counts>=threshold and test_area<area):

                        area=test_area
                        area_start=j
                        break
                    
            x=area/4
            x1=area_start+x
            x2=area_start+2*x
            x3=area_start+3*x
            
            x4=(x2-x1)/3.
            x5=(x2-x1)/3
            x6=(x3-x2)/3
            
            #calculates the gaussian function
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
            
            red=RGBfilter.gaussian(red,x1,x4)
            green=RGBfilter.gaussian(green,x2,x5)
            blue=RGBfilter.gaussian(blue,x3,x6)
            
            #combines the colorfunctions to a 3x1024 matrix and transposes it to 1024x3
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                dumspectrum=sumspectrum/np.amax(sumspectrum)
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian smallest area')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_Filter_smallest_area.png',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter       
    
    ##################### 7 #########################
            #Gaussian Filter over standard deviation

        if(filter_nr==7):
           
            sig_array=np.copy(RGBfilter.data_array)
        
            #the logarithm enhances small peaks compared to otherwise overdominant peaks 
            sig_array=np.log10(sig_array+1)
            
            #normalizes all channels with the highest value in all spectra
            sig_array=sig_array/np.amax(sig_array[:,:])
            std_array=np.zeros((RGBfilter.n_channel))
              
            for i in range(RGBfilter.n_channel):
                     if(np.std(sig_array[:,:,i])!=0):
                         std_array[i]=np.std(sig_array[:,:,i],ddof=1)/(np.sqrt(np.mean(sig_array[:,:,i])))
               
            #look for the 3 greatest maxima in the sumspectrum
            maxima=[]
        
            for i in range(len(std_array)):
                if(i>0 and i<1023 and std_array[i]>std_array[i-1] and std_array[i]>std_array[i+1]):
                    maxima.append((std_array[i],i))
            
            y1=max(maxima)      #first maximum
        
            maxima.remove(y1)
        
            y2=max(maxima)      #second maximum, atleast 50 channel apart from the first
            maxima.remove(y2)
            
            while (math.fabs(y2[1]-y1[1])<50):
                y2=max(maxima)
                maxima.remove(y2)
        
            y3=max(maxima)      #third maximum, atleast 50 channel apart from the first and second
            maxima.remove(y3)
        
            while (math.fabs(y3[1]-y1[1])<50 or math.fabs(y3[1]-y2[1])<50):
                y3=max(maxima)
                maxima.remove(y3)
            
            #sorts the maxima so they are in the following order: red, green, blue
            if (y1[1]>y3[1]):
                buffer1=y1
                y1=y3
                y3=buffer1
            
            if (y1[1]>y2[1]):
                buffer1=y1
                y1=y2
                y2=buffer1
            
            if (y2[1]>y3[1]):
                buffer1=y2
                y2=y3
                y3=buffer1
                
            #calculates the standard deviation for the gaussean function
            std1=(y2[1]-y1[1])/2
        
            std2=(y3[1]-y2[1])/2
        
            if (y2[1]-y1[1])<(y3[1]-y2[1]):
                std2=(y2[1]-y1[1])/2
        
            std3=(y3[1]-y2[1])/2
            
            #calculates the gaussian function
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            red=RGBfilter.gaussian(red,y1[1],std1)
            green=RGBfilter.gaussian(green,y2[1],std2)
            blue=RGBfilter.gaussian(blue,y3[1],std3)
            
            filter=np.matrix((red,green,blue)).transpose()
            
            for i in range(len(std_array)):
                std_array[i]=std_array[i]/np.amax(std_array)
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,std_array)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian Filter over standard deviation')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_Filter_standard_derivation.png',dpi=600)
                
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter    
    ##################### 8 #########################
            #Gaussian over reduced area
    
        if(filter_nr==8):   
            
            #finds first and last channel that have counts
            xmin=0
            xmax=0
            for i in range(len(sumspectrum)):
                if(sumspectrum[i]>np.amax(sumspectrum)/1000):
                    xmin=i
                    break
            
            for i in range(len(sumspectrum)-1,0,-1):
                if(sumspectrum[i]>np.amax(sumspectrum)/250):
                    xmax=i
                    break
       
           
            #length of the interval with counts > 0
            interval=xmax-xmin
        
        
            #determine the 4 points for the meanvalues of the geaussean functions
            ivalthird=interval/4
        
            x1=ivalthird      #first third
            x2=2*ivalthird    #second third
            x3=3*ivalthird    #third third
            
            #calculates the standard deviation for the gaussean function
            std1=(x2-x1)/2
        
            std2=(x3-x2)/2
        
            if (x2-x1)>(x3-x2):
                std2=(x2-x1)/2
        
            std3=(x3-x2)/2
        
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            #print(x1,x2,x3,std1,std2,std3)
            
            red=RGBfilter.gaussian(red,x1+xmin,std1)
            green=RGBfilter.gaussian(green,x2+xmin,std2)
            blue=RGBfilter.gaussian(blue,x3+xmin,std3)
            
            #print(x1+xmin,x2+xmin,x3+xmin, "filter 6")
            
            #combines the colorfunctions to a 3x1024 matrix and transposes it to 1024x3
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian over reduced area')
                #plt.title('Normalized sum spectrum of the ammonite sample')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_over_reduced_area.png',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter       

            
    ##################### 9 #########################
            #Gaussian minimal correlation

        if(filter_nr==9):
            
            #get the lowest and highest channel with a minimum amount of counts
            getborders()
            
            #print(RGBfilter.start,RGBfilter.end)
            
            #initial values for th minimization
            startvalues=RGBfilter.significance_signal()
            #print(startvalues)
            var=minimize(RGBfilter.fun ,(startvalues),method='nelder-mead')
            
            #print('OPTIMIZED')
            #print(var.x)
            
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            x1=var.x[0]
            x2=var.x[1]
            x3=var.x[2]
            x4=(x2-x1)/2
            x5=(x2-x1)/2
            x6=(x3-x2)/2
                
            red=RGBfilter.gaussian(red,var.x[0],x4/2.35)
            green=RGBfilter.gaussian(green,var.x[1],x5/2.35)
            blue=RGBfilter.gaussian(blue,var.x[2],x6/2.35)
            
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Gaussian minimal correlation')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_minimal_correlation_ammonit.png',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter    
            
######################### 10 #########################
            #Gaussian green complement

        if(filter_nr==10):
            
            getborders()
            
            #initial values for minimization
            startvalues=RGBfilter.significance_signal()
            
            #reduce the startvalues to red and blue
            startvalues2=[startvalues[0],startvalues[2]]
            
            #print(startvalues2)
            
            var2=minimize(RGBfilter.fun_simple ,(startvalues2),method='nelder-mead')
            
            for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
                
            x1=var2.x[0]
            x2=var2.x[1]
            
            x4=x1/3
            x5=(x2-x1)/3
            
            red=RGBfilter.gaussian(red,var2.x[0],x4)
            blue=RGBfilter.gaussian(blue,var2.x[1],x5)
            
            green=(1-(red+blue)/2)
            
            filter=np.matrix((red,green,blue)).transpose()
            
            if(RGBfilter.show_pics==True):
                
                fig = plt.figure()
                plt.plot(scale,normalized_sum)
                plt.plot(scale,blue,'b')
                plt.plot(scale,green,'g')
                plt.plot(scale,red,'r')
                plt.xlabel('Channel [#]')
                plt.ylabel('Intensity [I] arb. units')
                plt.title('Green complement Filter')
                
                if(save_filter==True):
                    fig.savefig(savedirectory+'/Gaussian_green_complement_mineral',dpi=600)
        
                plt.show()
            
            RGBfilter.time_list.append(math.floor(time.time()-start))
            
            return filter 
        
        
        
    def applyFilter(rawdata,rgbfilter):
        """
        Apply a given filter to a given image and return the filtered image
        
        
        
        """
        #creates the array that gets added to the list, filled with zeros, later get filled with filtered data
        output_array=np.zeros((len(rawdata),len(rawdata[0]),3))
        
        #applies the filter on every pixel and saves the RGB values in the output_array
        for i in range(len(rawdata)):
            for j in range(len(rawdata[i])):
                
                buffer_matrix=np.dot(np.matrix(rawdata[i,j]),rgbfilter)
                
                for k in range(0,3):
                    
                    output_array[i,j,k]=buffer_matrix[0,k]
                    
        
        return output_array
    
    def gaussian(x, mu, sig):
        #calculates the value for a given x on a gaussian curve with mean mu and 
        #standard deviation sig
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    def rgbnorm(temp):
        #normalizes each channel of the given RGB-image, so that every value is between 0 and 1
        
        if (temp[:,:,0].max()!=0):
            temp[:,:,0]=temp[:,:,0]/temp[:,:,0].max()
        if (temp[:,:,1].max()!=0):
            temp[:,:,1]=temp[:,:,1]/temp[:,:,1].max()
        if (temp[:,:,2].max()!=0):
            temp[:,:,2]=temp[:,:,2]/temp[:,:,2].max()
        return temp
        
        
    def fun(XX): 
                
        XX[2]=1000-abs(XX[2]-1000)
              
        x1=clamp(XX[0],RGBfilter.start,RGBfilter.start+(RGBfilter.end-RGBfilter.start)/2)
        x2=clamp(XX[1],RGBfilter.start+(RGBfilter.end-RGBfilter.start)/4,RGBfilter.end-(RGBfilter.end-RGBfilter.start)/4)
        x3=clamp(XX[2],RGBfilter.start+(RGBfilter.end-RGBfilter.start)/2,RGBfilter.end)
  
        x4=x1/5
        x5=(x2-x1)/5
        x6=(x3-x2)/6
        
        #initialising the values for every color over the spectrum
        red = np.zeros((RGBfilter.n_channel)) 
        green=np.zeros((RGBfilter.n_channel))
        blue=np.zeros((RGBfilter.n_channel))
        
        for i in range(0,RGBfilter.n_channel):
                red[i]=i
                green[i]=i
                blue[i]=i
            
        red=RGBfilter.gaussian(red,x1,x4)
        green=RGBfilter.gaussian(green,x2,x5)
        blue=RGBfilter.gaussian(blue,x3,x6)
        
        normalized_array=np.copy(RGBfilter.data_array)
        
        #the logarithm enhances small peaks compared to otherwise overdominant peaks 
        normalized_array=np.log10(normalized_array+1)
        
        #normalizes all channels with the highest value in all spectra
        normalized_array=normalized_array/np.amax(normalized_array[:,:])
            
        filter1=np.matrix((red,green,blue)).transpose()
         
        img2=RGBfilter.rgbnorm(RGBfilter.applyFilter(normalized_array,filter1))
        
        #calculating the correlation values for red=pr1, green=pr2 and blue=pr3
        pr1=pearsonr(img2[:,:,0].reshape(264*264),img2[:,:,1].reshape(264*264))[0]
        
        pr2=pearsonr(img2[:,:,0].reshape(264*264),img2[:,:,2].reshape(264*264))[0]
        
        pr3=pearsonr(img2[:,:,1].reshape(264*264),img2[:,:,2].reshape(264*264))[0]
    
        #the sum of all 3 colours shall be minimized
        sum=float(pr1)+float(pr2)+float(pr3)
        
        return sum
    
    def fun_simple(YY):
        
        x1=clamp(YY[0],RGBfilter.start,RGBfilter.start+(RGBfilter.end-RGBfilter.start)/2)
        x3=clamp(YY[1],RGBfilter.start+(RGBfilter.end-RGBfilter.start)/2,RGBfilter.end)
        
        x4=x1/3
        x6=(x3-x1)/5
        
        #initialising the values for every color over the spectrum
        red = np.zeros((RGBfilter.n_channel)) 
        green=np.zeros((RGBfilter.n_channel))
        blue=np.zeros((RGBfilter.n_channel))
        
        for i in range(0,RGBfilter.n_channel):
            
                red[i]=i 
                blue[i]=i
            
        red=RGBfilter.gaussian(red,x1,x4/2.35)
        
        blue=RGBfilter.gaussian(blue,x3,x6/2.35)
        
        normalized_array=np.copy(RGBfilter.data_array)
        
        #the logarithm enhances small peaks compared to otherwise overdominant peaks 
        normalized_array=np.log10(normalized_array+1)
        
        #normalizes all channels with the highest value in all spectra
        normalized_array=normalized_array/np.amax(normalized_array[:,:])
            
        filter1=np.matrix((red,green,blue)).transpose()
         
        img2=RGBfilter.rgbnorm(RGBfilter.applyFilter(normalized_array,filter1))
        
        #calculating the correlation values for red=pr1, green=pr2 and blue=pr3
        pr2=pearsonr(img2[:,:,0].reshape(264*264),img2[:,:,1].reshape(264*264))[0]
        
        #the sum of all 3 colours shall be minimized
        sum=float(pr2)

        return sum
    
    def significance_signal():
        #returns the 20%, 50% and 75% mark of the significance signal of the data 
        
        sig_array=np.copy(RGBfilter.data_array)
        
        #the logarithm enhances small peaks compared to otherwise overdominant peaks 
        sig_array=np.log10(sig_array+1)
        
        #normalizes all channels with the highest value in all spectra
        sig_array=sig_array/np.amax(sig_array[:,:])
         
        std_array=np.zeros((RGBfilter.n_channel))
          
        for i in range(RGBfilter.n_channel):
            if(np.std(sig_array[:,:,i])!=0):
                std_array[i]=np.std(sig_array[:,:,i])/(1/np.sqrt(np.mean(sig_array[:,:,i])))
            if i>0:
                std_array[i]=std_array[i]+std_array[i-1]
            
        std_array=std_array/np.amax(std_array)
        
        #create an axis to plot the filter
        scale=np.zeros((RGBfilter.n_channel))
        
        for i in range(0,RGBfilter.n_channel):
            scale[i]=i
            
        #plt.plot(scale, std_array)
        #plt.show()
        
        x1=0
        x2=0
        x3=0
        
        for i in range(len(std_array)):
            if(std_array[i]>0.2 and x1==0):
                x1=i
            if(std_array[i]>0.5 and x2==0):
                x2=i
            if(std_array[i]>0.75 and x3==0):
                x3=i
        
        return [x1,x2,x3]
        
def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n        
        
def getborders():
    
    #calculate the sumspectrum of all pixels and normalizes it 
    sumspectrum=np.zeros((RGBfilter.n_channel))
        
    for i in range(len(sumspectrum)):
        sumspectrum[i]=np.sum(RGBfilter.data_array[:,:,i])
    
    #finds the smallest area that includes 95% of all counts
    threshold=0.99*np.sum(sumspectrum[:])
    sum=0
    arealen=RGBfilter.n_channel
    areastart=0
    
    for i in range(len(sumspectrum)):
        for j in range(len(sumspectrum)-i):
            sum+=sumspectrum[i+j]
            if(sum>threshold and j<arealen):
                areastart=i
                arealen=j
                
        sum=0
    RGBfilter.start=areastart
    RGBfilter.end=areastart+arealen    
        
def getNames(numbers):
    #returns the names of the used filters
    
    names=["01_Simple_Human_Eye_filter",
           "02_Human_Eye_filter_scaled",
           "03_Gaussian_filter_simple",
           "04_Gaussian_equal_counts_distribution",
           "05_Gaussian_three_highest_maxima",
           "06_Simple_gaussean_smallest_area",
           "07_Gaussian_over_standard_deviation",
           "08_Gaussian_over_reduced_area",
           "09_Gaussian_minimal_correlation",
           "10_Gaussian_green_complement"]
    
    returnlist=[]
       
    for k in numbers:
        returnlist.append(names[k-1])
    
    return returnlist

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()*1000









    
        
        
        
        
        