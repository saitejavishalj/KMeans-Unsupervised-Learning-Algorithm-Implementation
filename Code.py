# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:17:47 2021

@author: Sai Teja Vishal J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

data= scipy.io.loadmat('AllSamples.mat', squeeze_me=True);
valX = data['AllSamples'];
valX = pd.DataFrame(valX)
valX.rename(columns = {0: "pt0", 1:"pt1"}, inplace = True)

##Plot the cluster for Strategy 1 and Strategy 2, and clsuter centers is denoted by "X" mark.
def plot_clusters(init_cetroids,valX,shades, title_):
    for pointer1 ,j7 in init_cetroids.iterrows():
            params_x =[]
            params_y =[]
            for pointer2,j8 in valX.iterrows():
                if j8["Cluster"]==pointer1 :
                    params_x.append(j8['pt0'])
                    params_y .append(j8['pt1'])
            plt.scatter(params_x,params_y ,c=shades[pointer1 ],marker='o')
            plt.scatter(j7['pt0'],j7['pt1'],c='black',marker='X')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(title_)

##Calculate Euclidian Distance using the below function in KMeans Implementation
def  euclidean_dist_calc(init_cetroids,dist,valX,p):
    for pointer1 ,j3 in init_cetroids.iterrows():
            euclid_dist=[]
            for pointer2,j4 in dist.iterrows():
                par1=(j3['pt0']-j4['pt0'])**2
                par2=(j3['pt1']-j4['pt1'])**2
                param=np.sqrt(par1+par2)
                euclid_dist.append(param)
            valX[p]=euclid_dist
            p=p+1
    return valX,p

##Form clusters from the datapoints is done using the below function, grouping the datapoints is done in this function.
def form_clusters(valX,p,clusters):
    for pointer,j in valX.iterrows():
        dist_minimum=j[0]
        location=0
        for i in range(p):
            if j[i] < dist_minimum:
                dist_minimum = j[i]
                location=i
        clusters.append(location)
    return clusters

##This function below will tell the difference between the initially considered centroilds and the final obtained centroids.
def calc_difference_centroids(updated_centroids,init_cetroids,valX,change,valN,p):
    for pointer1 ,j5 in updated_centroids.iterrows():
            res_diff=[]
            for pointer2,j6 in valX.iterrows():
                if j6["Cluster"]==pointer1 :
                    par1=(j5['pt0']-j6['pt0'])**2
                    par2=(j5['pt1']-j6['pt1'])**2
                    res_diff.insert(p,par1+par2)
    if valN == 0:
        change=1
        valN=valN+1
    else:
        change = (updated_centroids['pt0'] - init_cetroids['pt0']).sum() + (updated_centroids['pt1'] - init_cetroids['pt1']).sum()
    return res_diff,change,valN;

##KMeans algorithm is implemented in this function, this function is used by Strategy 1 and Strategy 2 implementations.
def implement_kMeans(title_, change,valX,init_cetroids,p,valN,centroids_number,points):
    while(change!=0):
        dist=valX
        pt=0
        valX,pt= euclidean_dist_calc(init_cetroids,dist,valX,pt)
        clusters=[]
        clusters=form_clusters(valX,p,clusters)
        valX["Cluster"]=clusters
        updated_centroids = valX.groupby(["Cluster"]).mean()[['pt0','pt1']]
        res_diff,change,valN = calc_difference_centroids(updated_centroids,init_cetroids,valX,change,valN,p)
        init_cetroids = updated_centroids
    shades=['palegreen','orange','violet','cyan','rosybrown','chocolate','yellow','pink','wheat','grey','c']
    plot_clusters(init_cetroids,valX,shades, title_)
    plt.show()
    points.append(sum(res_diff))
    centroids_number.append(p)
    return centroids_number,points;

##Strategy 1 for KMeans Implementation is done in this function, this function uses Implement_KMeans function to perform Strategy 1
##Strategy 1 is performed as per the instructions in the Canvas.
def strategy1_kMeans(valX,ranges):
    title1 = "Plot of Cluster in Strategy 1"
    centroids_number=[]
    points=[]
    for p in ranges: 
        init_cetroids = (valX.sample(n=p, random_state = np.random.RandomState()))
        print(init_cetroids)
        change = 1
        valN=0    
        centroids_number,points=implement_kMeans(title1, change,valX,init_cetroids,p,valN,centroids_number,points)
    plt.plot(centroids_number,points)
    plt.xlabel("Size of the Cluster")
    plt.ylabel("Calculated Loss (Objective Function)")
    plt.title("Strategy 1 :Cluster Size VS Loss")
    plt.show()

##Strategy 2 for KMeans Implementation is done in this function, this function uses Implement_KMeans and getCentroidsList functions to perform Strategy 2
##Strategy 2 is performed as per the instructions in the Canvas.    
def strategy2_kMeans(valX,ranges):
    title2 = "Plot of Cluster in Strategy 2"
    centroids_number=[]
    points=[]
    for p in ranges: 
        init_cetroids = getCentroidsList(valX,p)
        change = 1
        valN=0    
        centroids_number,points=implement_kMeans(title2, change,valX,init_cetroids,p,valN,centroids_number,points)
    plt.plot(centroids_number,points)
    plt.xlabel("Size of the Cluster")
    plt.ylabel("Calculated Loss (Objective Function)")
    plt.title("Strategy 2 :Cluster Size VS Loss")
    plt.show()

##Centroids for stratgy 2, for i>1 is calculated in the below function. For i=1, centroids are considered randomly, a list is generated.
def getCentroidsList(valX,p):
    cent1 = (valX.sample(n=1, random_state = np.random.RandomState()))
    list=pd.DataFrame()
    list=list.append(cent1,ignore_index = True)
    for dummy in range(p-1):
        maximum=0.0
        for pointer3,j11 in valX.iterrows():
            dist_sum=0.0
            for pointer4,j16 in list.iterrows():
                math_dist=np.sqrt((j11['pt0']-j16['pt0'])**2+(j11['pt1']-j16['pt1'])**2) 
                dist_sum=dist_sum+math_dist
            avg = dist_sum/len(list)
            if  avg>maximum:
                maximum=avg
                param=pointer3
        list=list.append(valX.iloc[param,:],ignore_index = True)
    return list;
    
##Cluster ranges from 2 to 10
ranges = [2,3,4,5,6,7,8,9,10];

##Scatter plots are plotted in the below lines, initial plotting is done below:
plt.scatter(valX["pt0"],valX['pt1'], color='red',marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Initial Plotting of the Dataset")
plt.show()


##Triggering two stratergies (Strategy 1 and Strategy 2) with the range values and dataset is performed below
for pt in range(1,3):
    strategy1_pts=valX
    strategy2_pts=valX
    strategy1_kMeans(strategy1_pts,ranges)
    strategy2_kMeans(strategy2_pts,ranges)