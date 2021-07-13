# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:17:47 2021

@author: Sai Teja Vishal J
"""

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