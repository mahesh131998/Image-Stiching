# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import json
from unittest import result
import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay(img1,img2):
    siftExt = cv2.xfeatures2d.SIFT_create()
    keys1, des1 = siftExt.detectAndCompute(img1, None)
    keys2, des2 = siftExt.detectAndCompute(img2, None)
    dist = np.zeros((len(des1),len(des2)))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):               
            d = np.sqrt( np.sum(np.square(np.subtract(des1[i],des2[j]))))
            dist[i][j]=d
    ratio_index={}
    for i in range(des1.shape[0]):
        fetch_dist = np.argsort(dist[i]) #fetches index
        best_1,best_2 = fetch_dist[0],fetch_dist[1]
        if dist[i][best_1] / dist[i][best_2 ] < 0.6:
            ratio_index.update({i:best_1})
    image1_key_index=[]
    image2_key_index=[]

    for key, values in ratio_index.items():
        image1_key_index.append(key)
        image2_key_index.append(values)

    m1= len(image1_key_index)
    m2= len(image2_key_index)
    k1= len(keys1)
    k2=len(keys2)

    f1=int((m1/k1)*100)
    f2=int((m2/k2)*100)
    
    if f1 >20 or f2>20:
        return 1
    else:
        return 0

def remove_padding(blk_img):
    bw = cv2.cvtColor(blk_img, cv2.COLOR_BGR2GRAY)
    cut = cv2.findNonZero(bw)
    x, y, w, h = cv2.boundingRect(cut)
    final = blk_img[y:y+h, x:x+w] 
    
    return final

def getPoints(img1, img2):
    siftExt = cv2.xfeatures2d.SIFT_create()
    keys1, des1 = siftExt.detectAndCompute(img1, None)
    keys2, des2 = siftExt.detectAndCompute(img2, None)
    dist = np.zeros((len(des1),len(des2)))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):               
            d = np.sqrt( np.sum(np.square(np.subtract(des1[i],des2[j]))))
            dist[i][j]=d
    ratio_index={}
    for i in range(des1.shape[0]):
        fetch_dist = np.argsort(dist[i]) #fetches index
        best_1,best_2 = fetch_dist[0],fetch_dist[1]
        if dist[i][best_1] / dist[i][best_2 ] < 0.6:
            ratio_index.update({i:best_1})
    image1_key_index=[]
    image2_key_index=[]

    for key, values in ratio_index.items():
        image1_key_index.append(key)
        image2_key_index.append(values)
    
    for i in range (len(image1_key_index)):
        ptsA = np.float32([keys1[i].pt for i in image1_key_index]).reshape(-1,1,2)
        ptsB = np.float32([keys2[i].pt for i in image2_key_index]).reshape(-1,1,2)

    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,5.0)
    if img1.shape[0]> img2.shape[0]:
        res= img1.shape[0]
    else:
        res=img2.shape[0]
    img12 = cv2.warpPerspective(img2, H,(img1.shape[1] + 2*img2.shape[1], (2*res)))
    img12[0:img1.shape[0], 0:img1.shape[1]] = img1
    img12=remove_padding(img12)
    return img12



def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    #########################################################################################################
    imgs3= []
    imgs4=[]
    if imgmark =='t3':
        for im in imgs:
            pot= cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgs3.append(pot)
        imgs4=imgs3

        overlap_arr2= np.zeros((len(imgs4),len(imgs4)))
        for i in range(len(imgs4)):
            for j in range(i,len(imgs4)):
                if overlay(imgs4[i],imgs4[j])==1:
                    overlap_arr2[i][j]=1
                    overlap_arr2[j][i]=1
                else:
                    overlap_arr2[i][j]=0
                    overlap_arr2[j][i]=0
        


        n= len(imgs3)-1
        img1= imgs3[n]      
        i=n-1
        while i >=0:
            img2= imgs3[i]
            img1=getPoints(img2, img1)
            i= i-1  
        img1=cv2.rotate(img1, cv2.cv2.ROTATE_90_CLOCKWISE)       
        cv2.imwrite('task3.png', img1)
        return overlap_arr2

        
        



    ###########################################################################################################
    ###########################################################################################################
    imgs2= imgs
    overlap_arr= np.zeros((len(imgs),len(imgs)))
    for i in range(len(imgs)):
        for j in range(i,len(imgs)):
            if overlay(imgs2[i],imgs2[j])==1:
                overlap_arr[i][j]=1
                overlap_arr[j][i]=1
            else:
                overlap_arr[i][j]=0
                overlap_arr[j][i]=0


    
    n= len(imgs)-1
    imgs1= imgs 
    img1= imgs[n]      
    i=n-1
    while i >=0:
        img2= imgs1[i]
        if overlay(img1,img2)==1:
            img1=getPoints(img2, img1)
            i= i-1
            
        else:
            top= imgs1.pop(i+1)
            imgs1[:0] = [top]
            img1= imgs1[n]
    cv2.imwrite('task2.png', img1)
    #################################################################################################

    


    

    





    return overlap_arr
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3',N=4, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
