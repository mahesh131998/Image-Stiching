#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def remove_padding(blk_img):
    bw = cv2.cvtColor(blk_img, cv2.COLOR_BGR2GRAY)
    cut = cv2.findNonZero(bw)
    x, y, w, h = cv2.boundingRect(cut)
    rect = blk_img[y:y+h, x:x+w] 
    
    return rect

def stitch_background(img1, img2, savepath=''):
    siftExt = cv2.xfeatures2d.SIFT_create()
    keys1, des1 = siftExt.detectAndCompute(img1, None)
    keys2, des2 = siftExt.detectAndCompute(img2, None)


    print(img1.shape, img2.shape)
    dist = np.zeros((len(des1),len(des2)))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):               
            d =  np.sqrt(np.sum(np.square(np.subtract(des1[i],des2[j]))))
            dist[i][j]=d
    print(dist.shape)


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
        ptsA = np.float32([keys1[i].pt for i in image1_key_index]).reshape(-1, 1, 2)
        ptsB = np.float32([keys2[i].pt for i in image2_key_index]).reshape(-1, 1, 2)
        keyA= [keys1[i] for i in image1_key_index]
        keyB= [keys2[i] for i in image1_key_index]


        

    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,5.0) #,reprojThresh
    
    # print(H)
    result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0] +img1.shape[0]))
    for i in range(0, img2.shape[0]):
        for j in range(0, img2.shape[1]):
            if np.sum(result[i][j]) > 0:
                if np.sum(result[i][j]) > np.sum(img2[i][j]):
                    result[i][j] = result[i][j]
                else:
                    result[i][j] = img2[i][j]
            else:
                result[i][j] = img2[i][j]
    
    # result[0:img2.shape[0], 0:img2.shape[1]] = img2
    result= remove_padding(result)
    cv2.imwrite('task1.png', result)

    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

