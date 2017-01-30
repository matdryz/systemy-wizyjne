import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats
coefficients = [0.114, 0.587, 0.299]
m = np.array(coefficients).reshape((1,3))
def localize_plate(path,INTER_CHARACTER_DISTANCE, MINWCHARS,MAXWCHARS, wP, MINHCHAR, MAXHCHAR, hP, a1, a2, a3, a4, a5, debug=False):
    img = cv2.imread(path)
    gray_scale= cv2.transform(img, m)
    # sobelx = cv2.Sobel(gray_scale,cv2.CV_32F,1,0,ksize=-1)
    sobelx = cv2.Sobel(gray_scale,cv2.CV_8U,1,0,ksize=-1)
    sobelx=cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('temp.jpg',sobelx)
    # sobelx=cv2.imread('temp.jpg',)
    if debug:
        cv2.imwrite('steps/sobel.jpg',sobelx)
    mean = cv2.blur(sobelx, (int(a1*wP), int(a1*hP)))
    if debug:
        cv2.imwrite('steps/mean.jpg',mean)
    opening = cv2.morphologyEx(mean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1, MINHCHAR)))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(INTER_CHARACTER_DISTANCE, 1)))
    if debug:
        cv2.imwrite('steps/opening-closing.jpg',closing)
    topHap = cv2.morphologyEx(closing, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(1, MAXHCHAR)))
    topHapClosing = cv2.morphologyEx(topHap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(INTER_CHARACTER_DISTANCE, 1)))
    if debug:
        cv2.imwrite('steps/topHap-c.jpg',topHapClosing)
    erosion = cv2.morphologyEx(topHap, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(int(a2*wP), 1)))
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(int(a3*wP), 1)))
    if debug:
        cv2.imwrite('steps/erosion-dilation.jpg',dilation)
    asGray=cv2.cvtColor(dilation, cv2.COLOR_RGB2GRAY)
      
    ret2,th2 = cv2.threshold(asGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholded=stats.threshold(th2, threshmin=0, threshmax=1, newval=255)
    if debug:
        cv2.imwrite('steps/binary.jpg',th2  )
    
    r1=np.copy(thresholded)
    contuoursImg, contours,hierarchy = cv2.findContours(thresholded,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    candidates=[]
    img_height=img.shape[0]
    img_width=img.shape[1]
    
    
    candidate_dilation_in_pixels=2
    # print(len(contours))
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print(x,y,w,h)
        if h>=w or w<MINWCHARS or x<candidate_dilation_in_pixels or y<candidate_dilation_in_pixels or x+w>img_width-candidate_dilation_in_pixels or y+h>img_height-candidate_dilation_in_pixels  or h<MINHCHAR:
            continue
        candidates.append(cv2.boundingRect(cnt))
    result=[]
    ind=0
    # print(len(candidates))
    for candidate in candidates:
        # print("c:" )
        # print(candidate)
        candidate_img=asGray[(candidate[1]-candidate_dilation_in_pixels):(candidate[1]+candidate[3]+candidate_dilation_in_pixels),(candidate[0]-candidate_dilation_in_pixels):(candidate[0]+candidate[2]+candidate_dilation_in_pixels)]
        # print(candidate_img.shape)
        # candidate_dilation = cv2.morphologyEx(candidate_img, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)))
        candidate_ret,candidate_binarized = cv2.threshold(candidate_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        candidate_erosion=cv2.morphologyEx(candidate_binarized, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(int(a4*wP), 1)))
        candidate_dilation_2=cv2.morphologyEx(candidate_erosion, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT,(int(a5*wP), int(a5*hP))))
        ind=ind+1
        candidate_ret2,candidate_th2 = cv2.threshold(candidate_dilation_2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        candidate_thresholded=stats.threshold(candidate_th2, threshmin=0, threshmax=1, newval=255)
        if debug:
            cv2.imwrite("steps/cand_org_"+str(ind)+".jpg",img[(candidate[1]-candidate_dilation_in_pixels):(candidate[1]+candidate[3]+candidate_dilation_in_pixels),(candidate[0]-candidate_dilation_in_pixels):(candidate[0]+candidate[2]+candidate_dilation_in_pixels)])
            cv2.imwrite("steps/cand_"+str(ind)+".jpg", candidate_img)
            cv2.imwrite("steps/cand"+str(ind)+".jpg", candidate_thresholded)
        contuoursImg, final_contours,hierarchy = cv2.findContours(candidate_thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE )
        # print(np.amax(candidate_binarized))
        # print(cv2.boundingRect(candidate_binarized) )
        
        for cnt in final_contours:
            x,y,w,h=cv2.boundingRect(cnt)
            if w>MINWCHARS and h>MINHCHAR and w<MAXWCHARS and h<MAXHCHAR:
                result.append((candidate[0]+x,candidate[1]+y,w,h))
            if debug and x!=0 and y!=0:
                print("from")
                print(candidate)
                print("to")
                print((x,y,w,h))

    return result
def area(a, b):  # returns 0 if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

def score(filePath, expected, INTER_CHARACTER_DISTANCE, MINWCHARS,MAXWCHARS, wP, MINHCHAR, MAXHCHAR,hP, a1, a2, a3, a4, a5):
    candidates=localize_plate(filePath,INTER_CHARACTER_DISTANCE, MINWCHARS,MAXWCHARS, wP, MINHCHAR, MAXHCHAR,hP, a1, a2, a3, a4, a5)
    if(len(candidates)==0):
        return (0,0)
    x,y,w,h =candidates[0]
    expected=(expected)
    result=area((x,y,x+w,y+h), expected)/((w)*(h))
    
    for r in candidates[1:]:
        x,y,w,h =r
        score=area((x,y,x+w,y+h), expected)/((expected[2]-expected[0])*(expected[3]-expected[1]))
        result=max(result, score)
    return (result, len(candidates))