import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
import pandas as pd

pro = pd.read_csv("/Users/aishwarya/PycharmProjects/flaskProject3/product.csv")
sh = pd.read_csv("/Users/aishwarya/PycharmProjects/flaskProject3/shades.csv")
col = pd.read_csv("/Users/aishwarya/PycharmProjects/flaskProject3/colors.csv")

def extractFace(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
    roi_color= face
    for (x, y, w, h) in faces:
        cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = face[y:y + h, x:x + w]
        faced = "/Users/aishwarya/PycharmProjects/flaskProject3/"+ str(w) + str(h) + "_faces.jpg"
        cv2.imwrite(faced, roi_color)
    return(faced)


def extractSkin(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):

    hasBlack = False
    occurance_counter = Counter(estimator_labels)
    def compare(x, y): return Counter(x) == Counter(y)
    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]
        if compare(color, [0, 0, 0]) == True:
            del occurance_counter[x[0]]
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break
    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    occurance_counter = None
    colorInformation = []
    hasBlack = False
    if hasThresholding == True:
        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)
    totalOccurance = sum(occurance_counter.values())
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))
        index = (index-1) if ((hasThresholding & hasBlack)& (int(index) != 0)) else index
        color = estimator_cluster[index].tolist()
        color_percentage = (x[1]/totalOccurance)
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}
        colorInformation.append(colorInfo)
    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):
    if hasThresholding == True:
        number_of_colors += 1
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1]), 3)
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)
    estimator.fit(img)
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def getcol(x):
    comp = {}
    append = {}
    for i in x:
        comp[i['cluster_index']] = i['color_percentage']
        append[i['cluster_index']] = i['color']
    m = sum(comp) / len(comp)
    col = []
    col.append(append[m])
    return col


def colorDist(image):
    r = image[0][0]
    b = image[0][0]
    g = image[0][0]
    ret = []
    l = col['Color']
    for i in l:
        j = i.strip("[,]")
        rr, bb, gg = j.split(",")
        ro = (abs(r - float(rr))) ** 2
        bo = (abs(b - float(bb))) ** 2
        go = (abs(g - float(gg))) ** 2
        ret.append(ro + bo + go)
    return ret


def getUndertone(i):
    r = i[0][0]
    b = i[0][1]
    g = i[0][2]
    if r > g and r > b:
        undertone = "warm"

    elif b > r and b > g:
        undertone = "cool"

    else:
        undertone = "neutral"
    return undertone


def suggest(ret,n):
    nm = {}
    col = pd.read_csv("/Users/aishwarya/PycharmProjects/flaskProject3/colors.csv")
    for i in range(n):
        a = min(ret)
        b = ret.index(a)
        print(b)
        ind=len(col['Name'][b])
        nm[col['Image'][b]] =col['Name'][b][0:ind-4]
        print(col['Name'][b])
        ret.remove(a)
    return nm


def pipeline(image):
    skin = extractSkin(image)
    x = extractDominantColor(skin, hasThresholding=True)
    y = getcol(x)
    z = colorDist(y)
    op = suggest(z, 3)
    return op


def pipeline2(image):
    skin = extractSkin(image)
    x = extractDominantColor(skin, hasThresholding=True)
    y = getcol(x)
    z = getUndertone(y)
    return z

def pipeline3(image):
    skin = extractSkin(image)
    x = extractDominantColor(skin, hasThresholding=True)
    y = getcol(x)
    r = y[0][0]
    b = y[0][1]
    g = y[0][2]
    print(type(r))
    return [b,g,r]



