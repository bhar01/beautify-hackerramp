import pandas as pd
import math
import ast
def calc_dist(lipstick, skin):
    res = ast.literal_eval(lipstick)
    dist=math.sqrt((res[0]-skin[0])**2+ (res[1]-skin[1])**2+ (res[2]-skin[2])**2)
    return dist

def most_similar(id,rgb, skin):
    dist_dict={}
    for i in id:
        dist_dict[i]=calc_dist(rgb[i], skin)
    dist_dict= {k: v for k, v in sorted(dist_dict.items(), key=lambda item: item[1])}
    l=list(dist_dict.keys())
    return l[-4:]


def lipstick_undertone(u, undertone):
    id=[]
    for i in range (0,len(u)):
        if undertone==u[i]:
            id.append(i)
    return id

def lipstick_reco(skin,undertone):
    df = pd.read_csv("/Users/aishwarya/PycharmProjects/flaskProject3/lipstickshades.csv")
    bgr = df['RGB']
    names=df['Name']
    images=df['Image']


    u = df['Undertone']
    res_dict={}
    id=lipstick_undertone(u, undertone)
    res=most_similar(id, bgr, skin)
    for i in res:
        res_dict[names[i]]= images[i]

    return res_dict





