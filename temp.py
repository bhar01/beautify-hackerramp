import pandas as pd
import numpy as np
import colour
import ast 
df=pd.read_csv(r'/Users/aishwarya/Downloads/lipshades.csv')
colors=df['Color']
names=df['Name']
images=df['Image']
undertone=[]
warm =0
cool =0
neutral=0
rgb=[]
for c in colors:
    res = ast.literal_eval(c)
   
    b=res[0][0]/255
    g=res[0][1]/255
    r=res[0][2]/255
    rgb.append([res[0][0],res[0][1],res[0][2]])
    RGB = np.array([r,g,b])
    XYZ = colour.sRGB_to_XYZ(RGB)
    xy = colour.XYZ_to_xy(XYZ)
    CCT = colour.xy_to_CCT(xy, 'hernandez1999')
    if CCT <= 2700:
        undertone.append("warm")
        warm+=1
    elif CCT>2700 and CCT<=4000:
        undertone.append("neutral")
        neutral+=1
    else:
        undertone.append("cool")
        cool+=1
df1= pd.DataFrame()
df1['Name']= names
df1['Color']=colors 
df1['Image']=images
df1['RGB']=rgb 
df1['Undertone']= undertone
df1.to_csv('lipstick.csv', index=False)
print(warm, cool, neutral)
