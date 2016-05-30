import neurolab as nl
import numpy as np
import matplotlib.pylab as pl
import ast



g=open('input1.txt','r')
line=g.readline()
x=line[1:-2]
x = ast.literal_eval(x)
input_house_sf=list(x)
line=g.readline()
x=line[1:-2]
x = ast.literal_eval(x)
input_lot_sf=list(x)
line=g.readline()
x=line[1:-2]
x = ast.literal_eval(x)
input_bedrooms=list(x)
line=g.readline()
x=line[1:-2]
x = ast.literal_eval(x)
input_bathrooms=list(x)
line=g.readline()
x=line[1:-2]
x = ast.literal_eval(x)
output1=list(x)
g.close()

##1st INPUT


input_house_sf_normalized=input_house_sf/np.linalg.norm(input_house_sf)
house_sf_max=np.max(input_house_sf)
house_sf_max_normalized=np.max(input_house_sf_normalized)

house_sf_min=np.min(input_house_sf)
house_sf_min_normalized=np.min(input_house_sf_normalized)

newhouse=np.linspace(house_sf_min,house_sf_max,18)


## 2ND INPUT


input_lot_sf_normalized=input_lot_sf/np.linalg.norm(input_lot_sf)
lot_sf_max=np.max(input_lot_sf)
lot_sf_max_normalized=np.max(input_lot_sf_normalized)

lot_sf_min=np.min(input_lot_sf)
lot_sf_min_normalized=np.min(input_lot_sf_normalized)

newlot=np.linspace(lot_sf_min,lot_sf_max,18)


## 3RD INPUT


input_bedrooms_normalized=input_bedrooms/np.linalg.norm(input_bedrooms)
bedrooms_max=np.max(input_bedrooms)
bedrooms_max_normalized=np.max(input_bedrooms_normalized)

bedrooms_min=np.min(input_bedrooms)
bedrooms_min_normalized=np.min(input_bedrooms_normalized)

newbedroom=np.linspace(bedrooms_min,bedrooms_max,18)


## 4TH INPUT


input_bathrooms_normalized=input_bathrooms/np.linalg.norm(input_bathrooms)
bathrooms_max=np.max(input_bathrooms)
bathrooms_max_normalized=np.max(input_bathrooms_normalized)

bathrooms_min=np.min(input_bathrooms)
bathrooms_min_normalized=np.min(input_bathrooms_normalized)

newbathroom=np.linspace(bathrooms_min,bathrooms_max,18)
## adjusting the inputs 
number_of_samples=len(input_house_sf)
input_data_1=(newhouse).reshape(number_of_samples,1)
input_data_2=(newlot).reshape(number_of_samples,1)
input_data=np.append(input_data_1,input_data_2,axis=1)

input_data_3=(newbedroom).reshape(number_of_samples,1)
input_data=np.append(input_data,input_data_3,axis=1)

input_data_4=(newbathroom).reshape(number_of_samples,1)
input_data=np.append(input_data,input_data_4,axis=1)
input_data_normalized=input_data/np.linalg.norm(input_data)

## constructing and adjusting the output

output=output1/np.linalg.norm(output1)
output_data=(np.array(output)).reshape(number_of_samples,1)

y=((newhouse*0.367934509)+(newlot*0.015595172)
+(newbedroom*0.119113849)+(newbathroom*input_bathrooms))
+(-1.218314642)


y=y/np.max(y)
output_labels=y.reshape(number_of_samples,1)

## mulitlayer network 


multilayer_net=nl.net.newff([[house_sf_max,house_sf_min],[lot_sf_max,lot_sf_min],[bedrooms_max,bedrooms_min],[bathrooms_max,bathrooms_min]],[10,10,10,10,10,10,10,10,10,10,10,1])

multilayer_net.trainf=nl.train.train_gd
error=multilayer_net.train(input_data,output_labels,epochs=1000,show=100,goal=0.01)

## simulated network and plotting


the_output=multilayer_net.sim(input_data)
y3=the_output.reshape(number_of_samples)

outputmin=np.min(output1)
outputmax=np.max(output1)
outputgraph=np.linspace(outputmin,outputmax,18)
outputgraphx=np.linspace(0,1,18)

x2=input_data_normalized
y2=multilayer_net.sim(x2)



pl.subplot(321)
pl.plot(input_data_1,y, '-',input_data_1,y3,'p')
pl.ylim(0,1.5)
pl.xlim(0,9)
pl.legend(['train target', 'network output'],loc='best')
pl.xlabel('House Square Feet in Thousands')
pl.ylabel('Normalized Price')

pl.subplot(322)
pl.plot(input_data_2,y, '-',input_data_2,y3,'p')
pl.ylim(0,1.5)
pl.xlim(0,50)
pl.legend(['train target', 'network output'],loc='best')
pl.xlabel('Lot Square Feet in Thousands')
pl.ylabel('Normalized Price')

pl.subplot(323)
pl.plot(input_data_3,y, '-',input_data_3,y3,'p')
pl.ylim(0,1.5)
pl.xlim(0,9)
pl.legend(['train target', 'network output'],loc='best')
pl.xlabel('Number of Rooms')
pl.ylabel('Normalized Price')

pl.subplot(324)
pl.plot(input_data_4,y, '-',input_data_4,y3, 'p')
pl.ylim(0,1.5)
pl.xlim(0,9)
pl.xlabel('Number of Bathrooms')
pl.ylabel('Normalized Price')
pl.legend(['train target', 'network output'],loc='best')

pl.subplot(325)
pl.plot(error)
pl.ylim(0,0.20)
pl.xlabel('Epoch number')
pl.ylabel('error')

pl.subplot(326)
pl.plot(outputgraphx,outputgraph)
pl.xlabel('Normalization')
pl.ylabel('Real Price')

pl.show()




