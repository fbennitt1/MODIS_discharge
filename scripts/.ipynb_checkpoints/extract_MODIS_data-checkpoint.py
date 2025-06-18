# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:04:18 2025

@author: p.filippucci
"""

# -*- coding: utf-8 -*-
"""
@author: p.filippu
"""


import ee
import numpy as np
from datetime import datetime,timedelta
import time
from scipy.io import savemat
from scipy.io import loadmat as loadmat2
from mat73 import loadmat
import h5py
import hdf5storage
                
# %% Input
tic=time.time()

service_account = 'riverdischarge@riverdischarge1.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '/home/pfilippucci/Desktop/River_Discharge/GEE/riverdischarge1-35925bb5bef6.json')
ee.Initialize(credentials)
folderGEE='projects/riverdischarge1/assets/CCI'; #e.g.: projects/riverdischarge/assets/research
user='paolo81990';
folder_pc='F:/z_scambio_files/IRPI CNR Dropbox/BACKUP/CURRENT_USERS/p.filippucci/angelica/CCI_discharge'


products=['MOD','MYD']
ray=0.075
ray0="0075"

# insert below station name, central area coordinates and river coordinates
file=loadmat('CCI_observed_data_2ndphase.mat')
lonlat=file['lonlat_ok']
name=file['name']
name_2=file['name_2']
basin=file['basin']
river=file['river']


for i,j in enumerate(name):
    print(str(i)+' '+basin[i][0]+' '+j[0])


for stat in range(23,28):#,len(name)):
    
    str2=name_2[stat][0]
    print(str(stat)+' '+basin[stat][0]+' '+name[stat][0])
    coordstat=lonlat[stat].tolist() #Central area coordinates. NB coordstat should not be taken inside river since snow period selection is calculated there
    years=[2000,2024] #Calibration period (has to be < 5000 days)
        
    #region=ee.Geometry.Point(coordstat).buffer(ray*100000).bounds()
    region=ee.Geometry.Rectangle([coordstat[0]-ray,coordstat[1]-ray,coordstat[0]+ray,coordstat[1]+ray])
    
    for prod in products:
        product1='MODIS/061/'+prod+'09GQ';
        product2='MODIS/061/'+prod+'09GA';

        datestartCAL=ee.Date(str(years[0])+'-01-01').millis();
        dateendCAL=ee.Date(str(years[-1])+'-12-31').millis();
        
        band=['sur_refl_b02','sur_refl_b01'];
        
        coll=ee.ImageCollection(product1).filterDate(datestartCAL,dateendCAL).filterBounds(region).select(band);
        proj=coll.first().projection();
        scale_mod=ee.Number(proj.nominalScale()).getInfo()
        trans_mod=proj.getInfo()['transform'];
        crs_mod=proj.getInfo()['crs'];

        maskband='state_1km';

        coll2=ee.ImageCollection(product2).filterDate(datestartCAL,dateendCAL).filterBounds(region).select(maskband);
        coll=coll.combine(coll2)
        n=coll.size().getInfo()
        print('dimension premasking:'+str(n))
        
        def calc_Date(image,llist):
            value =image.get('system:time_start')
            value=ee.Number(ee.Date(ee.List([value, -9999]).reduce(ee.Reducer.firstNonNull())).millis());
            return ee.List(llist).add(value)

        def selection(image):
            date =image.get('system:time_start')
            im=image.select(maskband).reproject(crs_mod,trans_mod).bitwiseAnd((46851))
            image=image.select(band).updateMask(im.gt(0).And(im.neq(3)).And(im.neq(2)).Not())
            image=image.updateMask(image.gt(0))
            
            value=image.select(band[0]).mask().reduceRegion(**{'reducer':ee.Reducer.mean(),'geometry':region,'crs':crs_mod,'crsTransform':trans_mod,'bestEffort':True})
            value=ee.Number(value.get(value.keys().get(0)))
            return image.set('Valid_perc',value).set('system:time_start',date)
        
        coll0=coll.map(selection)
        coll0=coll0.filterMetadata('Valid_perc',"not_less_than",0.20)
        first=ee.List([])
        dlistn=ee.List(coll0.iterate(calc_Date,first)).distinct().getInfo()
        dlistn=np.sort(dlistn).tolist()

        def selimage(nday):
            image=ee.ImageCollection(product1).filterDate(ee.Date(nday),ee.Date(ee.Number(nday).add(86400000))).filterBounds(region).select(band).first()#.reduce(ee.Reducer.mean()).rename(band[0]).reproject(crs_mod,trans_mod)
            image2=ee.ImageCollection(product2).filterDate(ee.Date(nday),ee.Date(ee.Number(nday).add(86400000))).filterBounds(region).select(maskband).first()#.reduce(ee.Reducer.mean()).rename(maskband).reproject(crs_mod,trans_mod)
            image2=image2.bitwiseAnd((46851)).reproject(crs_mod,trans_mod)
            
            return image.updateMask(image2.gt(0).And(image2.neq(3)).Not())
            

        coll=ee.ImageCollection(ee.List(dlistn).map(selimage))

        n=coll.size().getInfo()
        
        print('dimension postmasking:'+str(n))
        

        MOD0 = np.array(ee.ImageCollection(coll.first()).select(band).getRegion(region, scale_mod).getInfo())
        #MOD=np.zeros([np.shape(MOD0)[0]-1,6,n])
        MOD=np.zeros([np.shape(MOD0)[0]-1,5,n])
        for i in range(n):
            print('reading image '+str(i)+' of '+str(n))
            tic=time.time()
            ee.Initialize(credentials)

            nn=0
            while nn<3:
                try:
                    nn+=1
                    coll0=ee.ImageCollection(ee.Image(ee.List(coll.toList(n)).get(i)))
                    arr=np.array(coll0.getRegion(region, scale_mod).getInfo())
                    arr=arr[1:,1:].astype(float)                
                    MOD[:,:,i]=arr
                finally:
                    nn=4
            toc=time.time()
            print(toc-tic)
            
        D=MOD[0,2,:]
        llonlat=MOD[:,0:2,0]
        surf2=MOD[:,3,:]
        surf1=MOD[:,4,:]
        #cloud=MOD[:,5,:]
        matfiledata = {}
        matfiledata['surf2'] = surf2
        matfiledata['surf1'] = surf1
        #matfiledata['cloud'] = cloud
        matfiledata['D'] = D
        matfiledata['lonlat'] = llonlat
        savemat(prod+'_'+str2+'_data.mat', matfiledata)
        #hdf5storage.write(matfiledata,'.',prod_name+'_'+str2+'_data.mat', matlab_compatible=True)
