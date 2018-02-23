"""
Created on January 2018
@author: Freddy Santiago Duarte Prieto
"""

import csv # Write results to a csv file
import datetime as dt # Work with datetime format
import math # Mathematic operations
import matplotlib.pyplot as plt # 2D plotting library
import netCDF4 as nc # Work with NetCDF format files
import nolds # Nonlinear measures for dynamical systems
import numpy as np # Array processing for numbers, strings, records, and objects
import os # use OS functions
import random # Work with random numbers
import scipy # Scientific Library for Python
import sklearn.metrics # Machine Learning in Python
import sklearn.neighbors # Machine Learning in Python
from operator import sub # Standard operators as functions
from toolz import curry # List processing tools and functional utilities

def chaos_analysis(x):   
    # Chaos analyis of the time series
    # Use Eckmann et al algorithm for lyapunov exponents and FNN for embedding dimension
    # Returns the time delay, the embedding dimension and the lyapunov spectrum
    # Returns caos=1 if existe deterministic chaos otherwise caos=0
    lag=delay(x)
    mmax=2*int(np.floor(2*math.log10(len(x))))+1 
    fnn=global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=mmax)            
    if len(fnn[1][fnn[1]<=0.15])!=0:
        m=np.where(fnn[1]<=0.15)[0][0]+1
        lyapunov=nolds.lyap_e(x,emb_dim=2*(m)-1,matrix_dim=m,tau=lag)
        if sum(lyapunov)<0 and max(lyapunov)>0:
            caos=1
        else:
            caos=0 
    else:
        caos=0
        m=99
        lyapunov=99
    return lag,m,lyapunov,caos

def delay(x):
    # Returns the optimal Time-Delay from a time series
    # Use the autocorrelation function and the mutual information score
    da=0;dm=0
    n = len(x)
    y = x-x.mean()
    r = np.correlate(y, y, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(n)]))
    auto = r/(x.var()*(np.arange(n, 0, -1)))
    while (auto[da]*auto[da+1])>0:
        da=da+1
    da=da+1    
    while sklearn.metrics.mutual_info_score(None,None,contingency=np.histogram2d(x, np.roll(x,dm),20)[0])>=sklearn.metrics.mutual_info_score(None,None, contingency=np.histogram2d(x, np.roll(x,dm+1),20)[0]):
        dm=dm+1
    lag=min(da,dm)+1
    return lag

def prom_data(x,step):
    # Return the means values of a time series intervals given by a step
    k=0; b=0;
    data=np.ones(int(np.floor((len(x)-step)/step)+1))*np.nan
    while k<= (len(x)-step):
                data[b]=np.nanmean(x[k:k+step])                
                b=b+1; k=k+step
    return data            
    
def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=15, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = _vector(x)
    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts


def _gfnn(x, lag, n_dims, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    # Global false nearest neighbors at a particular dimension.
    # Returns percent of all nearest neighbors that are still neighbors when the next dimension is unfolded.
    # Neighbors that can't be embedded due to lack of data are not counted in the denominator.
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])
      
@curry
def _is_true_neighbor(x, attractor_radius, offset, indices, distance,relative_distance_cutoff=15, relative_radius_cutoff=2):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distance_increase = np.abs(sub(*x[indices + offset])) 
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)
    
def _nearest_neighbors(y):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]

def _radius(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    return np.sqrt(((x - x.mean())**2).mean())

def reconstruct(x, lag, n_dims):
    # create the delayed vector from a time serie
    x = _vector(x)
    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()

def deconstruct(x, lag, n_dims):
    # create the time serie from a delayed vector
    dec=np.empty(len(x)+lag*(n_dims-1))* np.nan
    dec[:len(x)]=x[:,0]
    dec[len(x):]=x[-lag*(n_dims-1):,-1]
    return dec

def _vector(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x    

def deprom_data(x,step):
    # Return the interval values of a mean values time serie for a certain step
    k=0; b=0;
    data=np.ones((len(x)*step))*np.nan
    serie=np.ones((step))*np.nan
    if min(x)==0:
        min_val=0
    else:
        min_val=np.mean(x)-3*np.std(x)
    max_val=np.mean(x)+3*np.std(x)
    while k<=len(x)-1:
        for i in range(0,step):
            serie[i]=random.uniform(min_val,max_val) # random()
        data[b:b+step]=x[k]*step*serie/sum(serie)
        b=b+step
        k=k+1
    return data

def anually(data_var,data_time,data_type):
    # Accumulate data anually
    p=0
    if data_time[p].month!=1 and data_time[p].day!=1:
        n_years=data_time[-1].year-data_time[0].year+1
    else:
        p = (dt.date(data_time[p].year+1, 1, 1)-data_time[p]).days
        n_years=data_time[-1].year-data_time[0].year
    if data_time[-1].month!=12 and data_time[p].day!=31:
        n_years= n_years-1   
    anual_value=np.empty((n_years))* np.nan
    anual_date=np.empty((n_years))* np.nan   
    for ye in range (0,n_years):
        if (data_time[p].year % 4 == 0 and data_time[p].year % 100 != 0 or data_time[p].year % 400 == 0):
            if data_type=='pr':
                factor=1
            else:
                factor=366
            anual_value[ye]=sum(data_var[p:p+366])/factor
            anual_date[ye]=data_time[p].year
            p=p+366
        else:
            if data_type=='pr':
                factor=1
            else:
                factor=365
            anual_value[ye]=sum(data_var[p:p+365])/factor
            anual_date[ye]=data_time[p].year
            p=p+365
    return anual_date,anual_value

def monthly(data_var,data_time,data_type):
    # Accumulate data monthly
    p=0
    if data_time[p].month!=1 and data_time[p].day!=1:
        n_years=data_time[-1].year-data_time[0].year+1
    else:
        p = (dt.date(data_time[p].year+1, 1, 1)-data_time[p]).days
        n_years=data_time[-1].year-data_time[0].year
    if data_time[-1].month!=12 and data_time[p].day!=31:
        n_years= n_years-1       
    month_days=[[31,28,31,30,31,30,31,31,30,31,30,31],[31,29,31,30,31,30,31,31,30,31,30,31]]
    month_mean=np.empty((12))*0
    for mo in range (0,n_years):
        month_value=np.empty((12,1))*np.nan
        if (data_time[p].year % 4 == 0 and data_time[p].year % 100 != 0 or data_time[p].year % 400 == 0):
            for pp in range(0,12):
                if data_type=='pr':
                    factor=1
                else:
                    factor=month_days[1][pp]
                month_value[pp]=sum(data_var[p:p+month_days[1][pp]])/factor
                p=p+month_days[1][pp]
                month_mean[pp]=month_mean[pp]+month_value[pp]
        else:          
            for pp in range(0,12):
                if data_type=='pr':
                    factor=1
                else:
                    factor=month_days[0][pp]
                month_value[pp]=sum(data_var[p:p+month_days[0][pp]])/factor
                p=p+month_days[0][pp] 
                month_mean[pp]=month_mean[pp]+month_value[pp]
        if mo==0:
            month_date=month_value
        else:
            month_date=np.vstack([month_date,month_value])   
    month_mean=month_mean/n_years
    return  month_mean, month_date

def yn_obj(yn_val,*args):
    # Objetive function for yn
    sta_sync1,ynr1,distanceA,indA,dist_gcm_gcm,s_max=args
    distanceB=np.sqrt((yn_val-sta_sync1)**2+ynr1)
    dist_sta_sta=min(distanceB)
    indB=np.where(distanceB==dist_sta_sta)
    if (distanceB[indA][0]*distanceA[indB][0])/(dist_sta_sta*dist_gcm_gcm)<=s_max:
        return distanceB[indA][0]+distanceA[indB][0]+dist_sta_sta
    else:
        return (distanceB[indA][0]+distanceA[indB][0]+dist_sta_sta)**3

import warnings
warnings.filterwarnings("ignore")

def chaotic_statistcal_downscaling(station_data,station_catalog,model_historic,model_rcp,output_folder,int_acum,cal_date,val_date,app_date):
    # station_data: csv file with the climatic data of the stations
    # station_catalog: csv file with latitude and longitude of the stations
    # model_historic:  folder with the nc files with the historic experiment of the GCM
    # model_rcp: folder with the nc files with the RCP experiment of the GCM
    # output_folder: folder to save the results and plots
    # int_acum: vector with the time steps to evaluate
    # cal_date: last date of the calibration process
    # val_date: last date of the validation process
    # app_date: last date of the application process
    print "CSD: CHAOTIC STATISTICAL DOWNSCALING"
    print "Importing CSV Data"
    station=np.genfromtxt(station_data,delimiter=',') 
    location=np.genfromtxt(station_catalog,delimiter=',')
    if any(location[1:,2]<0):
        location[1:,2]=location[1:,2]+360
    sta_lat=location[1:,1]
    sta_lon=location[1:,2]
    time_sta=[ dt.date(int(station[t,0]), int(station[t,1]),int( station[t,2])) for t in range(1,len(station))]
    ns=len(station[0,:])
    with open(station_data, 'rb') as f:
        reader = csv.reader(f, delimiter=',')
        station_names=next(reader)
    f.close()
    print "Loading NetCDF Data: Historic Data "
    time_his=[];time_rcp=[];    
    for filename in os.listdir(model_historic):
        his = nc.Dataset(model_historic+filename)
        if filename==os.listdir(model_historic)[0]:
            # Variable type and units data
            time_unit=his.variables['time'].units 
            time_cal=his.variables['time'].calendar
            var_type=his.variables.keys()[-1]
            # Grid Data
            gcm_lat=his.variables['lat'][:]
            gcm_lon=his.variables['lon'][:]
        time_valh=his.variables['time'][:]-0.5 # The code works for daily values only right now
        time_his=np.concatenate((time_his,nc.num2date(time_valh,units=time_unit,calendar=time_cal)))    
    # Units Conversion
    if var_type=='pr':
        factor1=86400 # Conversion from kg/m^2/s to mm/day 
        factor2=0
        units='(mm)'
    else: # Temperature: tas, tas_min, tas_max
        factor1=1
        factor2=-273.15 # Conversion from °K to ° C
        units='(grados C)'     
    print "Loading NetCDF Data: RCP Data "
    for filename in os.listdir(model_rcp):
        rcp = nc.Dataset(model_rcp+filename)  
        if filename==os.listdir(model_rcp)[0]:
            rcp_model=rcp.model_id
            rcp_esc=rcp.experiment_id
            time_unit1=rcp.variables['time'].units 
            time_cal1=rcp.variables['time'].calendar
        rcp = nc.Dataset(model_rcp+filename)
        time_valf=rcp.variables['time'][:]-0.5 # The code works for daily values only right now  
        time_rcp=np.concatenate((time_rcp,nc.num2date(time_valf,units=time_unit1,calendar=time_cal1)))
    time_gcm=np.concatenate((time_his, time_rcp))
    print "Get GCM cell and model information"
    gcm_cell=np.empty((len(location)-1,3))* np.nan 
    model=[None]*(len(location)-1)
    n=0
    for z in range(3,ns):
        cell_lat=1;cell_lon=1;
        # Get stations location in cell grids 
        while cell_lat<=len(gcm_lat):
            if (gcm_lat[cell_lat-1]+gcm_lat[cell_lat])/2<=sta_lat[z-3] and sta_lat[z-3]<=(gcm_lat[cell_lat+1]+gcm_lat[cell_lat])/2:
                break
            cell_lat=cell_lat+1;
        gcm_cell[z-3,0]= cell_lat    
        while cell_lon<=len(gcm_lon):
            if (gcm_lon[cell_lon-1]+gcm_lon[cell_lon])/2<=sta_lon[z-3] and sta_lon[z-3]<=(gcm_lon[cell_lon+1]+gcm_lon[cell_lon])/2:
                break
            cell_lon=cell_lon+1;
        gcm_cell[z-3,1]= cell_lon
        # Loading GCM data (HIS+RCP)
        if  cell_lat in np.delete(gcm_cell,z-3,0)[:,0] and cell_lon in np.delete(gcm_cell,z-3,0)[:,1]:       
            cell_pos=np.where((gcm_cell[:,0:2]==[cell_lat, cell_lon]).all(axis=1))      
            gcm_cell[z-3,2]=gcm_cell[cell_pos[0][0],2]
        else:         
            data_his=[];data_rcp=[]
            for filename in os.listdir(model_historic):
                his = nc.Dataset(model_historic+'\\'+filename)    
                data_his=np.concatenate((data_his,his.variables[var_type][:,cell_lat,cell_lon]*factor1+factor2))    
            for filename in os.listdir(model_rcp):
                rcp = nc.Dataset(model_rcp+'\\'+filename)      
                data_rcp=np.concatenate((data_rcp,rcp.variables[var_type][:,cell_lat,cell_lon]*factor1+factor2))    
            model[n]=np.concatenate((data_his, data_rcp))
            gcm_cell[z-3,2]=n
            n=n+1
    his.close
    rcp.close
    caos_results=np.empty((n,len(int_acum)),dtype=list)
    ds_results=[]
    for z in range(3,ns):
        print "Station: " +station_names[z]
        data_sta= station[1:,z]
        time_z=np.array(time_sta)[~np.isnan(data_sta)]   
        data_sta= data_sta[~np.isnan(data_sta)]
        data_gcm=model[int(gcm_cell[z-3,2])]
        # Dates:calibration, validation and application in time_series
        offset=int(np.where(time_gcm==dt.datetime.combine(time_z[0],dt.datetime.min.time()))[0])
        cal_end=int(np.where(time_z==cal_date)[0])
        if not (time_z==val_date).all():
            val_end=len(time_z)-1
        else:
            val_end=int(np.where(time_z==val_date)[0])
        app_end=int(np.where(time_gcm==dt.datetime.combine(app_date,dt.datetime.min.time()))[0])
        print "    Chaos Analysis"
        for step in int_acum:
            print "      step: " + str(step)
            # Load existing results and mean data values for different mean intervals
            caos_pos=int(np.where(np.array(int_acum)==step)[0])
            if caos_results[int(gcm_cell[z-3,2]),caos_pos] is None:
                pr_sta=prom_data(data_sta,step)       
                lag_sta,m_sta,lyapunov_sta,caos_sta=chaos_analysis(pr_sta)
            elif caos_results[int(gcm_cell[z-3,2]),caos_pos][0]==1:
                pr_sta=prom_data(data_sta,step)       
                lag_sta,m_sta,lyapunov_sta,caos_sta=chaos_analysis(pr_sta)    
                if caos_sta==1:
                    lag_gcm=caos_results[int(gcm_cell[z-3,2]),caos_pos][1]
                    m_gcm=caos_results[int(gcm_cell[z-3,2]),caos_pos][2]
                    break
                else:
                    next                    
            else:
                next 
            if caos_sta==1:
                pr_gcm=prom_data(data_gcm,step) 
                lag_gcm,m_gcm,lyapunov_gcm,caos_gcm=chaos_analysis(pr_gcm)
                caos_results[int(gcm_cell[z-3,2]),caos_pos]=[caos_gcm,lag_gcm,m_gcm]
                if caos_gcm==1:
                    break    
        print "    Phase Space Reconstruction"
        # Mean time series
        len_cal=int(math.ceil(len(data_sta[:cal_end+1])/float(step)))
        len_app=int(math.ceil(len(data_gcm[offset+cal_end:app_end+1])/float(step)))
        sta_acum=prom_data(data_sta,step) 
        gcm_acum=prom_data(data_gcm,step) 
        # Mean delayed vectors
        sta_vec=reconstruct(sta_acum, lag_sta, m_sta)
        gcm_vec=reconstruct(gcm_acum, lag_gcm, m_gcm)  
        sta_cal_vec=sta_vec[:len_cal-(m_sta-1)*lag_sta+1]
        delta=(m_gcm-1)*lag_gcm-(m_sta-1)*lag_sta  
        gcm_cal_vec=gcm_vec[-len_app-len_cal-delta:-len_app-(m_gcm-1)*lag_gcm +1]      
        print "    Calibration"
        # Parameter lag: Generalized Similarity Function
        gcm_cal=deconstruct(gcm_cal_vec, lag_gcm, m_gcm)
        leng=len(gcm_cal)
        sta_cal=sta_acum[:leng] 
        sg_mat=np.empty((int(np.floor(180/step))))* np.nan 
        lag_mat=range(0,int(np.floor(180/step)))
        lag_c=0
        for lag in lag_mat:  
            sta_cal_nlag=sta_cal[1+lag:leng+1]-np.tile(np.mean(sta_cal[1+lag:leng+1]),leng-lag-1)
            sta_cal_vn=sta_cal[:leng-lag-1]-np.tile(np.mean(sta_cal[:leng-lag-1]),leng-lag-1)
            gcm_cal_vn=gcm_cal[:leng-lag-1]-np.tile(np.mean(gcm_cal[:leng-lag-1]),leng-lag-1)
            sg_mat[lag_c]=((np.mean(np.power(sta_cal_nlag-gcm_cal_vn,2))/(leng-lag))/np.power(np.multiply(np.mean(np.power(sta_cal_vn,2)),np.mean(np.power(gcm_cal_vn,2)))/(leng-lag)**2,0.5))**0.5            
            lag_c=lag_c+1 
        lag=lag_mat[int(np.where(sg_mat==np.min(sg_mat))[0])]
        # Lag Correction
        gcm_cal_vec=gcm_vec[-len_app-len_cal-delta-lag:-len_app-lag-(m_gcm-1)*lag_gcm +1]
        if lag==0:
           gcm_app_vec=gcm_vec[-len_app-(m_gcm-1)*lag_gcm+1-lag:]   
        else:
            gcm_app_vec=gcm_vec[-len_app-(m_gcm-1)*lag_gcm+1-lag:-lag] 
        lencal=len(gcm_cal_vec)
        lenapp=len(gcm_app_vec)
        # Parameters u and ss: Mutual False Nearest Neighbor"
        ss=np.empty((lencal+lenapp,lag_sta+1))* np.nan
        u=np.empty((lencal+lenapp,1))* np.nan
        k=0;a=0
        for t in range(0,lencal):
            # Distance in STATION
            distanceA=sklearn.metrics.pairwise.euclidean_distances(gcm_cal_vec,np.reshape(gcm_cal_vec[t],(1,m_gcm))) 
            distanceA1=np.delete(distanceA, t)
            dist_gcm_gcm=min(i for i in distanceA1 if i > 0) 
            indA=np.where(distanceA1==dist_gcm_gcm)[0] 
            # Distance in GCM
            distanceB=sklearn.metrics.pairwise.euclidean_distances(sta_cal_vec,np.reshape(sta_cal_vec[t],(1,m_sta)))
            distanceB1=np.delete(distanceB,t)
            dist_sta_sta= min(j for j in distanceB1 if j > 0)            
            indB=np.where(distanceB1==dist_sta_sta)[0]
            # Distance GCM-STATION,STATION-GCM
            dist_gcm_sta=distanceA[indB][0]    
            dist_sta_gcm=distanceB[indA] [0]     
            # Compute ss and u
            ss[t,lag_sta]=(dist_gcm_sta*dist_sta_gcm)/(dist_gcm_gcm*dist_sta_sta)
            ss[a,k]=ss[t,lag_sta]
            u[t]=np.nanmean(ss[:t+1,lag_sta])
            if k==lag_sta-1:
                k=0
                a=a+1
            else:
                k=k+1
        print "    Synchronization"
        sta_sync=np.empty((lencal+lenapp,m_sta))* np.nan 
        sta_sync[:lencal]=sta_cal_vec
        distanceC=sklearn.metrics.pairwise.euclidean_distances(gcm_cal_vec,gcm_app_vec)
        yn_max=np.nanmax(sta_sync)*2
        if np.min(sta_sync)>=0:
            yn_min=np.min(sta_sync)/2   
        else:
            yn_min=np.min(sta_sync)*2   
        s_max_list=[np.nanmax(ss[:a,r]) for r in range(0,lag_sta)]       
        for v in range(6203,lenapp): #lenapp              
            # Find xn, xnd and ynd (Direct)
            distanceA=np.reshape(distanceC[:,v],lencal)
            dist_gcm_gcm=min(distanceA)
            indA=np.where(distanceA==dist_gcm_gcm)[0]
            # Create variable yn
            yn=np.empty((1,m_sta))* np.nan
            yn[0,0:-1]=sta_sync[t+v+1-lag_sta,1:]        
            # Compute ynr1 and ynd1
            ynr1=np.reshape(sklearn.metrics.pairwise.euclidean_distances(sta_sync[:t+1,0:-1],np.reshape(yn[0,0:-1],(1,m_sta-1)))**2,t+1)
            # Find yn and distances    
            s_max=s_max_list[k]
            args=(sta_sync[:t+1,-1],ynr1,distanceA,indA,dist_gcm_gcm,s_max)           
            min_dist=scipy.optimize.differential_evolution(yn_obj, [(yn_min, yn_max)],args=args,popsize=50, mutation=(1, 1.5), recombination=0.5)          
            distanceB=np.sqrt((min_dist.x-sta_sync[:t+1,-1])**2+ynr1)
            dist_sta_sta=min(distanceB)
            indB=np.where(distanceB==dist_sta_sta)
            dist_gcm_sta=distanceA[indB][0]    
            dist_sta_gcm=distanceB[indA][0]
            # Compute ss and u
            ss[t+v+1,lag_sta]=(dist_sta_gcm*dist_gcm_sta)/(dist_sta_sta*dist_gcm_gcm)
            u[t+v+1]=np.nanmean(ss[:t+v+1,lag_sta]) 
            if k==lag_sta-1:
                k=0
            else:
                k=k+1 
            # Add yn to sta_sync
            yn[0][-1]=min_dist.x 
            sta_sync[t+v+1]=yn
        print "    Time Series Reconstruction"
        # Amp Correction: Mean correction
        data_gcm_original=data_gcm
        amp=np.mean(data_sta[:cal_end+1])/np.mean(data_gcm[offset:offset+cal_end+1])       
        data_gcm=data_gcm*amp  
        # Reconstruct prom data
        sta_app_step=deconstruct(sta_sync,lag_sta, m_sta)
        sta_app=deprom_data(sta_app_step,step) 
        time_ds=[time_z[0]+ dt.timedelta(days=t) for t in range(0,len(sta_app))]
        end_index=time_ds.index(app_date)
        sta_app=sta_app[:end_index+1]
        sta_app[:len(data_sta[:cal_end+1])]=data_sta[:cal_end+1]
        time_ds=time_ds[:end_index+1]
        print "    Validation"           
        ds_mean,ds_date=monthly(sta_app[cal_end+1:val_end+1],time_ds[cal_end+1:val_end+1],var_type)
        val_mean,val_da=monthly(data_sta[cal_end+1:val_end+1],time_ds[cal_end+1:val_end+1],var_type)  
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(ds_date, val_da))
        print "    Writing results to CSV"
        name_file=station_names[z]+'_'+var_type+'_'+rcp_esc
        file_csv=output_folder+'CSD_'+name_file+ '.csv'
        ref=sta_app[:cal_end+1]
        eoc=sta_app[-len(ref):]
        moc=sta_app[val_end+1+len(sta_app[cal_end+1:-len(ref)+1])/2-len(ref)/2:val_end+1+len(sta_app[cal_end+1:-len(ref)+1])/2+len(ref)/2]
        ref_eoc=(np.mean(eoc)-np.mean(ref))/np.mean(ref)*100
        ref_moc=(np.mean(moc)-np.mean(ref))/np.mean(ref)*100
        eoc_start=time_ds[-len(ref)]
        eoc_end=time_ds[-1] 
        moc_start=time_ds[val_end+1+len(sta_app[cal_end+1:-len(ref)+1])/2-len(ref)/2]
        moc_end=time_ds[val_end+1+len(sta_app[cal_end+1:-len(ref)+1])/2+len(ref)/2]
        time_ds[val_end+1+len(sta_app[cal_end+1:-len(ref)+1])/2-len(ref)/2]
        with open(file_csv, 'wb') as csvfile:
            write_ans = csv.writer(csvfile, delimiter=',')
            write_ans.writerow(['Chaotic Statistical Downscaling'])
            write_ans.writerow(['Variable: ',var_type,'Model: ',rcp_model,'Experiment: ',rcp_esc])
            write_ans.writerow(['Time delay:',lag_sta,'Dimension:',m_sta,'Max Lyapunov:',"%.3f" %(max(lyapunov_sta))])      
            write_ans.writerow(['Step(days): ',step,'Calibration Start :',time_z[0],'Validation Start:',time_z[cal_end+1]])        
            write_ans.writerow(['MOC(start-end) :',moc_start,moc_end,'EOC(start-end):',eoc_start,eoc_end])
            write_ans.writerow(['RMSE (monthly): ',"%.2f" %(rmse), 'inc/dec MOC(%)',"%.1f" %(ref_moc),'inc/dec EOC(%)',"%.1f" %(ref_eoc)]) #Reference period Calibration Start- cAlibration End
            write_ans.writerow(['Year','Month','Day','Data'])
            for i in range(val_end+1,len(sta_app)):
                write_ans.writerow([time_ds[i].year,time_ds[i].month,time_ds[i].day, sta_app[i]]) 
        print "    Making some plots"
        #u function
        fig = plt.figure()
        plt.plot(u,'r');plt.xlabel('Time');plt.ylabel('U');plt.title('U function')
        fig.savefig(output_folder+name_file+'_u_function.jpg')
        plt.close()
        #Validation
        fig = plt.figure()
        ticks = np.arange(0, len(val_da), 12); labels=range(ticks.size); plt.xticks(ticks, labels)
        a,=plt.plot(ds_date,'r', label='CSD model'); b,=plt.plot(val_da,'b', label='historic')
        plt.xlabel('Time (years)');plt.ylabel(var_type+' '+units);plt.title('Validation')
        plt.legend(handles=[a, b])
        fig.savefig(output_folder+name_file+'_validation.jpg')
        plt.close()
        #Future  Annual
        anual_date,anual_value=anually(sta_app,time_ds,var_type)
        fig = plt.figure()
        plt.plot(anual_date,anual_value)
        plt.xlabel('Time (years)');plt.ylabel(var_type+' '+units);plt.title('Future Annual')
        fig.tight_layout()
        fig.savefig(output_folder+name_file+'_future_anual.jpg')
        plt.close()
        # Historic Comparation
        sta_his_date,sta_his_value=anually(data_sta,time_ds[:len(data_sta)],var_type)     
        gcm_his_date_original,gcm_his_value_original=anually(data_gcm_original[offset:offset+len(data_sta)],time_ds[:len(data_sta)],var_type)   
        gcm_his_date,gcm_his_value=anually(data_gcm[offset:offset+len(data_sta)],time_ds[:len(data_sta)],var_type)   
        fig = plt.figure()
        a,=plt.plot(sta_his_date,sta_his_value, label='Station');
        b,=plt.plot(gcm_his_date_original,gcm_his_value_original, label='GCM original: '+rcp_model)
        c,=plt.plot(gcm_his_date,gcm_his_value, label='GCM adjusted: '+rcp_model)
        plt.xlabel('Time (years)');plt.ylabel(var_type+' '+units);plt.title('Historic Comparation')
        plt.legend(handles=[a, b, c])
        fig.savefig(output_folder+name_file+'_historic_comp.jpg')
        plt.close()
        # Monthly Multiannual
        month_his=monthly(data_sta,time_ds[:len(data_sta)],var_type)
        month_fut=monthly(sta_app[len(data_sta):],time_ds[len(data_sta):],var_type)
        fig = plt.figure() 
        a,=plt.plot(range(0,12),month_his[0], label='historic'); b,=plt.plot(range(0,12),month_fut[0], label='future')  
        labels = ['January','February','March','April','May','June','July','August','September','October','November','December'];
        plt.xticks(range(0,12),labels,rotation='vertical')
        plt.ylabel(var_type+' '+units);plt.title('Monthly Multiannual')
        fig.tight_layout()
        plt.legend(handles=[a, b])
        fig.savefig(output_folder+name_file+'_monthly_multianual.jpg')
        plt.close()
        ds_results.append([station_names[z],rmse,ref_moc,ref_eoc ,time_ds,sta_app])
    print "Done"
    return ds_results
