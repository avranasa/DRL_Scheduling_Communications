from os import listdir
from os.path import isfile, join
import math
from geopy import distance #https://geopy.readthedocs.io/en/stable/#module-geopy.distance
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import jv as Bessel
from scipy.interpolate import interp1d
import pdb

#Data taken from paper of "HTTP/2-Based Adaptive Streaming of HEVC Video over 4G/LTE Networks"

class GenerateRealUsers():
    def __init__(self, ProbTypeUser, TimeSlot):       
        self.Prob_typeOfuser = ProbTypeUser 
        self.Categ = ['bicycle', 'bus', 'car', 'foot', 'train', 'tram']
        self.Ncateg = len(self.Categ)
        self.Prob=[self.Prob_typeOfuser[cat] for cat in self.Categ]
        self.fcarrier = 2600*1e6
        self.TimeSlot = TimeSlot
        sumProb=0.0
        for cat in self.Categ:
            sumProb = sumProb + self.Prob_typeOfuser[cat]
        assert(sumProb == 1.0)
        self.AssumeBW = 15*1e6#Unfortunately in their paper is not stated how much BW they used. It can be estimated if
                              #we assume the mean SNR to be around 6dB. If they used 10MHz then mean(SNR) = 11.5 (approx)
                              #if 15MHz then mean(SNR) = 6dB. 
                              #We will assume that SNR is the same in all the sub-bands 
        self.RhoSNR_per_Cat = self.Create_RhoSNR_distributions('./logs_traffic','./ExpectedRate.log') #Output a dictionary per Cat with [Nsamples,2] (first column the rho and second the rate)


    def MeanRate2MeanSNR(self,path2ExpectedRate, DataRates):
        #Inputs (Bits in one Sec)/AssumeBW and outputs mean(SNR)
        L=[]
        with open(path2ExpectedRate,'r') as File:
            for line in File:
                L.append([float(x) for x in line.split()])
        MeanRate2MeanSNR = interp1d(L[0],L[1],fill_value="extrapolate")
        MeanSNR = MeanRate2MeanSNR(DataRates/self.AssumeBW)
        return MeanSNR


    def smoothening_speeds(self,data_log):
        aux = data_log[:,0][data_log[:,0]>0]
        median_of_positives = np.median(aux)
        data_log=np.clip(data_log,0,[3*median_of_positives,np.inf] )
        data_log[:,0]=savgol_filter(data_log[:,0],11,3)# window size 11, polynomial order 3
        data_log[:,0]=np.maximum(data_log[:,0],0 )
        return data_log


    def Speed2TimeCorrelation(self, Speeds):
        #Implements the Jakes model
        c = 299792458 #Speed Of Light
        args = 2*math.pi * Speeds * self.fcarrier/c*self.TimeSlot  *2
        ind_zero = args > 2.4048 #this is the first root of the Bessel. For speeds (usually higher than 150km/h) that result in bigger than that argument we assume i.i.d. channels 
        rho = Bessel(0, args)
        rho[ind_zero] = 0
        return rho


    def Create_RhoSNR_distributions(self,path2logs_traffic, path2ExpectedRate):
        #Returns a dictionary which for every transportation mean has a numpy.array of [Nsamples,2] (first column the RHO and second the mean(SNR))
        onlyfiles = [f for f in listdir(path2logs_traffic) if isfile(join(path2logs_traffic, f))]        
        DATA_categ = {}
        for f in onlyfiles:
            #per log get the Speed-Rate
            data_log_name = f.replace('.','_').split('_')#First the name of the trace and the rest (speed, data rate) in (Km/h, Mbytes per sec)
            data_log = []
            with open( join(path2logs_traffic, f), "r") as fread:
                first = True
                for line in fread:
                    #Compute speed and get rates  
                    aux = line.split()
                    if first:
                        XY_0=(float(aux[2]), float(aux[3]) )
                        first = False
                        continue
                    XY_1 = (float(aux[2]), float(aux[3]) )
                    Data, Dt = float(aux[4]) * 8, float(aux[5])/1e3 #Turn units into Bits and sec
                    Dist = distance.distance(XY_0, XY_1, ellipsoid='WGS-84').km * 1000 #In meters
                    speed, rate = Dist/Dt, Data/Dt
                    data_log.append([speed, rate])            
                    XY_0 = XY_1
            data_log = np.array(data_log)
            data_log = self.smoothening_speeds(data_log)
            if data_log_name[1] in DATA_categ:
                DATA_categ[data_log_name[1]] = np.append(DATA_categ[data_log_name[1]], np.copy(data_log), axis=0)
            else:
                DATA_categ[data_log_name[1]] = np.copy(data_log)

        #Turn speeds->Rho & DataRate->mean(snr) 
        for cat in self.Categ:
            DATA_categ[cat][:,0] = self.Speed2TimeCorrelation( DATA_categ[cat][:,0] )
            DATA_categ[cat][:,1] = self.MeanRate2MeanSNR(path2ExpectedRate, DATA_categ[cat][:,1])

        return DATA_categ


    def __call__(self, IndNewUsers):
        '''Takes the input IndNewUsers, a binary of dimensions [N_parallel_env, Kusers], and generates a 
        new user per every true value of that matrix. Outputs N users of some category, their mean SNR-LargeScale
        '''
        #
        NnewUsersAllEnv = np.sum(IndNewUsers)
        UsersPerCat = np.random.multinomial(n=NnewUsersAllEnv,pvals=self.Prob)
        IndUserRandom =  np.random.permutation(NnewUsersAllEnv)

        for cat,i in zip(self.Categ,range(self.Ncateg)):
            y = self.RhoSNR_per_Cat[cat][np.random.choice(len(self.RhoSNR_per_Cat[cat]),UsersPerCat[i])]
            assert(list(y.shape)==[UsersPerCat[i], 2])
            if i==0:
                Y=y
            else:
                Y=np.append(Y,y, axis=0)
        return Y[IndUserRandom,0],Y[IndUserRandom,1] #rho , mean(SNR)
        



    

    








