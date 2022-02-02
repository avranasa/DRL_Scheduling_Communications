import numpy as np
import math
from collections import namedtuple, Counter
import queue


class Environment:
    def __init__(self, Protocol, Geometry, Channel, Traffic, Resources, Hyperparameters, Purpose, EnvSEED=None): 
        
        if EnvSEED is not None:
            np.random.seed(EnvSEED)
        
        #N parallel environments
        self.N_parallel_env = Hyperparameters['N_parallel_env_test'] if Purpose == 'test' else Hyperparameters['N_parallel_env_train'] 
        
        #TRAFFIC Information
        self.Kusers = Traffic['Kusers']
        self.Classes = np.empty(len(Traffic['ClassesInfo']), dtype=[('lat',int),('data',float),('imp',float),('prob',float)]) #data "abusively" are in float because mutual information (with which we compare) can be float 
        self.Classes['lat'] = [class_user['Max_Latency'] for class_user in Traffic['ClassesInfo']]
        self.Classes['data'] = [class_user['Data']   for class_user in Traffic['ClassesInfo']]
        self.Classes['imp'] = [class_user['Importance']   for class_user in Traffic['ClassesInfo']]
        self.Classes['prob'] = [class_user['Arrival_Prob']   for class_user in Traffic['ClassesInfo']]       
        assert np.sum(self.Classes['prob']) == 1, "Probabilities of class users have to sum to 1.0"
        assert np.all(self.Classes['prob'] >= 0), "All class probabilities have to be higher than 0"
        self.Nclasses = self.Classes.shape[0] 
        self.classOfEveryUser = np.empty((self.N_parallel_env, self.Kusers))
        
        #PROTOCOL
        self.Prot = Protocol['RetransProtocol']
        self.CSIestimation =  Protocol['CSIestimation']

        #Geometry
        self.RadiusRange = [Geometry['Rmin'], Geometry['Rmax']]
        self.DistUsersBS = np.empty((self.N_parallel_env, self.Kusers))

        #CHANNEL set-up
        self.ChannelType = Channel['ChannelType']
        self.ChannelInfo = Channel['ChannelInfo']
        self.PathLoss = Channel['PathLoss']
        self.ConstLoss_Noise = Channel['ConstLoss_div_Noise']
        self.Channels = None #The dim after ResetEnv: (self.N_parallel_env, self.Kusers) ... can be real or complex number
        self.OracleT = Hyperparameters['OracleTimesteps']  if 'OracleTimesteps' in Hyperparameters else 0 #0 if only the current time is needed and for example 1 if also the next time slot's information will be given 
        if self.OracleT > 0:#the environment need to "forsee" OracleTimesteps the channel            
            self.futureChannels = queue.Queue(maxsize=self.OracleT)
            self.futureDistUsersBS = queue.Queue(maxsize=self.OracleT)
            self.futureArrivalUserInd = queue.Queue(maxsize=self.OracleT)
            self.futureNewUserClass = queue.Queue(maxsize=self.OracleT)
            self.futureLastLat = np.empty((self.N_parallel_env, self.Kusers), dtype=int)
            #The following are computed only for conveniently passing them to "BaselineIntLPoracle"
            self.Wth = queue.Queue(maxsize=self.OracleT+1)
            self.futureUserImp = queue.Queue(maxsize=self.OracleT)
            self.futureUserLat = queue.Queue(maxsize=self.OracleT)
            self.futureLastDat = np.empty((self.N_parallel_env, self.Kusers))
       
        #RESOURCE constraints
        self.BW = Resources['BW']
        self.EperSym = Resources['EnergyPerSymbol' ]           
        
        #Input for scheduling Algorithms
        if self.CSIestimation == "Full":
            self.InputFeaturesType = np.dtype([('lat',int,self.Kusers),('data',float,self.Kusers),('imp',float,self.Kusers),('dist',float,self.Kusers), ('absH_2',float,self.Kusers)])#instead of channel gain you could also give immidiately the required BW                         
        elif self.CSIestimation == "Statistical":
            self.InputFeaturesType = np.dtype([('lat',int,self.Kusers),('data',float,self.Kusers),('imp',float,self.Kusers),('dist',float,self.Kusers)])
        self.StateUsers = np.zeros((self.N_parallel_env,), dtype=self.InputFeaturesType)    
        self.NewUsersInd = np.zeros((self.N_parallel_env,self.Kusers),dtype=bool) #Stored mainly so as to adjust the memory of previous actions
        
        #Keeping Statistics            
        self.satisfiedPerClass = Counter()
        self.NusersPerClass = Counter() #One class can be the no-user, Counts only users of the past (dead)
                
        #Starting Channel and State of users   
        self.time = 0   
        self.ResetEnv()


    def CreateNextEntriesOf_Channels_UsersClass_Location(self, prevChannels, prevDistances, UsersUpdate=None):
            ''' It outputs the next time slot the state(channels, characteristic of users, locations)
            if in oracle mode then it refers to OracleT step in the future

            prevChannels/prevDistances  : None if the simulation starts, else dim=(self.N_parallel_env, self.Kusers)
            UsersUpdate                 : A boolean matrix defining which users to update, dim=(self.N_parallel_env, self.Kusers)
            ChannelState and UsersState : used if you want to define yourself the environment
            '''
            if (UsersUpdate is None) and (self.OracleT > 0): 
                UsersUpdate = (self.futureLastLat == 0) 
            if prevChannels is None:
                assert(np.all(UsersUpdate))


            #New channels
            if self.ChannelType == "ExpMarkovTimeCorr":
                NewChannels = np.empty( (self.N_parallel_env, self.Kusers), dtype = complex)
                rho = self.ChannelInfo
                sigma = math.sqrt((1-rho**2)/2)
                NewChannels[ UsersUpdate] = (np.random.randn(self.N_parallel_env,self.Kusers)[UsersUpdate]/math.sqrt(2) + 1j*np.random.randn(self.N_parallel_env,self.Kusers)[UsersUpdate]/math.sqrt(2))
                if prevChannels is not None:
                    NewChannels[~UsersUpdate] =  rho*prevChannels[~UsersUpdate] + (sigma*np.random.randn(self.N_parallel_env,self.Kusers)[~UsersUpdate] + 1j*sigma*np.random.randn(self.N_parallel_env,self.Kusers)[~UsersUpdate])  
            else:
                print("give valid channel type")   

            
            #New distances form BS
            UsersDist = np.empty( (self.N_parallel_env, self.Kusers))
            UsersDist[UsersUpdate] = np.sqrt(np.random.rand(self.N_parallel_env, self.Kusers)[UsersUpdate]*(self.RadiusRange[1]**2-self.RadiusRange[0]**2)+self.RadiusRange[0]**2 )#the necessary trasformation to get the linear pdf in [Rmin,Rmax]
            if prevChannels is not None:
                UsersDist[~UsersUpdate] = prevDistances[~UsersUpdate]


            #Indexes of the class each new user belongs
            indClasses =  np.random.choice(self.Nclasses, (self.N_parallel_env, self.Kusers), p=self.Classes['prob']) #dim = (self.N_parallel_env,self.Kusers)
            if prevChannels is not None:
                indClasses = indClasses[UsersUpdate]  #dim = (sum(UsersUpdate),)
            
            if (self.OracleT > 0) and (prevChannels is not None):
                self.futureLastLat[UsersUpdate] = self.Classes['lat'][indClasses]
                return NewChannels, indClasses, UsersDist, UsersUpdate
            else:
                return NewChannels, indClasses, UsersDist

    
    def ResetCounter(self):
        for c in range(self.Nclasses):
            self.satisfiedPerClass[c] = 0
        self.NusersPerClass = Counter(self.classOfEveryUser.ravel())


    def ResetEnv(self):      
        self.time = 0        
        # Update channels, reset time to 0, pick randomly Kuser from the given Classes as if they just appeared            
        self.Channels, self.classOfEveryUser, self.DistUsersBS = self.CreateNextEntriesOf_Channels_UsersClass_Location( None, None, np.full((self.N_parallel_env, self.Kusers), True) ) 

        self.StateUsers['lat'] = self.Classes['lat'][self.classOfEveryUser]
        self.StateUsers['data'] = self.Classes['data'][self.classOfEveryUser]
        self.StateUsers['imp'] = self.Classes['imp'][self.classOfEveryUser]
        self.StateUsers['dist'] = self.DistUsersBS.copy()
        if self.CSIestimation == "Full":  
            if self.ChannelType == "ExpMarkovTimeCorr":
                self.StateUsers['absH_2'] = np.real(self.Channels)**2 + np.imag(self.Channels)**2
        self.ResetCounter()

        # For the Oracle case
        if self.OracleT>0:
            self.futureLastLat = self.StateUsers['lat'].copy()#.copy() is obligatory
            for t in range(self.OracleT):
                self.futureLastLat -= 1
                if t == 0:
                    T_futureChannels, T_futureNewUsersClass, T_futureDistUsersBS, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass_Location(self.Channels,self.DistUsersBS)
                else:
                    T_futureChannels, T_futureNewUsersClass, T_futureDistUsersBS, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass_Location(self.futureChannels.queue[-1],self.futureDistUsersBS.queue[-1])
                self.futureChannels.put(T_futureChannels)
                self.futureArrivalUserInd.put(T_futureArrivalInd)
                self.futureNewUserClass.put(T_futureNewUsersClass) 
                self.futureDistUsersBS.put(T_futureDistUsersBS) 
    

    def UpdateListsOf_Channels_StateUsers(self, UsersUpdate):
        #Creates the new channel state for ALL users (updates: self.Channels)
        #updates the self.StateUsers only for the NEW users that appear
        if self.OracleT == 0: 
            self.Channels, NewUsersClass, self.DistUsersBS = self.CreateNextEntriesOf_Channels_UsersClass_Location(self.Channels, self.DistUsersBS, UsersUpdate)
        elif self.OracleT > 0:
            self.futureLastLat -= 1
            T_futureChannels, T_futureNewUsersClass, T_futureDistUsersBS, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass_Location(self.futureChannels.queue[-1],self.futureDistUsersBS.queue[-1])
            self.Channels, x, NewUsersClass, self.DistUsersBS = self.futureChannels.get(), self.futureArrivalUserInd.get(), self.futureNewUserClass.get(), self.futureDistUsersBS.get()
            assert(np.array_equal(UsersUpdate, x))
            self.futureChannels.put(T_futureChannels)
            self.futureArrivalUserInd.put(T_futureArrivalInd)
            self.futureNewUserClass.put(T_futureNewUsersClass)  
            self.futureDistUsersBS.put(T_futureDistUsersBS)

        #Update State of Users
        self.StateUsers['lat'][UsersUpdate] = self.Classes['lat'][NewUsersClass]
        self.StateUsers['data'][UsersUpdate] = self.Classes['data'][NewUsersClass]
        self.StateUsers['imp'][UsersUpdate] = self.Classes['imp'][NewUsersClass]
        self.StateUsers['dist'] = self.DistUsersBS.copy()
        if self.CSIestimation == "Full":             
            if self.ChannelType == "ExpMarkovTimeCorr":
                self.StateUsers['absH_2'] = np.real(self.Channels)**2 + np.imag(self.Channels)**2     
      
        self.classOfEveryUser[UsersUpdate] = NewUsersClass
        self.NusersPerClass.update(NewUsersClass)  

   
    def Step(self,W_alloc, E_alloc):   
        #Asserting right input of scheduling algorithms
        assert(np.all(np.sum(W_alloc, axis=1)<1.01*self.BW) and np.all(W_alloc>=0 ))#1.002 there may be some floating number approximated leading to imprecisions
        assert(W_alloc.shape == (self.N_parallel_env, self.Kusers) and E_alloc.shape == W_alloc.shape)
        
        #update time
        self.time += 1

        #update latency
        self.StateUsers['lat'] -= 1

        #Compute Mutual Information to every user     
        if self.ChannelType == "ExpMarkovTimeCorr":
            absH_2 =  np.real(self.Channels)**2 + np.imag(self.Channels)**2
            MutInfo = W_alloc * np.log2(1 + self.ConstLoss_Noise*absH_2*(self.DistUsersBS**-self.PathLoss)*E_alloc)        

        #update requested data        
        if (self.Prot=="HARQ-type I"):
            indSuccess=(self.StateUsers['data'] <= MutInfo)          
        elif (self.Prot=="HARQ-type II"):
            self.StateUsers['data'] -= MutInfo
            indSuccess = (self.StateUsers['data']<=0)
        else:
            print("give valid retransmission protocol")        
        
        #compute how many were satisfied in THIS time slot and update importance
        PreviousRound_Unsatisfied = (self.StateUsers['imp'] > 0)
        assert  (PreviousRound_Unsatisfied == (self.StateUsers['data'] > 0)).all()
        ThisRound_Satisfied = indSuccess & PreviousRound_Unsatisfied
        self.satisfiedPerClass.update(self.classOfEveryUser[ThisRound_Satisfied])
        Reward_from_satisfied = np.sum( ThisRound_Satisfied * self.StateUsers['imp'],1 )  
        assert(Reward_from_satisfied.shape == (self.N_parallel_env,) ) 
        self.StateUsers['imp'][ThisRound_Satisfied] = 0 #so as to not give reward for the same user in the future    
        self.StateUsers['data'][indSuccess] = 0 

        #Compute penalty and unsatisfied users  
        indDead = (self.StateUsers['lat'] == 0)   
        self.NewUsersInd = indDead
        indDeadAndUnsatisfied = indDead & (self.StateUsers['imp'] > 0)  
        Punish_from_unsatisfied = np.sum( indDeadAndUnsatisfied * self.StateUsers['imp'], 1 )  
        assert(Punish_from_unsatisfied.shape == (self.N_parallel_env,) )  
       
        #update State and return reward
        self.UpdateListsOf_Channels_StateUsers(indDead)   
        return Reward_from_satisfied, Punish_from_unsatisfied

    
    def SuccessRatePerClass(self):
        return self.satisfiedPerClass, self.NusersPerClass


    def Provide_ILPoracle_matrices(self):
        #Assuming energy per channel use always equal to E_alloc
        #The BaselineIntLPoracle in EACH STEP calls this function so as to get its required matrices so as to find the scheduling
        if self.Wth.qsize() == 0:
            self.futureLastDat = self.StateUsers['data'].copy()#.copy() is obligatory....!
            self.Wth.put(self.futureLastDat / np.log2(1 + self.ConstLoss_Noise*self.StateUsers['absH_2']*(self.DistUsersBS**-self.PathLoss)*self.EperSym ) + 1e-3)#CHANGE

        
        IterRange = range(self.OracleT) if self.futureUserImp.qsize() == 0 else [-1]
        for t in IterRange:#from next time slot to OracleT slots ahead
            aux1 = np.zeros( (self.N_parallel_env,self.Kusers) )
            aux2 = np.zeros( (self.N_parallel_env,self.Kusers),dtype=int )            
            aux1[self.futureArrivalUserInd.queue[t]] = self.Classes['imp'][self.futureNewUserClass.queue[t]]
            aux2[self.futureArrivalUserInd.queue[t]] = self.Classes['lat'][self.futureNewUserClass.queue[t]]
            self.futureLastDat[self.futureArrivalUserInd.queue[t]] = self.Classes['data'][self.futureNewUserClass.queue[t]]                   

            if self.ChannelType == "ExpMarkovTimeCorr":
                absH_2 = np.real(self.futureChannels.queue[t])**2 + np.imag(self.futureChannels.queue[t])**2
            if t == -1:
                self.futureUserImp.get()
                self.futureUserLat.get()
                self.Wth.get()
            self.futureUserImp.put(aux1)
            self.futureUserLat.put(aux2)
            self.Wth.put( self.futureLastDat/np.log2(1+self.ConstLoss_Noise*absH_2*(self.futureDistUsersBS.queue[t]**-self.PathLoss)*self.EperSym ) + 1e-3)
        futUserImp = np.moveaxis(np.array(self.futureUserImp.queue), 0, -1)
        futUserLat = np.moveaxis(np.array(self.futureUserLat.queue), 0, -1)
        Wth = np.moveaxis(np.array(self.Wth.queue), 0, -1)

        assert(futUserImp.shape == (self.N_parallel_env, self.Kusers, self.OracleT) and futUserLat.shape == futUserImp.shape )
        assert(Wth.shape == (self.N_parallel_env, self.Kusers, self.OracleT+1))
        
        return futUserLat, futUserImp, Wth