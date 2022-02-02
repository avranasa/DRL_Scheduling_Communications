import numpy as np
import math
from collections import namedtuple, Counter
import queue
import RealUsersGenerator as UG

class Environment:
    def __init__(self, Protocol, Channel, Traffic, Resources, Hyperparameters, Purpose):         
        #np.random.seed(51)
        
        #N parallel environments
        self.N_parallel_env = Hyperparameters['N_parallel_env_test'] if Purpose == 'test' else Hyperparameters['N_parallel_env_train'] 
        
        #TRAFFIC Information
        self.Kusers = Traffic['Kusers']
        self.Classes = np.empty(len(Traffic['ClassesInfo']), dtype=[('lat',int),('data',float),('imp',float),('prob',float)]) 
        self.Classes['lat'] = [class_user['Max_Latency'] for class_user in Traffic['ClassesInfo']]
        self.Classes['data'] = [class_user['Data']   for class_user in Traffic['ClassesInfo']]
        self.Classes['imp'] = [class_user['Importance']   for class_user in Traffic['ClassesInfo']]
        self.Classes['prob'] = [class_user['Arrival_Prob']   for class_user in Traffic['ClassesInfo']]       
        assert np.sum(self.Classes['prob']) == 1, "Probabilities of class users have to sum to 1.0"
        assert np.all(self.Classes['prob'] >= 0), "All class probabilities have to be higher than 0"
        self.Nclasses = self.Classes.shape[0] 
        self.classOfEveryUser = np.empty((self.N_parallel_env, self.Kusers))
        self.UserGen = UG.GenerateRealUsers(Traffic['ProbTypeUser'], Protocol['TimeSlot'])
        
        #PROTOCOL
        self.Prot = Protocol['RetransProtocol']
        self.CSIestimation =  Protocol['CSIestimation']
        self.TimeSlot = Protocol['TimeSlot']

        #CHANNEL set-up
        self.MeanSNR = None
        self.SmallScaleFading = None #The dim after ResetEnv: (self.N_parallel_env, self.Kusers) ... can be real or complex number
        self.RHO = None
        self.ChannelType = "ExpMarkovTimeCorr"#For the small scaling fading
        self.OracleT = Hyperparameters['OracleTimesteps']  if 'OracleTimesteps' in Hyperparameters else 0 
        if self.OracleT > 0:#the environment needs to "forsee" OracleTimesteps the channel            
            self.futureSmallScaleFading = queue.Queue(maxsize=self.OracleT)
            self.futureRHO = queue.Queue(maxsize=self.OracleT)
            self.futureMeanSNR = queue.Queue(maxsize=self.OracleT)
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
        self.BwBlock = Resources['ResourceBlock']
        self.N_blocks = self.BW / self.BwBlock
        
        #Input for scheduling Algorithms
        if self.CSIestimation == "Full":
            #Maybe add speed in the input
            self.InputFeaturesType = np.dtype([('lat',int,self.Kusers),('data',float,self.Kusers),('imp',float,self.Kusers),('meanSNR',float,self.Kusers), ('rho',float,self.Kusers), ('absH_2',float,self.Kusers)])                         
        elif self.CSIestimation == "Statistical":
            self.InputFeaturesType = np.dtype([('lat',int,self.Kusers),('data',float,self.Kusers),('imp',float,self.Kusers)])
        self.StateUsers = np.zeros((self.N_parallel_env,), dtype=self.InputFeaturesType)    
        self.NewUsersInd = np.zeros((self.N_parallel_env,self.Kusers),dtype=bool) #Stored mainly so as to adjust the memory of previous actions
        
        #Keeping Statistics, Aggregates all the parallel environments           
        self.satisfiedPerClass = Counter()
        self.NusersPerClass = Counter() #One class can be the no-user, Counts only users of the past (dead)
                
        #Starting Channel and State of users   
        self.time = 0   
        self.ResetEnv()


    def CreateNextEntriesOf_Channels_UsersClass(self, prevSmallScaleFading, RHO, SNR , UsersUpdate=None):
            '''It outputs the next time slot the state(channels, characteristic of users, locations)
            iif in oracle mode then it refers to OracleT step in the future

            prevSmallScaleFading, RHO, SNR : None if the simulation starts, else dim=(self.N_parallel_env, self.Kusers)
            UsersUpdate                 : A boolean matrix defining which users to update, dim=(self.N_parallel_env, self.Kusers)
            ChannelState and UsersState : used if you want to define yourself the environment
            '''
            if (UsersUpdate is None) and (self.OracleT > 0): 
                UsersUpdate = (self.futureLastLat == 0) 
            if prevSmallScaleFading is None:
                assert(np.all(UsersUpdate))

            
            #New RHO, SNR
            NewRHO = np.empty( (self.N_parallel_env, self.Kusers))
            NewSNR = np.empty( (self.N_parallel_env, self.Kusers))
            NewRHO[UsersUpdate], NewSNR[UsersUpdate] = self.UserGen(UsersUpdate)
            if prevSmallScaleFading is not None:
                NewRHO[~UsersUpdate], NewSNR[~UsersUpdate] = RHO[~UsersUpdate], SNR[~UsersUpdate]


            #New Small Scale Fading 
            if self.ChannelType == "ExpMarkovTimeCorr":
                NewSmallScaleFading = np.empty( (self.N_parallel_env, self.Kusers), dtype = complex)
                ComplexGauss = (np.random.randn(self.N_parallel_env,self.Kusers)/math.sqrt(2) + 1j*np.random.randn(self.N_parallel_env,self.Kusers)/math.sqrt(2))
                NewSmallScaleFading[ UsersUpdate] = ComplexGauss[UsersUpdate]
                if prevSmallScaleFading is not None:
                    sigma = np.sqrt((1-RHO[~UsersUpdate]**2)/2)
                    NewSmallScaleFading[~UsersUpdate] =  RHO[~UsersUpdate]*prevSmallScaleFading[~UsersUpdate] + sigma*ComplexGauss[~UsersUpdate] 
            else:
                print("give valid channel type")   

            
            #Indexes of the class each new user belongs
            indClasses =  np.random.choice(self.Nclasses, (self.N_parallel_env, self.Kusers), p=self.Classes['prob']) #dim = (self.N_parallel_env,self.Kusers)
            if prevSmallScaleFading is not None:
                indClasses = indClasses[UsersUpdate]  #dim = (sum(UsersUpdate),)
            
            if (self.OracleT > 0) and (prevSmallScaleFading is not None):
                self.futureLastLat[UsersUpdate] = self.Classes['lat'][indClasses]
                return NewSmallScaleFading, NewRHO, NewSNR, indClasses, UsersUpdate
            else:
                return NewSmallScaleFading, NewRHO, NewSNR, indClasses

    
    def ResetCounter(self):
        for c in range(self.Nclasses):
            self.satisfiedPerClass[c] = 0
        self.NusersPerClass = Counter(self.classOfEveryUser.ravel())


    def ResetEnv(self):      
        self.time = 0        
        # Update channels, reset time to 0, pick randomly Kuser from the given Classes as if they just appeared            
        self.SmallScaleFading, self.RHO, self.MeanSNR, self.classOfEveryUser= self.CreateNextEntriesOf_Channels_UsersClass( None, None, None, np.full((self.N_parallel_env, self.Kusers), True) )

        self.StateUsers['lat'] = self.Classes['lat'][self.classOfEveryUser]
        self.StateUsers['data'] = self.Classes['data'][self.classOfEveryUser]
        self.StateUsers['imp'] = self.Classes['imp'][self.classOfEveryUser]
        if self.CSIestimation == "Full":  
            if self.ChannelType == "ExpMarkovTimeCorr":
                self.StateUsers['meanSNR'] = self.MeanSNR
                self.StateUsers['rho'] = self.RHO
                self.StateUsers['absH_2'] = np.real(self.SmallScaleFading)**2 + np.imag(self.SmallScaleFading)**2
        self.ResetCounter()

        # For the Oracle: case
        if self.OracleT>0:
            self.futureLastLat = self.StateUsers['lat'].copy()#.copy() is obligatory
            for t in range(self.OracleT):
                self.futureLastLat -= 1
                if t == 0:
                    T_futureSmallScaleFad, T_futureRHO, T_futureSNR, T_futureNewUsersClass, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass(self.SmallScaleFading, self.RHO, self.MeanSNR)
                else:
                    T_futureSmallScaleFad, T_futureRHO, T_futureSNR, T_futureNewUsersClass, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass(self.futureSmallScaleFading.queue[-1],self.futureRHO.queue[-1], self.futureMeanSNR.queue[-1])
                self.futureSmallScaleFading.put(T_futureSmallScaleFad)
                self.futureArrivalUserInd.put(T_futureArrivalInd)
                self.futureNewUserClass.put(T_futureNewUsersClass) 
                self.futureRHO.put(T_futureRHO)
                self.futureMeanSNR.put(T_futureSNR) 
    

    def UpdateListsOf_Channels_StateUsers(self, UsersUpdate):
        '''Creates the new channel state for ALL users (updates: self.SmallScaleFading)
        updates the self.StateUsers only for the NEW users that appear
        '''
        if self.OracleT == 0: 
            self.SmallScaleFading, self.RHO, self.MeanSNR, NewUsersClass= self.CreateNextEntriesOf_Channels_UsersClass(self.SmallScaleFading, self.RHO, self.MeanSNR, UsersUpdate)
        elif self.OracleT > 0: 
            self.futureLastLat -= 1
            T_futureSmallScaleFad, T_futureRHO, T_futureSNR, T_futureNewUsersClass, T_futureArrivalInd = self.CreateNextEntriesOf_Channels_UsersClass(self.futureSmallScaleFading.queue[-1],self.futureRHO.queue[-1], self.futureMeanSNR.queue[1])
            self.SmallScaleFading, x, NewUsersClass, self.RHO, self.MeanSNR = self.futureSmallScaleFading.get(), self.futureArrivalUserInd.get(), self.futureNewUserClass.get(), self.futureRHO.get(), self.futureMeanSNR.get()
            assert(np.array_equal(UsersUpdate, x))
            self.futureSmallScaleFading.put(T_futureSmallScaleFad)
            self.futureArrivalUserInd.put(T_futureArrivalInd)
            self.futureNewUserClass.put(T_futureNewUsersClass)  
            self.futureRHO.put(T_futureRHO)
            self.futureMeanSNR.put(T_futureSNR) 

        #Update State of Users
        self.StateUsers['lat'][UsersUpdate] = self.Classes['lat'][NewUsersClass]
        self.StateUsers['data'][UsersUpdate] = self.Classes['data'][NewUsersClass]
        self.StateUsers['imp'][UsersUpdate] = self.Classes['imp'][NewUsersClass]
        if self.CSIestimation == "Full":             
            if self.ChannelType == "ExpMarkovTimeCorr":
                self.StateUsers['meanSNR'] = self.MeanSNR
                self.StateUsers['rho'] = self.RHO
                self.StateUsers['absH_2'] = np.real(self.SmallScaleFading)**2 + np.imag(self.SmallScaleFading)**2

        self.classOfEveryUser[UsersUpdate] = NewUsersClass
        self.NusersPerClass.update(NewUsersClass)  

   
    def Step(self,W_alloc):   
        '''Asserting right input of scheduling algorithms
        W_alloc [N_parallel_env, Kusers] ->shows for every user how many resource blocks are allocated
        '''
        
        if np.all(np.sum(W_alloc, axis=1)>self.N_blocks)  or np.all(W_alloc <0 ):
            import pdb;pdb.set_trace()
        assert(W_alloc.shape == (self.N_parallel_env, self.Kusers) )
        #update time
        self.time += 1

        #update latency
        self.StateUsers['lat'] -= 1

        #Compute Mutual Information to every user     
        if self.ChannelType == "ExpMarkovTimeCorr":
            absH_2 =  np.real(self.SmallScaleFading)**2 + np.imag(self.SmallScaleFading)**2
            DataRate = W_alloc * self.BwBlock * np.log2(1 + absH_2*self.MeanSNR)    
            MaxDataSent = DataRate * self.TimeSlot  

        #update requested data        
        if (self.Prot=="HARQ-type I"):
            indSuccess=(self.StateUsers['data'] <= MaxDataSent)          
        else:
            print("give valid retransmission protocol")        
        
        #compute how many were satisfied in THIS time slot and update importance
        PreviousRound_Unsatisfied = (self.StateUsers['imp'] > 0)#So the null users not requiring data will not contribute
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
        #The BaselineIntLPoracle in EACH STEP calls this function so as to get its required matrices so as to find the scheduling
        if self.Wth.qsize() == 0:
            self.futureLastDat = self.StateUsers['data'].copy()#.copy() is obligatory....!   
            BW_float =  self.futureLastDat /np.maximum(np.log2(1 + self.StateUsers['absH_2']*self.MeanSNR )*self.TimeSlot, 1e-6 )+1            
            self.Wth.put(np.ceil(BW_float/self.BwBlock))

        
        IterRange = range(self.OracleT) if self.futureUserImp.qsize() == 0 else [-1]
        for t in IterRange:#from next time slot to OracleT slots ahead
            aux1 = np.zeros( (self.N_parallel_env,self.Kusers) )
            aux2 = np.zeros( (self.N_parallel_env,self.Kusers),dtype=int )            
            aux1[self.futureArrivalUserInd.queue[t]] = self.Classes['imp'][self.futureNewUserClass.queue[t]]
            aux2[self.futureArrivalUserInd.queue[t]] = self.Classes['lat'][self.futureNewUserClass.queue[t]]
            self.futureLastDat[self.futureArrivalUserInd.queue[t]] = self.Classes['data'][self.futureNewUserClass.queue[t]]                   

            if self.ChannelType == "ExpMarkovTimeCorr":
                absH_2 = np.real(self.futureSmallScaleFading.queue[t])**2 + np.imag(self.futureSmallScaleFading.queue[t])**2
            if t == -1:
                self.futureUserImp.get()
                self.futureUserLat.get()
                self.Wth.get()
            self.futureUserImp.put(aux1)
            self.futureUserLat.put(aux2)  
            BW_float = self.futureLastDat /np.maximum(np.log2(1 + absH_2*self.futureMeanSNR.queue[t] )*self.TimeSlot, 1e-6 )+1        
            self.Wth.put(np.ceil(BW_float/self.BwBlock))
        futUserImp = np.moveaxis(np.array(self.futureUserImp.queue), 0, -1)
        futUserLat = np.moveaxis(np.array(self.futureUserLat.queue), 0, -1)
        Wth = np.moveaxis(np.array(self.Wth.queue), 0, -1)

        assert(futUserImp.shape == (self.N_parallel_env, self.Kusers, self.OracleT) and futUserLat.shape == futUserImp.shape )
        assert(Wth.shape == (self.N_parallel_env, self.Kusers, self.OracleT+1))
        
        return futUserLat, futUserImp, Wth