import math
import numpy as np
import TestMethods



class ExponentialRuleMethod():
    #Exponential Rule from Paper: "Scheduling Algorithms for a Mixture of Real-Time and Non-Real-Time Data in HDR"
    def __init__(self, Resources, hyper_param):
        self.BW = Resources['BW']  
        self.BwBlock = Resources['ResourceBlock']
        self.N_blocks = self.BW / self.BwBlock
        self.delta = hyper_param['delta']
        self.sum_mu_all = None #The sum of all the rates every active user got since appearing 
        self.UserLastRound = None

    def UpdateEstimationOf_mean_mu(self, env, mu):
        if self.sum_mu_all is None:
            self.sum_mu_all = mu
            self.UserLastRound = env.StateUsers['lat']==1.0
        else:
            self.sum_mu_all[self.UserLastRound] = mu[self.UserLastRound]
            self.sum_mu_all[~self.UserLastRound] += mu[~self.UserLastRound]
            self.UserLastRound = env.StateUsers['lat']==1.0



    def __call__(self, env):        
        '''The algorithm does the following:
        1)compute the required bandwidth for every user
          with the comparisons when env.step() is called
        2)compute priorities for every user using the Exponential Rule
        3)Sort the N first possible to be scheduled    
        '''
        Walloc = np.zeros((env.N_parallel_env,env.Kusers), dtype=float)       
        if env.CSIestimation == "Full":
            BW_float_needed = env.StateUsers['data'] /np.maximum(np.log2(1 + env.StateUsers['absH_2']*env.MeanSNR )*env.TimeSlot, 1e-6 )+1#1Hz more 
            Blocks_needed = np.ceil(BW_float_needed/self.BwBlock)
            mu_All = np.log2(1 + env.StateUsers['absH_2']*env.MeanSNR )
            self.UpdateEstimationOf_mean_mu(env,mu_All)
            for sample in range(env.N_parallel_env):   
                ind_ImpossibleToSatisfy = Blocks_needed[sample]>self.N_blocks
                # Below it avoids getting negative number because huge float numbers when quantized need more than 32bits integer. 
                # Anything bigger than capacity self.BW by default is not being satisfied
                Blocks_needed[sample][ind_ImpossibleToSatisfy] = 2*self.N_blocks 
                imp = env.StateUsers[sample]['imp']   
                T = env.StateUsers[sample]['lat']#maximum delay user can tolerate
                W = env.Classes['lat'][env.classOfEveryUser[sample]] - T#Waiting time = Maximum Latency of every user's class - maximum delay user can tolerate
                alpha = -np.log(self.delta)/T      
                mu = np.log2(1 + env.StateUsers[sample]['absH_2']*env.MeanSNR[sample] )
                mu = mu_All[sample]
                mean_mu = self.sum_mu_all[sample]/(W+1)
                gamma = alpha/np.maximum(mean_mu,1e-6)
                denom = 1+np.sqrt(np.mean(alpha*W))
                values = imp * gamma * mu * np.exp(alpha*W/denom)
                indSort = np.argsort(values)[::-1]                
                BlocksUsed = 0
                i=0
                for user in indSort:
                    n = Blocks_needed[sample,user]
                    if BlocksUsed+n<self.N_blocks:
                        i=i+1
                        BlocksUsed = BlocksUsed+n
                        Walloc[sample,user] = n  
        else:
            #Assume that you know the mean rate of the user (needs some oracle power)
            Blocks_needed = env.StateUsers['data'] /np.maximum(np.log2(1 + env.MeanSNR )*env.TimeSlot, 1e-6 )+1#1Hz more 
            for sample in range(env.N_parallel_env):   
                ind_ImpossibleToSatisfy = Blocks_needed[sample]>self.BW
                # Below it avoids getting negative number because huge float numbers when quantized need more than 32bits integer. 
                # Anything bigger than capacity self.BW by default is not being satisfied
                Blocks_needed[sample][ind_ImpossibleToSatisfy] = 2*self.BW 
                imp = env.StateUsers[sample]['imp']   
                T = env.StateUsers[sample]['lat']#maximum delay user can tolerate
                W = env.Classes['lat'][env.classOfEveryUser[sample]] - T#Waiting time = Maximum Latency of every user's class - maximum delay user can tolerate
                alpha = -np.log(self.delta)/T
                gamma = alpha
                denom = 1+np.sqrt(np.mean(alpha*W))
                values = imp * gamma * np.exp(alpha*W/denom)
                indSort = np.argsort(values)[::-1]         
                BlocksUsed = 0
                i=0
                for user in indSort:
                    n = Blocks_needed[sample,user]
                    if BlocksUsed+n<self.N_blocks:
                        i=i+1
                        BlocksUsed = BlocksUsed+n
                        Walloc[sample,user] = n 
        return Walloc

  
def BaselineTest_ExponentialRule(env, Resources, hyper_param):
    Scheduler = ExponentialRuleMethod(Resources, hyper_param)
    TestMethods.test_Scheduler("ExpRule", Scheduler, env)
