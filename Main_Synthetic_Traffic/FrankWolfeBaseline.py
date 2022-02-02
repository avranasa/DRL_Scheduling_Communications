
import math
from collections import deque
import numpy as np
import scipy as scp
from scipy.optimize import minimize_scalar
import TestMethods


class FrankWolfeBaseline():
    ''' It is used in the paper for the no-CSI case. But it has access to the statistics of channel and traffic generation.
    A kind of HARQ type I is assumed where the Base Station uses FEC and the user if doesn't decode correctly then discards 
    completely the received packet and waits for new transmission.
    '''
    def __init__(self, hyperparameters, env, Resources, Geometry, ChannelStat):
        #getting the parameters
        self.Tsteps_greedy = hyperparameters['T_greedysteps']
        self.N_Initializations = hyperparameters['N_DifferentInitializations']
        self.InitializationType = hyperparameters['InitializationType']
        self.MaxSteps = hyperparameters['MaxSteps']
        self.AssumeChannel = hyperparameters['AssumeChannel']
        self.BW, self.EperSymb  = Resources['BW'], Resources['EnergyPerSymbol']
        self.Rmin, self.Rmax = Geometry['Rmin'], Geometry['Rmax']
        self.PL, self.ConstPL = ChannelStat['PathLoss'], ChannelStat['ConstLoss_div_Noise' ]
        self.Wvar_prev = np.empty((env.N_parallel_env,env.Kusers,self.Tsteps_greedy))
        self.Start = True


        #Setting the constraints for every time slot( sum(w)=BW and w>0 && constraints of simplex)
        self.Aeq = np.zeros((self.Tsteps_greedy,env.Kusers*self.Tsteps_greedy)) #The dirWar is "ravel"-eld accordingly below 
        for i in range(self.Tsteps_greedy):
            self.Aeq[i , i*env.Kusers:(i+1)*env.Kusers]=np.ones(env.Kusers)
        self.Beq = self.BW * np.ones(self.Tsteps_greedy)
        self.Bounds = [(0,None) for i in range(env.Kusers*self.Tsteps_greedy)]
        if self.AssumeChannel == 'Constant':
            self.Wmax_previous = np.zeros((env.N_parallel_env,env.Kusers))
        self.sample = 0
        

    def __call__(self,env):        
        '''For each sample the following steps are taken:
            1)Repeat the following steps self.N_Initializations and keep the best (local optimum) solution
            2)Random Initialization of Wvar
            3)Run FrankWolfe until either no significant improvement or a maximum number of steps is reached
        '''
        Ealloc = self.EperSymb*np.ones((env.N_parallel_env,env.Kusers),dtype=float)  
        Walloc = np.zeros_like(Ealloc)
        for sample in range(env.N_parallel_env): 
            self.sample = sample

            Best_ObjValue = -1
            for i_random_init in range(self.N_Initializations):

                #==============================
                #Random Initialization of Wvar 
                #==============================
                if (i_random_init == 0) and (self.Start == False):
                    WvarLastColumn = np.random.rand(env.Kusers,1)#in the first f
                    WvarLastColumn = self.BW * WvarLastColumn / np.sum( WvarLastColumn )
                    Wvar = np.hstack((self.Wvar_prev[sample,:,1:], WvarLastColumn))
                else:
                    if self.InitializationType[0] == 'absGaussian':
                        Wvar = np.random.normal(loc=self.InitializationType[1], scale=self.InitializationType[2], size=(env.Kusers,self.Tsteps_greedy))
                        Wvar = np.abs(Wvar)
                    elif self.InitializationType[0] == 'Gamma':
                        Wvar = np.random.gamma(shape=self.InitializationType[1], scale=self.InitializationType[2], size=(env.Kusers,self.Tsteps_greedy)) 
                    elif self.InitializationType[0] == 'Uniform':
                        Wvar = np.random.rand(env.Kusers,self.Tsteps_greedy)
                    #Satisfying the constraints
                    Wvar = self.BW * Wvar / np.expand_dims( np.sum(Wvar,axis=0),axis=0 )
                step = 0
                objV = 1e-6 #really small (positive) as initialization for the objective function
                improvement = 1.0



                #===============
                #Run Frank-Wolfe
                #===============
                while step<self.MaxSteps and improvement>0.02:  
                    #Compute the derivatives on Wvar point so as to linearly approximate the objective function
                    dirWvar = self.Total_Reward(env, Wvar, 'Partial_direvatives')

                    #Find direction to move by solving the simplex optimization problem (with interior point method as default)
                    #The minus in "-dirWar" is because linprog is for min and we want max
                    LPsol = scp.optimize.linprog(-dirWvar.ravel(order='F'), A_eq=self.Aeq, b_eq=self.Beq, bounds=self.Bounds, method='interior-point', options = {"maxiter":400})#simplex method often fails
                    if LPsol["status"] != 0:
                        print("\n Problems in LP, status = ", LPsol["status"])
                    Direction = np.reshape( LPsol["x"], (self.Tsteps_greedy,env.Kusers) ).T

                    #Find an (maybe local) optimum       
                    g_res = minimize_scalar(lambda g: -self.Total_Reward(env,Wvar+g*(Direction-Wvar)), bounds=(0,1), method='Bounded', options = {"maxiter":50})    
                    NewObjV = -g_res["fun"]
                    #Take the upadate-step                
                    Wvar += g_res["x"]*(Direction-Wvar)             
                    improvement = (NewObjV-objV)/objV               
                    objV = NewObjV
                    step += 1            
                

                #==================================================
                #Keep the best value of every random initialization
                #==================================================
                if objV > Best_ObjValue:
                    BestWvar = Wvar
                    Best_ObjValue = objV   


            #====================================================================
            #Prepare output, and Belief of channel realization for next iteration
            #====================================================================
            self.Wvar_prev[sample,:,:] = BestWvar
            Walloc[sample,:] = np.expand_dims(BestWvar[:,0], axis = 0)
            if env.ChannelType == 'ExpConstant':
                indUsersWillRemain = (env.StateUsers['lat'][sample]>=2)
                self.Wmax_previous[sample,~indUsersWillRemain] = 0
                indMoreBW = (Walloc[sample]>self.Wmax_previous[sample])
                indChangeWmax_previous = (indUsersWillRemain & indMoreBW)
                self.Wmax_previous[sample,indChangeWmax_previous] = Walloc[sample,indChangeWmax_previous]

        self.Start = False
        return Walloc, Ealloc


    def Ugamma(self, s,x):
        #Computes the upper incomplete gamma function
        return scp.special.gammainc(s,x)*scp.special.gamma(s)

        
    def OneUser_ProbSuccess_PerTimeslot(self, data, dist, Use_dist, Walloc, ChannelType, ChannelInfo, OutputType='Values'):
        # When OuputType == 'Values':
        #   For a user taking resources Walloc, Palloc in length(Walloc) timesteps it is computed
        #   in every of those the success probability given pdf of the channel (and assuming in every 
        #   timestep the user is not yet satisfied and we don't know what happened before)
        # When OutputType == 'Direvatives':
        #   In the same scenario it is computed the derivative of the success probability with respect 
        #   to w_allocated   

        indNonZero = (Walloc/data)>0.03 #To avoid numerical errors/warnings for super small probabilities or direvatives
        zeta = (2**(data/Walloc[indNonZero]) - 1)/(self.ConstPL * self.EperSymb)
        if Use_dist:
            x = zeta*(dist**self.PL)
        else:
            x1, x2 =  zeta*(self.Rmin**self.PL), zeta*(self.Rmax**self.PL)  
    
        if OutputType == 'Values':
            values = np.zeros_like(Walloc)
            if Use_dist:
                values[indNonZero] = np.exp(-x)
            else:
                values[indNonZero] = 2/self.PL * (self.Ugamma(2/self.PL, x2)-self.Ugamma(2/self.PL, x1)) / (x2**(2/self.PL)-x1**(2/self.PL))     
            return values
        elif OutputType == 'Direvatives':            
            direvatives = np.zeros_like(Walloc)
            if Use_dist:
                direvatives[indNonZero] = x*np.exp(-x) * data/(Walloc[indNonZero]**2)/(1 - 2**(-data/Walloc[indNonZero]) ) *math.log(2)
            else:
                direvatives[indNonZero] =  2/self.PL*(self.Ugamma(2/self.PL+1, x2) - self.Ugamma(2/self.PL+1, x1)) / (x2**(2/self.PL)- x1**(2/self.PL)) * data/(Walloc[indNonZero]**2)  /(1 - 2**(-data/Walloc[indNonZero]) ) * math.log(2)
            return direvatives 


    def OneUser_Reward(self, server, data, imp, dist, Walloc, ChannelType, ChannelInfo, OutputType, Use_History, Use_dist):
        # When OuputType == 'Value':
        #   For a user taking resources Walloc, Palloc in length(Walloc) timesteps it is computed
        #   the overall expected reward (after giving Walloc,Palloc) given pdf of the channel 
        # When OutputType == 'Partial_direvatives':
        #   In the same scenario it is computed the partial derivatives of the expected reward with respect 
        #   to every element of Walloc
        
        if imp==0 or data==0:
            return  np.zeros_like(Walloc) if OutputType == "Partial_direvatives" else 0

        #============================
        #channel is assumed to be iid
        #============================
        if self.AssumeChannel == 'iid':
            ProbSuccess = self.OneUser_ProbSuccess_PerTimeslot(data, dist, Use_dist, Walloc, ChannelType, ChannelInfo, 'Values')        
            ProbfailureTotal = np.prod(1-ProbSuccess)
            if OutputType == 'Value':
                return imp*(1 - ProbfailureTotal)
            elif OutputType == 'Partial_direvatives':
                dirProbSuccess = self.OneUser_ProbSuccess_PerTimeslot(data, dist, Use_dist, Walloc, ChannelType, ChannelInfo, 'Direvatives')
                return imp * ProbfailureTotal/(1-ProbSuccess) * dirProbSuccess     

        #===========================
        #channel is assumed constant
        #===========================
        elif self.AssumeChannel == 'Constant':              
            ind_Wmax = np.argmax(Walloc)  
            # When considering a currently active user                 
            if Use_History: 
                w_threshold = self.Wmax_previous[self.sample,server]
                if w_threshold > Walloc[ind_Wmax]:
                    return 0 if OutputType == 'Value' else np.zeros_like(Walloc)
                Prob_condition_fail = 1-self.OneUser_ProbSuccess_PerTimeslot(data, dist, True, np.array([w_threshold]), ChannelType, ChannelInfo, 'Values')                          
                if OutputType == 'Value':
                    Prob_max_fail = 1 - self.OneUser_ProbSuccess_PerTimeslot(data, dist, True, Walloc[ind_Wmax:ind_Wmax+1], ChannelType, ChannelInfo, 'Values')
                    return imp*(1-Prob_max_fail/Prob_condition_fail)
                elif OutputType == 'Partial_direvatives':
                    res = np.zeros_like(Walloc)
                    res[ind_Wmax] = imp*self.OneUser_ProbSuccess_PerTimeslot(data, dist, True, Walloc[ind_Wmax:ind_Wmax+1], ChannelType, ChannelInfo, 'Direvatives')/Prob_condition_fail
                    return res 
            # When considering a future usser     
            else:       
                if OutputType == 'Value':
                    return imp*self.OneUser_ProbSuccess_PerTimeslot(data, dist, False, Walloc[ind_Wmax:ind_Wmax+1], ChannelType, ChannelInfo, 'Values')
                elif OutputType == 'Partial_direvatives':
                    res = np.zeros_like(Walloc)
                    res[ind_Wmax] = imp*self.OneUser_ProbSuccess_PerTimeslot(data, dist, False, Walloc[ind_Wmax:ind_Wmax+1], ChannelType, ChannelInfo, 'Direvatives')
                    return res


        else:
            print("The FrankWolfe Optimization method is not implemented yet for this channel type")
    

    def OneServer_Reward(self, env, server, ProbBranch, i_start, i_end, data, imp, dist, Walloc,  OutputType='Value'):
        # It is a recursive function (every recursion is associated for one user) and at the end computes:
        # -when OuputType == 'Value':
        #       For a "server" (serving one user per timeslot, in total there are Kuser-number of "servers")
        #       taking resources Walloc, Palloc in length(Walloc) timesteps it is computed
        #       the overall expected reward (after giving Walloc,Palloc) given pdf of the channel 
        # -when OutputType == 'Partial_Direvatives':
        #       In the same scenario it is computed the partial derivatives of the expected reward with respect 
        #       to every element of Walloc
        # In the initial(root of tree) run: *ProbBranch == 1
        #                                  *i_start == 0 , i_end == min(maxLatOfCurrentUser-1,self.Tsteps_greedy-1)
        Use_History = (i_start==0) and (self.AssumeChannel == 'Constant')
        Use_dist = (i_start==0) #for furure users location is unkonwn
        if OutputType=='Value':
            output = ProbBranch * self.OneUser_Reward(server, data, imp, dist, Walloc[i_start:i_end+1], env.ChannelType, env.ChannelInfo, OutputType, Use_History, Use_dist)#scalar
        elif OutputType == "Partial_direvatives":
            output = np.zeros_like(Walloc)
            output[i_start:i_end+1] = ProbBranch * self.OneUser_Reward(server, data, imp, dist, Walloc[i_start:i_end+1],  env.ChannelType, env.ChannelInfo, OutputType, Use_History, Use_dist)
        if i_end+1 < self.Tsteps_greedy:
            i_start = i_end+1
            for c in range(env.Nclasses):
                maxLat, data, imp, prob_class = env.Classes[c] 
                output += self.OneServer_Reward( env, server, prob_class*ProbBranch, i_start, min(i_start+maxLat-1,self.Tsteps_greedy-1), data, imp, dist, Walloc, OutputType)
        return output


    def Total_Reward(self, env, Wvar, OutputType='Value'):
        # ==>When OuputType == 'Value':
        #    Returns the total reward (aggregating the expected reward of all the "servers")
        # ==>when OutputType == 'Partial_direvatives':
        #    Returns the direvarives with respect to every scheduled bandwidth.
        
        output = np.zeros_like(Wvar) if (OutputType=='Partial_direvatives') else 0
        
        for server in range(env.Kusers):
            maxLat = env.StateUsers[self.sample]['lat'][server]
            data = env.StateUsers[self.sample]['data'][server]
            imp = env.StateUsers[self.sample]['imp'][server]  
            dist = env.StateUsers[self.sample]['dist'][server]       
            if OutputType == 'Value':
                output += self.OneServer_Reward(env, server, 1, 0, min(maxLat-1,self.Tsteps_greedy-1), data, imp, dist, Wvar[server,:], OutputType)
            elif OutputType == 'Partial_direvatives':
                output[server,:] = self.OneServer_Reward(env, server, 1, 0, min(maxLat-1,self.Tsteps_greedy-1), data, imp, dist, Wvar[server,:],  OutputType)
        return output     
                




def BaselineTest_FrankWolfeOpt(hyperparameters, env, Resources, Geometry, ChannelStat):    
    assert( env.CSIestimation == "Statistical" )
    if env.N_parallel_env>20:
        print("Warning... too big N_parallel_env, it will take time")
    
    Scheduler = FrankWolfeBaseline(hyperparameters, env, Resources, Geometry, ChannelStat)
    TestMethods.test_Scheduler("FrankWolfe", Scheduler, env)