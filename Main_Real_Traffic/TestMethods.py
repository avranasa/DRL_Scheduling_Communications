import numpy as np
import torch
import Modules_and_Models as Models
import collections




def test_Scheduler(NameMethod, ActionScheduler, env, RESOURCES=None, device="cpu", PrintReport=True):
    '''Every method gets some information from the environment and ouput the action Walloc. It is then applied 
    to the environment. Here we measure the performance of each scheduler.'''
    Dt_CheckConvergence = 10 * max(env.Classes['lat'])    
    MaxNsamples = 100*1000 # Described as NumberOfParallelEnv*Timesteps
    SuccessRatesPerClass_old = {}
    TotalPosReward = TotalNegReward = 0
    MoreSteps = True    
    step = 0

    while MoreSteps:
        with torch.no_grad():
            step += 1
            #Apply method of scheduler to get the allocation
            if NameMethod in ['DeepScheduler']: 
                if step == 1: 
                    env.ResetCounter()                      
                    if ActionScheduler.N_MemorizedPrevActions > 0:
                        Memory_PrevActions = collections.deque( maxlen=ActionScheduler.N_MemorizedPrevActions )
                        for _ in range(ActionScheduler.N_MemorizedPrevActions): 
                            Memory_PrevActions.append(torch.zeros(env.N_parallel_env,env.Kusers))
                    else:
                        Memory_PrevActions = None 
                states = Models.EnvironmentState_2_InputNN(env, device, Memory_PrevActions)
                bw_alloc,_ = ActionScheduler(states)
                Walloc =  bw_alloc.data.cpu().numpy()#.data.cpu().numpy()
            elif NameMethod in ["Knapsack", "ExpRule", "IntegerLPoracle", "FrankWolfe","KnapsackCheat"]:
                #if (not step % 200): print (step)
                Walloc = ActionScheduler(env) 
            
            #Allocate and get rewards
            PosReward, NegReward = env.Step(Walloc)  #Dim: N_parallel_env  
            TotalPosReward += np.sum( PosReward )#PosReward: gain from satisfied the current time slot
            TotalNegReward += np.sum( NegReward )#NegReward: missed gain from unsatisfied people who "died" in current time slot
            
            #Memorizing the previous allocation for current users
            if NameMethod in ['DeepScheduler']: 
                if ActionScheduler.N_MemorizedPrevActions > 0:
                    Memory_PrevActions.append(bw_alloc)
                    for prevActions in Memory_PrevActions:
                        prevActions.mul_(~torch.from_numpy(env.NewUsersInd))#We care only for the allocation history of the current users
                
            #Check if Max number of Samples is reached. If yes break
            if env.time*env.N_parallel_env > MaxNsamples:
                print("maximum number of samples reached (uncertain convergence)", env.time * env.N_parallel_env)
                SuccessRatesPerClass = None
                break               
            
            #Check if the probabilities of satisfaction for each class have converged 
            if (env.time % Dt_CheckConvergence) or (len(env.NusersPerClass) < env.Nclasses):#to avoid division by zero 
                continue 

            
            SuccessRatesPerClass = {}
            for UserClass in env.satisfiedPerClass:
                SuccessRatesPerClass[UserClass] = env.satisfiedPerClass[UserClass]/env.NusersPerClass[UserClass]# (number of past and present users who were/are satisfied)/(number of past and present users of a class)
            
            
            if bool(SuccessRatesPerClass_old): #To avoid first run
                MoreSteps = False
                #Check Convergence
                for UserClass in env.satisfiedPerClass:
                    MoreSteps |= abs(SuccessRatesPerClass[UserClass] - SuccessRatesPerClass_old[UserClass]) > 0.001* SuccessRatesPerClass[UserClass]                         
            SuccessRatesPerClass_old = SuccessRatesPerClass            
    
    Total_Data = 0
    for UserClass in env.satisfiedPerClass:
        Total_Data  += env.satisfiedPerClass[UserClass] * env.Classes['data'][UserClass]/env.N_parallel_env
    Mean_Rate = Total_Data / (env.time * env.TimeSlot)
    print("The mean rate from the start of the environment simulation is: ",Mean_Rate/1e6, "Mbits/sec")
    NusersAppeared = sum(env.NusersPerClass.values())#users from the Null-Class (past and current) are also accounted for
    PosRewardPerUser, NegRewardPerUser = TotalPosReward/NusersAppeared, TotalNegReward/NusersAppeared

    #If it is asked the performance of the scheduling method is printed
    if PrintReport: 
        print("\n===========>..........step: ", step, "...........<===========")
        print("Total positive per user reward: ", PosRewardPerUser, " and lost reward: ", NegRewardPerUser)
        relPos , relNeg = [PosRewardPerUser, NegRewardPerUser]/(PosRewardPerUser+NegRewardPerUser)
        print("and the relative positive reward: ", relPos, " and lost reward: ", relNeg)
        print("Success rate per class:")
        for indUserClass in range(env.Nclasses):
            print("    Class", indUserClass,": ", SuccessRatesPerClass[indUserClass])   
            
    return PosRewardPerUser, SuccessRatesPerClass
