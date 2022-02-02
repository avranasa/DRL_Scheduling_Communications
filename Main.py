import os
import sys
import environment as env
import KnapsackBaseline
import OracleIntLPbaseline
import ExponentialRule
import DeepScheduler
from torch import cuda




#~~~~~~HYPER-PARAMETERS OF ALGORITHMS ~~~~~~~~~
def Define_HyperParameters(Method):
    '''Here for each algorithm the necesssary hyperparameters can be chosen. In general:

    -'N_parallel_env_test': Defines the number of independent experiments that are simultaneously run. The performance 
                            is measured as the average over all those experiments.
    -'delta': It is a hyper-parameter that the exponential rule uses. Even though 0.01 was used in the original paper here
              for  3,5,10,15MHz  better to use 0.999 and for  20MHz 0.68. ()
    -'OracleTimeSteps': Defines how many timesteps ahead the algorithm knows using an oracle the future traffic 
                        characteristics. In paper  it is the horizon T.(If it is 0 then ILP coincides to 'Knapsack' solution)
    
    -'TD3': If true then a baseline that is built on the basis of TD3 algorithm is used.
    -'MemorizedActions': How many actions of previous time steps to be stored and will be used in the state. For Full CSI
                         (i.e. PROTOCOL['CSIestimation']=="Full") put it to 0. For Statistical CSI (better referred in the
                         paper as no-CSI since only the baseline FrankWolfe has access to the traffic and channel statistics)
                         then make it equal to Max_latency of the class with highest lateny.
    -'TradeOff_between_lossesCritic': With 0 you minimize L2-loss (i.e. DDPG). With 1 you do distibutional (Distributional DDPG). 
                                      If in (0,1), it will use both losses but it is not recommended.
    -'TradeOff_reward_punish': In the code there is both positive reward (when a user is satisfied and negative (when its
                               latency constraint is exceeded without getting the required data). This arguments balances 
                               between the positive rewards and the negative approximately like: "a*PosRewards - NegRewards*(1-a)"
                               In the paper only the case of a=1 is considered (ignoring negative rewards).
    -'SyncProcess_act': If it is in (0,1) then it is the momentum with which the target pocily network is updated.  If it is
                        integer then the target policy network is synced after that number of iterations. $m_{target} in the paper.
    -'SyncProcess_crt': The same as 'SyncProcess_act' but for the target value network.
    -'ExploreProbability': With that probability the actor will randomly (using noisy networks) distort its parameters which 
                           will lead to a different action. In the paper it is '\sigma_{explore}'.
    -'AfterSamplesToWrite_train': After that many training samples it will log some training statistics.       
    -'AfterSamplesToTest': After 'AfterSamplesToTest'/'BatchSize' iterations the agent stops to be trained and tested. If not 
                          interested in such procedure increase its value more than 'MaxNsamples'.
    -'MaxIter'/'MaxNsamples': The training procedure will stop after either more than 'MaxNsamples' the agent has been trained
                              or more than 'MaxIter' iterations passed.                         

    '''
    if Method == 'Knapsack':
        hyper_param = {
            'N_parallel_env_test':  1
        }
    elif Method == 'ExpRule':
        hyper_param = {
            'N_parallel_env_test':  3,
            'delta': 0.999
        }
    elif Method == "IntegerLPoracle":
        hyper_param ={
            'N_parallel_env_test':  2,
            'OracleTimesteps': 3
        }
    elif Method == 'DeepScheduler':  
        hyper_param = {
            'device': "cpu", #not tested on GPU
            #Architecture of NNs  
            'TD3': False,
            'N_Quantiles': 50,
            'MemorizedActions': 0, 

            #Training hyperparameters
            'TradeOff_between_lossesCritic': 1.0, 
            'TradeOff_reward_punish':1.0,   
            'Gamma': 0.95,    
            'BatchSize': 32,
            'CapacityPool': 5*1000, 
            'LearningRate_Critic': 10e-4,   
            'LearningRate_Actor' : 10e-4,    
            'SyncProcess_act': 0.2e-2,  
            'SyncProcess_crt': 0.2e-2,
            'ExploreProbability':0.25, 
            'N_parallel_env_train': 16,  


            #Monitoring progress
            'CreateWriter': True,       
            'AfterSamplesToWrite_train': 10*1000,            
            'AfterSamplesToTest': 20*1000,
            'N_parallel_env_test': 10,   


            #Termination   
            'MaxIter': 1000*1000, 
            'MaxNsamples': 20*1000*1000
        }
    return hyper_param        






if __name__ == "__main__":
#~~~~~ SYSTEM PARAMETERS ~~~~~~~
    #PROTOCOL
    PROTOCOL = {
        'CSIestimation' : "Full", #Choice between "Full" or "Statistical"
        'RetransProtocol' : "HARQ-type I", #For the moment the only choice 
        'TimeSlot': 1e-3
        }

    #CHANNEL
    CHANNEL = {
        'ChannelType': "ExpMarkovTimeCorr",
        }


    #TRAFFIC
    TRAFFIC = {    
        'ClassesInfo':          [{'Max_Latency': 5, 'Data':1000, 'Importance':1.0, 'Arrival_Prob': 0.2}, 
                                {'Max_Latency': 25, 'Data':5*1000, 'Importance':1.0, 'Arrival_Prob': 0.3},],   
        'Kusers': 100,
        'ProbTypeUser':         {'bicycle':0.2, 'bus':0.2, 'car':0.2, 'foot':0.2, 'train':0.2, 'tram':0.0}  
        }
    p_null = 1-sum( [cl['Arrival_Prob'] for cl in TRAFFIC['ClassesInfo'] ])
    assert p_null > 0, "In every time slot there must be some probability that now new arrival happens."
    TRAFFIC['ClassesInfo'].append({'Max_Latency':1, 'Data':0, 'Importance':0, 'Arrival_Prob':p_null})


    #RESOURCES
    RESOURCES = {
        'BW' : 3.0e6   ,#in Hz
        'ResourceBlock': 0.2e6# Must be a divisor/factor of BW
        }  

#~~~~~~~ METHOD/Algorithm ~~~~~~~~~  
    methodAvailable = ["Knapsack","ExpRule", "IntegerLPoracle",  "DeepScheduler"]
    MethodUsed = methodAvailable[0] #Define your method
    if MethodUsed in ["DeepScheduler"]:
        LOAD_MODEL = False #Choose if you want to load the NN model. If true you will be asked to enter the name of the model
    else: 
        LOAD_MODEL = None


#~~~~~~ RUNNING THE ALGORITHMS ~~~~~~~~~       
    #~~~ Creating the test-environment & Printing general Info ~~~~
    HYPER_PARAMETERS = Define_HyperParameters(MethodUsed)
    env_test  = env.Environment( PROTOCOL, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'test')  
    

    #~~~~~ Run the algortihms and test them ~~~~~
    if MethodUsed == "Knapsack":  
        env_test  = env.Environment( PROTOCOL, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'test')  
        print("Method Used:  ", MethodUsed,)
        print("BW: ", RESOURCES['BW'])
        print("Protocol Used: ", PROTOCOL["CSIestimation"])   
        KnapsackBaseline.BaselineTest_Knapsack(env_test, RESOURCES)     

    elif MethodUsed == "ExpRule":
        env_test  = env.Environment( PROTOCOL, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'test')  
        print("Method Used:  ", MethodUsed,)
        print("BW: ", RESOURCES['BW'])
        print("Protocol Used: ", PROTOCOL["CSIestimation"])        
        ExponentialRule.BaselineTest_ExponentialRule(env_test, RESOURCES, HYPER_PARAMETERS)       


    elif MethodUsed == "IntegerLPoracle":
        env_test  = env.Environment( PROTOCOL, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'test')  
        print("Method Used:  ", MethodUsed,)
        print("BW: ", RESOURCES['BW'])
        print("Protocol Used: ", PROTOCOL["CSIestimation"])  
        OracleIntLPbaseline.BaselineTest_intLPoracle(env_test, RESOURCES)
            

    elif MethodUsed == "DeepScheduler":
        #Creating a tag Name with which we will save the NNs and for TensorBoard naming curves
        TagName = 'example'#Put the Tag Name you desire


        #Define Saving path and inform for possible overwritting
        save_path = os.path.join("saves", MethodUsed) 
        if os.path.exists(save_path) and LOAD_MODEL:
            print("Model if there is with tag name: ", TagName)
            print("will be loaded and if its performance is improved previous version will be overwritten. Press Enter")
            
        else:
            print("Be sure that a useful model will not be lost due to same TagName")
            os.makedirs(save_path, exist_ok=True)
        print('HyperParam', HYPER_PARAMETERS)
        print('Resources', RESOURCES)
       
        #Run    
        env_train = env.Environment( PROTOCOL, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'train')
        DeepScheduler.train_test_DeterDistribDueling(env_train, env_test, RESOURCES, HYPER_PARAMETERS, save_path, LOAD_MODEL, TagName )

