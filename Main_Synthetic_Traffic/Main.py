import os
import sys
import environment as env
import KnapsackBaseline
import OracleIntLPbaseline
import FrankWolfeBaseline
import DeepScheduler
from torch import cuda




#~~~~~~HYPER-PARAMETERS OF ALGORITHMS ~~~~~~~~~
def Define_HyperParameters(Method):
    '''Here for each algorithm the necesssary hyperparameters can be chosen. In general:

    -'N_parallel_env_test': Defines the number of independent experiments that are simultaneously run. The performance 
                            is measured as the average over all those experiments.
    -'OracleTimeSteps': Defines how many timesteps ahead the algorithm knows using an oracle the future traffic 
                        characteristics. In paper  it is the horizon T.(If it is 0 then ILP coincides to 'Knapsack' solution)
    -'AssumeChannel': The way channel of a user is assumed to evolve per time slots. Choices: "iid","Constant"
                      Carefull this affects only the math used to define the objective function and doesn't determine 
                      the real channel a user experiences. "iid" corresponds to \rho=0 in the paper and "Constant"
                      to \rho=1
    -'T_greedysteps': Defines how many steps in the future are considered in the objective function that the Frank-Wolfe
                      approach considers. In paper it is the horizon T.
    -'N_DifferentInitializations': There are a lot of local optimums. Frank-Wolfe converges to one of them. Because it 
                                   may not be a good point we run the algorithm N_DifferentInitializations times and 
                                   choose the best one. In paper it is N_{init}
    -'InitializationType': For each of the "N_DifferentInitializations' runs the algorithm must start from a random point.
                           This random initial point is generated by some random distribution. The distributions we 
                           implemented here from which we can get some inital point were Uniform, Gamma and Half-normal.
                           The form to choose them is: ["Uniform"] or ["absGaussian", mean, scale] or ["Gamma",0.3,1].
                           Good choice is ["Gamma",0.3,1].
    -'MaxSteps': The maximum number of iterations we allow the FrankWolfe to run until reaches a local optimum. It 
                 converges very fast in general.
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
            'N_parallel_env_test':  3
        }
    elif Method == "IntegerLPoracle":
        hyper_param ={
            'N_parallel_env_test':  2,
            'OracleTimesteps': 6
        }
    elif Method == "FrankWolfe":
        hyper_param = {
            'N_parallel_env_test': 2,
            'AssumeChannel': "iid",
            'T_greedysteps': 4, 
            'N_DifferentInitializations':  20, 
            'InitializationType':["Gamma",0.3,1] ,
            'MaxSteps': 15,
        }
    elif Method == 'DeepScheduler':  
        hyper_param = {
            'device': "cpu", #not tested on GPU
            
            #Architecture of NNs  
            'N_Quantiles': 50,
            'MemorizedActions': 0, 

            #Training hyperparameters
            'TradeOff_between_lossesCritic': 1.0, 
            'TradeOff_reward_punish':1.0,   
            'Gamma': 0.95,    
            'BatchSize': 64,
            'CapacityPool': 5*1000, 
            'LearningRate_Critic': 1e-3,   
            'LearningRate_Actor' : 1e-3,    
            'SyncProcess_act': 0.5e-2,    
            'SyncProcess_crt': 0.5e-2,
            'ExploreProbability':0.25, 
            'N_parallel_env_train': 16,  


            #Monitoring progress
            'CreateWriter': True,       
            'AfterSamplesToWrite_train': 10*1000,            
            'AfterSamplesToTest': 20*1000, 
            'N_parallel_env_test': 10,   


            #Termination   
            'MaxIter': 1000*1000, 
            'MaxNsamples': 6*1000*1000
        }
    return hyper_param        






if __name__ == "__main__":
#~~~~~ SYSTEM PARAMETERS ~~~~~~~
    #PROTOCOL
    PROTOCOL = {
        'CSIestimation' : "Full", #Choice between "Full" or "Statistical"
        'RetransProtocol' : "HARQ-type I" #For the moment the only choice 
        }

    #GEOMETRY, Defining the disk the base station is covering
    GEOMETRY = {
        'Rmin' : 0.05,#in Km 
        'Rmax' : 1.0
        }   
    
    #CHANNEL
    CHANNEL = {
        'ChannelType': "ExpMarkovTimeCorr", #For the moment the only choice       
        'ChannelInfo': 0.0, #   For "ExpMarkovTimeCorr" is  "r" in the equation : h_t = r*h_{t-1} + CN(0,sqrt(1-r^2)/2), where
                            #   CN(0,sigma) is a complex gaussian random variable with real and imaginary part being
                            #   independent with variance=sigma^2
        'PathLoss' : 3.7,
        'ConstLoss_div_Noise' : 204 #ConstLoss/sigma_noise^2 ... (for example 10^(-0.1*120.9)/(-114dbm) = 204 )
        }

    #TRAFFIC
    TRAFFIC = {    
        'ClassesInfo':  [{'Max_Latency': 2, 'Data':32*8*256, 'Importance':1.0, 'Arrival_Prob': 0.2}, 
                         {'Max_Latency': 10, 'Data':32*8*2048, 'Importance':1.0, 'Arrival_Prob': 0.3},],   
        'Kusers': 100
        }
    p_null = 1-sum( [cl['Arrival_Prob'] for cl in TRAFFIC['ClassesInfo'] ])
    assert p_null > 0, "In every time slot there must be some probability that now new arrival happens."
    TRAFFIC['ClassesInfo'].append({'Max_Latency':1, 'Data':0, 'Importance':0, 'Arrival_Prob':p_null})

    #RESOURCES
    RESOURCES = {
        'BW' :5.0e6 , #in Hz
        'EnergyPerSymbol' : 1.0/320  #energy per symbol per Hz. We assumed no Power allocation but it can be added as the
                                     #environment is designed so as to accept users being allocated with different powers
        }  

    #Seed controling the environment randomness
    EnvSEED = None 


#~~~~~~~ METHOD/algorithm ~~~~~~~~~
    methodAvailable = ["Knapsack", "IntegerLPoracle", "FrankWolfe", "DeepScheduler"]
    MethodUsed = methodAvailable[3] #Define your method
    LOAD_MODEL =  False#Choose if you want to load the NN model. If true you will be asked to enter the name of the model
    LOAD_MODEL = LOAD_MODEL if MethodUsed in ["DeepScheduler"] else None
     

#~~~~~~ RUNNING THE ALGORITHMS ~~~~~~~~~        
    #~~~ Creating the test-environment & Printing general Info ~~~~
    HYPER_PARAMETERS = Define_HyperParameters(MethodUsed)
    env_test  = env.Environment( PROTOCOL, GEOMETRY, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'test')  
    print("Method Used:  ", MethodUsed, "\nChannel corr.: ", CHANNEL['ChannelInfo'])
    print("BW: ", RESOURCES['BW'], "/// Power: ", RESOURCES['EnergyPerSymbol'])
    print("Protocol Used: ", PROTOCOL["CSIestimation"])

    #~~~~~ Run the algortihms and test them ~~~~~
    if MethodUsed == "Knapsack":     
            KnapsackBaseline.BaselineTest_Knapsack(env_test, RESOURCES)                   


    elif MethodUsed == "IntegerLPoracle":
        OracleIntLPbaseline.BaselineTest_intLPoracle(env_test, RESOURCES)


    elif MethodUsed == "FrankWolfe":  
        FrankWolfeBaseline.BaselineTest_FrankWolfeOpt(HYPER_PARAMETERS, env_test, RESOURCES, GEOMETRY, CHANNEL)
                

    elif MethodUsed == "DeepScheduler":
        #Creating a tag Name with which we will save the NNs and for TensorBoard naming curves
        TagName = 'example'#Put the Tag Name you desire


        #Define Saving path and inform for possible overwritting
        save_path = os.path.join("saves", MethodUsed) 
        if os.path.exists(save_path) and LOAD_MODEL:
            print("Model if there is with tag name: ", TagName)
            print("will be loaded and if its performance is improved previous version will be overwritten.")
        else:
            print("Be sure that a useful model will not be lost due to same TagName")
            os.makedirs(save_path, exist_ok=True)

       
        #Run    
        env_train = env.Environment( PROTOCOL, GEOMETRY, CHANNEL, TRAFFIC, RESOURCES, HYPER_PARAMETERS, 'train',  EnvSEED)
        DeepScheduler.train_test_DeterDistribDueling(env_train, env_test, RESOURCES, HYPER_PARAMETERS, save_path, LOAD_MODEL, TagName )

