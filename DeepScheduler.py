import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import collections
import numpy as np
import copy
import os
import sys
import TestMethods
import Modules_and_Models as Models


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

#======================================= 
#++++++ POOL and TENSORise INPUT +++++++
#=======================================
Experience = collections.namedtuple('Experience', field_names=['state','action','reward','new_state'])
class ExperiencePool:
    def __init__(self, capacity, gamma):
        self.pool = collections.deque(maxlen=capacity)#The oldest samples first to be removed
        self.moment = 1e-4
        self.gamma = gamma
        self.Returns = 0
        self.MeanReturn = 0
        self.MeanOfsquaresReturns = 0
    
    def __len__(self):
        return len(self.buffer)

    def append(self, states, actions, rewards, next_states):
        #Reward Scaling
        self.Returns = self.gamma*self.Returns + rewards
        self.MeanReturn = torch.mean(rewards)*self.moment + (1-self.moment)*self.MeanReturn
        self.MeanOfsquaresReturns = torch.mean(rewards**2)*self.moment + (1-self.moment)*self.MeanOfsquaresReturns
        sigma = torch.sqrt(self.MeanOfsquaresReturns-self.MeanReturn**2)
        rewards = (rewards - self.MeanReturn)/sigma
        #Add the experience in the pool
        for state, action, reward, next_state in zip(states, actions, rewards, next_states):
            self.pool.append( Experience(state, action, reward, next_state) )    
    
    def sample(self, batch_size):
        ind = torch.randint(len(self.pool), (batch_size,))#there is replacement in the sampling...
        states, actions, rewards, next_states = zip(*[self.pool[i] for i in ind])
        return torch.stack(states),torch.stack(actions), torch.stack(rewards), torch.stack(next_states)    
    

 

#=============================================        
#+++++++++ SYNC, TEST, SAVE functions ++++++++
#=============================================    
def SyncNetwork(syncTargetNN, net_target, net, iter_train):
    if isinstance(syncTargetNN, float) and (0.0 < syncTargetNN < 1.0):#shmuma (github)
        crt_tgt_state = net_target.state_dict()
        for key, value in net.state_dict().items():
            crt_tgt_state[key] = crt_tgt_state[key] * (1-syncTargetNN) + syncTargetNN * value
        net_target.load_state_dict(crt_tgt_state) 
    elif isinstance(syncTargetNN, int) and iter_train % syncTargetNN == syncTargetNN-1:
        net_target.load_state_dict(net.state_dict())


def Test_SaveIfGood(act_net, env_test, Resources, writer_loss, Nsamples, BestPerformance, device):
    with torch.no_grad():
        env_test.ResetEnv()
        ExpRewardPerUser, SuccessRatesPerClass = TestMethods.test_Scheduler("DeepScheduler", act_net , env_test, Resources, device, False)
        #TensorBoard
        if (writer_loss is not None) and (SuccessRatesPerClass is not None):               
            MeanProbSatisfaction = 0
            Prob_notNull = 0
            for indUserClass in range(env_test.Nclasses):#LAST class MUST the "0-class"    
                if env_test.Classes['data'][indUserClass] == 0 or env_test.Classes['imp'][indUserClass] == 0: continue 
                writer_loss.add_scalar("Test/Class_"+str(indUserClass), SuccessRatesPerClass[indUserClass], Nsamples)
                MeanProbSatisfaction += env_test.Classes['prob'][indUserClass] * SuccessRatesPerClass[indUserClass]
                Prob_notNull += env_test.Classes['prob'][indUserClass] 
            writer_loss.add_scalar("Test/Mean_SatisfactionRate", MeanProbSatisfaction/Prob_notNull, Nsamples)

        #Print if a better NN found and Save it
        if (ExpRewardPerUser > BestPerformance) and (SuccessRatesPerClass is not None):        
            BestPerformance = ExpRewardPerUser
            print("=============================================")
            print("Best Performance: ", ExpRewardPerUser, "  after samples: ",Nsamples)
            for indUserClass in range(env_test.Nclasses):                    
                if env_test.Classes['imp'][indUserClass] == 0: continue
                print("Class",indUserClass," ->ProbSuccess: ",SuccessRatesPerClass[indUserClass])            
            print("Previous saved model will be substituted")
            print("=============================================\n")
            sys.stdout.flush()
    return ExpRewardPerUser 


def SaveNN(act_net, crt_net, act_path, crt_path):    
    torch.save(act_net, act_path)   
    torch.save(crt_net, crt_path) 




#================================        
#+++++++++ MAIN PART ++++++++++++
#================================
def train_test_DeterDistribDueling(env_train, env_test, Resources, HyperParameters, save_path, LoadModel, TagName=None ):  
#~~~~~~ SET PARAMETERS ~~~~~~~~
    device = HyperParameters['device'] 

    #Architecture of NNs
    N_MemorizedPrevActions = HyperParameters['MemorizedActions']
    N_quant = HyperParameters['N_Quantiles']
    cdf_arange = torch.Tensor((2 * np.arange(N_quant) + 1) / (2.0 * N_quant)).view(1, N_quant).to(device)
    
    #Training Hyperparameters
    rho_losses = HyperParameters['TradeOff_between_lossesCritic']  
    rho_rewards = HyperParameters['TradeOff_reward_punish']
    gamma = HyperParameters['Gamma']  
    BatchSize = HyperParameters['BatchSize']
    CapacityPool = HyperParameters['CapacityPool']
    lr_Critic, lr_Actor = HyperParameters['LearningRate_Critic'], HyperParameters['LearningRate_Actor']        
    syncTargetNN_act, syncTargetNN_crt = HyperParameters['SyncProcess_act'], HyperParameters['SyncProcess_crt'] 
    ExplorationProb = HyperParameters['ExploreProbability']

    #Monitoring progress
    writer_loss = SummaryWriter(comment=TagName) if HyperParameters['CreateWriter'] else None
    Dsamples_writerTrain =  HyperParameters['AfterSamplesToWrite_train']
    N_writerPoints = 0
    Dsamples_Test =  HyperParameters['AfterSamplesToTest'] 
    N_testPoints = 0   
    MaxIters, MaxNsamples  = HyperParameters['MaxIter'], HyperParameters['MaxNsamples'] 


    Delayed = 2 if HyperParameters['TD3'] else 1
    rho_losses = 0.0 if HyperParameters['TD3'] else HyperParameters['TradeOff_between_lossesCritic']


   
#~~~~~~ Create/load NN, Optimizer, Pool ~~~~~~~~
    BwComputed = 1 if env_train.CSIestimation == "Full" else 0 
    InputStateInfo = [ len(env_train.InputFeaturesType)+BwComputed+N_MemorizedPrevActions, env_train.Kusers, N_MemorizedPrevActions, env_train.CSIestimation] 
    BestPerformance = float("-inf") 
    N_ActionFeatures = 1 #1 if BW only is allocated, 2 if BW and Power is allocated (the second not yet implemented)
    act_path, crt_path = save_path + '/ACTOR_' + TagName + '.pt', save_path + '/CRITIC_' + TagName + '.pt'
    if os.path.exists(act_path) and LoadModel: 
        act_net=torch.load(act_path).to(device)
        crt_net=torch.load(crt_path).to(device)
    else:
        act_net = Models.Actor(InputStateInfo, Resources, None).to(device)    
        crt_net = Models.DistrQN(InputStateInfo, N_ActionFeatures, Resources, N_quant).to(device)     
        if HyperParameters['TD3']:
            crt_net2 = Models.DistrQN(InputStateInfo, N_ActionFeatures, Resources, N_quant).to(device)     
    act_net_target = copy.deepcopy(act_net).to(device)
    crt_net_target = copy.deepcopy(crt_net).to(device) 
    if HyperParameters['TD3']:
        crt_net_target2 = copy.deepcopy(crt_net2).to(device)  
    act_opt = optim.Adam(act_net.parameters(), lr=lr_Actor, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    crt_opt = optim.Adam(crt_net.parameters(), lr=lr_Critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  
    if HyperParameters['TD3']:
        crt_opt2 = optim.Adam(crt_net2.parameters(), lr=lr_Critic, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 
    LossMSE = nn.MSELoss()
    PoolOfSamples = ExperiencePool(CapacityPool, gamma)



#~~~~~~ MAIN BODY OF TRAINING ~~~~~~~~    
    for iter_train in range(MaxIters):
        if (iter_train*BatchSize > MaxNsamples): break


    
    #~~~~~ Interact with environment and add SAMPLES TO POOL ~~~~~
        with torch.no_grad():
            if iter_train == 0:
                if N_MemorizedPrevActions > 0:
                    Memory_PrevActions = collections.deque( maxlen=N_MemorizedPrevActions )
                    for _ in range(N_MemorizedPrevActions): 
                        Memory_PrevActions.append(torch.zeros((env_train.N_parallel_env,env_train.Kusers),device=device))
                else:
                    Memory_PrevActions = None
                StatesUsers = Models.EnvironmentState_2_InputNN(env_train, device, Memory_PrevActions) 
            Explore = np.random.rand() <= ExplorationProb 
            bw_alloc, values = act_net(StatesUsers,Explore)                
            Walloc =  bw_alloc.data.cpu().numpy()
            PosRewards, NegRewards = env_train.Step(Walloc)  
            if N_MemorizedPrevActions > 0:
                Memory_PrevActions.append(bw_alloc)
                #assert(list(bw_alloc.shape)==[env_train.N_parallel_env,env_train.Kusers])
                for prevActions in Memory_PrevActions:
                    prevActions.mul_(~torch.from_numpy(env_train.NewUsersInd).to(device))
            rewards = rho_rewards*PosRewards - NegRewards*(1-rho_rewards)                
            next_StatesUsers = Models.EnvironmentState_2_InputNN(env_train, device, Memory_PrevActions) 
            PoolOfSamples.append(StatesUsers, values, torch.from_numpy(rewards).float().to(device), next_StatesUsers)
            StatesUsers = next_StatesUsers.detach().clone()         


    #~~~~~ Training ~~~~~~~
        if iter_train <40 : continue #to fill a bit the Pool at the beginning
        states, actions, rewards, next_states = PoolOfSamples.sample(BatchSize)

        #Critic            
        crt_opt.zero_grad()
        if HyperParameters['TD3']:
            crt_opt2.zero_grad()
        left_quantiles, left_value,  Distr_Mean_Zero = crt_net(states, actions) 
        if HyperParameters['TD3']:
            _,left_value2,_ = crt_net2(states,actions)
        assert(list(left_quantiles.shape) == [BatchSize, N_quant])
        _, actions_target = act_net_target(next_states)
        right_nextQuantiles, right_nextValue, _ = crt_net_target(next_states, actions_target.detach() ) 
        if HyperParameters['TD3']:
            _,right_nextValue2,_ = crt_net_target2(states,actions)
        right_quantiles = rewards.view(-1,1) + gamma*right_nextQuantiles.detach()  
        if HyperParameters['TD3']:
            right_value = rewards +  gamma*torch.min(right_nextValue.detach(),right_nextValue2.detach())
        else:
            right_value = rewards + gamma*right_nextValue.detach()
        diff_distr = right_quantiles.t().unsqueeze(-1) - left_quantiles 
        assert(list(diff_distr.shape) == [N_quant, BatchSize, N_quant])
        loss_distr = diff_distr * (cdf_arange - (diff_distr.detach()<0).float())
        #loss_distr = huber(diff_distr) * (cdf_arange - (diff_distr.detach()<0).float())
        loss_value = LossMSE( left_value, right_value)
        loss =  rho_losses*loss_distr.mean() + (1-rho_losses)*loss_value +Distr_Mean_Zero**2
        loss.backward()
        nn.utils.clip_grad_norm_(crt_net.parameters(), 0.2)
        crt_opt.step() 
        if HyperParameters['TD3']:
            loss = LossMSE(left_value2,right_value)
            loss.backward()
            crt_opt2.step()

        #Actor  
        if iter_train%Delayed ==0:#For delayed training
            act_opt.zero_grad()
            _, actions_train_act = act_net(states)
            _, crt_DiscountedRewards, _ = crt_net(states, actions_train_act) 
            actor_loss = -crt_DiscountedRewards.mean()        
            actor_loss.backward()
            nn.utils.clip_grad_norm_(act_net.parameters(), 0.2)
            act_opt.step()  

        #Synchronise Target NNs
        SyncNetwork(syncTargetNN_act, act_net_target, act_net, iter_train)
        SyncNetwork(syncTargetNN_crt, crt_net_target, crt_net, iter_train) 
                 
    #~~~~~ TensorBoard, Testing and Saving ~~~~~~~
        #Writers for train
        Nsamples = (iter_train+1)*BatchSize       
        if (writer_loss is not None)  and (Nsamples > N_writerPoints*Dsamples_writerTrain): 
            writer_loss.add_scalar("Critic/LossTotal/Train", loss, Nsamples)           
            writer_loss.add_scalar("Critic/LossDistr/Train", loss_distr.mean(), Nsamples)
            writer_loss.add_scalar("Critic/LossMeanValue/Train", loss_value, Nsamples)
            writer_loss.add_scalar("Critic/LossMeanDistr/Train", Distr_Mean_Zero, Nsamples)
            writer_loss.add_scalar("Actor/Loss/Train", actor_loss, Nsamples)           
            N_writerPoints += 1         

        #test and save
        if (Nsamples >= N_testPoints*Dsamples_Test):
            N_testPoints += 1
            TestPerformance = Test_SaveIfGood(act_net, env_test, Resources, writer_loss, Nsamples, BestPerformance, device)
            if TestPerformance > BestPerformance:
                BestPerformance = TestPerformance
                SaveNN(act_net, crt_net, act_path, crt_path)        


    if writer_loss is not None:
        writer_loss.close()        
