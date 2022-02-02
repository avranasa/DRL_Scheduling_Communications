import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.module import Module
import numpy as np
from collections import deque
import math

#================================================
#++ Prepare Environment state as input for NNs ++
#================================================
def EnvironmentState_2_InputNN(env, device, PreviousActions=None):
    #full CSI: ASSUMPTION THAT THE LAST IS THE REQUIRED bw AND THE PREVIOUS imp 
    #No CSI: ASSUMPTION THAT THE LAST IS imp
    #it will also take previous action and concatenate with states, both in testMethods and Training will be a deque storing previous actions
    BW, N_blocks, bw_block = env.BW, env.N_blocks, env.BwBlock  
    FullCSI = False
    MaxData = np.max(env.Classes['data']) #The Number of Bytes the class wanting the most 
    for featureName in env.InputFeaturesType.names:
        if featureName == 'lat':
            lat = env.StateUsers[featureName]
        elif featureName == 'data':
            dat = env.StateUsers[featureName]
        elif featureName == 'imp' :
            imp = env.StateUsers[featureName]
        elif featureName == 'meanSNR':
            FullCSI = True
            meanSNR = env.StateUsers[featureName]
        elif featureName == 'rho':
            rho = env.StateUsers['rho']
        elif featureName == 'absH_2':
            absH_2 = env.StateUsers[featureName]
    if FullCSI:       
        Weights_float =  dat/np.maximum(np.log2(1 + absH_2 * meanSNR )*env.TimeSlot, 1e-6 )+1#1Hz more 
        blocks_needed = np.ceil(Weights_float/bw_block)#1Hz more 
        state = torch.FloatTensor( np.stack( (lat, absH_2, rho, meanSNR, dat/MaxData, imp, blocks_needed/N_blocks), axis=1) ).to(device)
    else:
        #Case of Statistical CSI
        # PreviousActions contains the previous actions as a deque containing Tensors (for every previous timestep)
        currentState = torch.FloatTensor( np.stack( (lat, dat/MaxData, imp), axis=1) ).to(device)
        if PreviousActions is not None:
            previousActions = torch.stack(tuple(PreviousActions), dim=1)
            state = torch.cat((currentState,previousActions/BW), dim=1)
        else:
            state = currentState
    return state


#==================================================
#++ modules to build the actor and critic models ++
#==================================================
class AllocatingLayer(Module): 
    '''The actor NN base its output for the case of full CSI  on a continuous relaxation of the problem. Specifically it gives
    a value for every user. This layer will start allocating to the most valuable bw until no more resources are available for 
    the least valuable users
    '''

    def __init__(self, Resource):
        super(AllocatingLayer, self).__init__()
        self.Nblocks = Resource


    def forward(self, values, weights):
        batchSize, Kusers = values.shape
        assert( list(weights.size()) == [batchSize, Kusers]  and (values>=0).all())
        #Compare everu user value with the rest. A binary matrix Kusers*Kusers shows at a cell i*j if j user is more valuable than i.       
        VperW_diff = values.unsqueeze(dim=1).detach() - values.unsqueeze(dim=2).detach()
        assert( list(VperW_diff.shape) == [batchSize, Kusers, Kusers]  )
        Better_j_than_i = 1.0* (VperW_diff >=0)
        #A vector of Kusers shows for every user if there are enough resources to satisfy it
        Satisfying_Constr = (self.Nblocks - torch.matmul(Better_j_than_i, weights.unsqueeze(dim=2)).squeeze() )>=0
        assert( list(Satisfying_Constr.shape) == [batchSize, Kusers] )
        alloc = Satisfying_Constr*weights
        if torch.all(torch.sum(alloc, axis=1)>self.Nblocks):
            import pdb; pdb.set_trace()
        return alloc


class GraphConvLayer(Module):
    '''Implementing a layer satisfying permutating equivariance between K components with each having In_feat_dim attributes/features'''

    def __init__(self, in_feat_dim, out_feat_dim, K, typeLayer):
        super(GraphConvLayer, self).__init__()
        self.K = K
        self.type = typeLayer
        self.register_buffer('adj', (torch.ones(K, K) - torch.eye(K))*  1.0/self.K)
        self.register_buffer('direct_adj', torch.eye(K) - (torch.ones(K, K) - torch.eye(K))*  1.0/(self.K-1))
        self.register_buffer('adj_v2', torch.ones(K, K)* 1.0/self.K)
        self.bias = nn.parameter.Parameter(torch.zeros(1,out_feat_dim))
        self.linear_others = nn.Linear(in_feat_dim, out_feat_dim, bias=False)
        self.linear_self = nn.Linear(in_feat_dim, out_feat_dim, bias=False)

    def forward(self, x):
        if self.type == 'simple':
            #As in the paper
            feat_others = torch.matmul(self.adj_v2,  self.linear_others(x))
            feat_self = self.linear_self(x)
            out = feat_others+feat_self       
        elif self.type == 'simple_simple':
            #Works sligtly better than 'simple'
            feat_others = torch.matmul(self.adj, self.linear_others(x))
            feat_self = self.linear_self(x)
            out = feat_others+feat_self
        elif self.type == 'simple_direct':
            out = torch.matmul(self.direct_adj, self.linear_others(x))+self.bias
        elif self.type == 'simple_relu':
            feat_others = torch.matmul(self.adj, self.linear_others(x))
            feat_self = self.linear_self(x)
            out = feat_others+F.relu(feat_self)
        elif self.type == 'relu_relu':
            feat_others = torch.matmul(self.adj, self.linear_others(x))
            feat_self = self.linear_self(x)
            out = F.relu(feat_others)+F.relu(feat_self)   
        elif self.type == 'tanh_tanh':
            feat_others = torch.matmul(self.adj, self.linear_others(x))
            feat_self = self.linear_self(x)
            out = torch.tanh(feat_others) + torch.tanh(feat_self)     
        return out




#==========================
#++++++++ Actor +++++++++++
#==========================
class Actor(nn.Module):
    def __init__(self, InputStateInfo, Resources, ExploreStatInfo):
        super(Actor, self).__init__() 
        self.BW = Resources['BW']  
        self.BwBlock = Resources['ResourceBlock']
        self.N_blocks = np.ceil(self.BW / self.BwBlock)#the ceil to avoid float number errors
        self.CSIestimation = InputStateInfo[3] #Full or Statistical
        self.N_MemorizedPrevActions = InputStateInfo[2]        
        self.Kusers = InputStateInfo[1]
        self.N_in_Features = InputStateInfo[0] 
                
        self.conv1 = nn.Conv1d(self.N_in_Features, 10, 1)
        self.conv2 = nn.Conv1d(10, 10, 1)
        self.softplus = nn.Softplus()
        self.PermutLayer0 = GraphConvLayer(10, 10, self.Kusers, 'simple')
        self.PermutLayer1 = GraphConvLayer(10, 1, self.Kusers, 'simple')  
        self.Allocator = AllocatingLayer(self.N_blocks)


    def forward(self, state, explore=False):           
        if self.CSIestimation == "Full":
            bw_blocks_needed = state[:,-1,:] * self.N_blocks#ASSUMPTION THAT THE LAST IS THE REQUIRED bw AND THE PREVIOUS imp
            mask_activeUsers = (state[:,-2,:]>0) #users of positive importance
        elif self.CSIestimation == "Statistical":
            mask_activeUsers = (state[:,2,:]>0) #users of positive importance
        if explore:
            #perturbate weights
            snr_noise_1 = 0.2
            snr_noise_2 = 0.3
            noise_1 = snr_noise_1 * self.conv1.weight.data * torch.randn_like(self.conv1.weight.data)
            #noise_1 = snr_noise_1 * torch.randn_like(self.conv1.weight.data)
            self.conv1.weight.data += noise_1
            noise_2 = snr_noise_2 * self.conv2.weight.data * torch.randn_like(self.conv2.weight.data)
            #noise_2 = snr_noise_2 * torch.randn_like(self.conv2.weight.data)
            self.conv2.weight.data += noise_2
            #Run the NN
            x = self.conv1(state)
            x = F.relu(x)                
            x = self.conv2(x) 
            #weights back to normal
            self.conv1.weight.data -= noise_1
            self.conv2.weight.data -= noise_2
        else:
            x = self.conv1(state)
            x = F.relu(x)  
            x = self.conv2(x)
        #x = F.relu6(x)
        x = torch.sigmoid(x)
        x = self.PermutLayer0(x.transpose(1,2))             
        x = self.PermutLayer1(F.relu(x))

        x = x.squeeze() * mask_activeUsers 
        x = x - torch.sum(x,dim=1, keepdim=True)/(torch.sum(mask_activeUsers,dim=1,keepdim=True)+1e-5)
        x = 10 * F.normalize(x, dim=1)
        x = self.softplus(x) * mask_activeUsers
        if self.CSIestimation == "Full":
            #Turn to value per block (The *self.N_blocks works just as a scalining constant)
            x = x*self.N_blocks /(bw_blocks_needed + ~mask_activeUsers*self.BW*10)    #~mask_activeUsers*self.BW*10 -> To avoid division with 0  
            #Turn to value per block (The *self.N_blocks works just as a scalining constant)
            bw_alloc = self.Allocator(x.detach(), bw_blocks_needed) 
        elif self.CSIestimation == "Statistical":                                
            x = x/(torch.sum(x,dim=1, keepdim=True) + 1e-8)
            x = x * mask_activeUsers# to nullify the chance of dividing previously with 1e-8 and now having some huge value
            bw_alloc = x*self.BW        
        if torch.all(torch.sum(bw_alloc, axis=1)>self.N_blocks):
            import pdb; pdb.set_trace()
        return bw_alloc.detach(), x



#==========================
#++++++++ Critic ++++++++++
#==========================
class DistrQN(nn.Module):    
    def __init__(self, InputStateInfo, action_size, Resources, Nquantiles):
        super(DistrQN, self).__init__()
        self.Kusers = InputStateInfo[1]  
        self.Nquantiles = Nquantiles   
        self.N_in_Features = InputStateInfo[0] + action_size   
        self.conv1 = nn.Conv1d(self.N_in_Features, 10,1)    
        self.conv2 = nn.Conv1d(10, 10,1)        
        self.PermutLayer1 = GraphConvLayer(10, 10, self.Kusers, 'simple_relu')
        #Mean Value
        self.PermutLayer2_value = GraphConvLayer(10, 1, self.Kusers, 'simple_simple')
        #Distribution            
        self.PermutLayer2_distr = GraphConvLayer(10, self.Nquantiles, self.Kusers, 'simple_simple')


    def forward(self, state, action):        
        state_action_in = torch.cat((state, action.unsqueeze(1)),dim=1)
        x_personal = F.relu(self.conv1( state_action_in ))
        x_personal = F.relu6(self.conv2( x_personal )).transpose(1,2)        
        x_total = self.PermutLayer1(x_personal) 

        #Mean Value
        output_value = self.PermutLayer2_value(x_total).squeeze().mean(1)
        
        #Distribution
        output_distr = self.PermutLayer2_distr(x_total).mean(1)
        final_output = output_value.unsqueeze(dim=1) + output_distr - output_distr.mean(1,keepdim=True)
        return  final_output, output_value, output_distr.mean() 