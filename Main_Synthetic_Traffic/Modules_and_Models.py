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
def EnvironmentState_2_InputNN(env, device, PreviousActions=None):#it will also take previous action and concatenate with states, both in testMethods and Training will be a deque storing previous actions
    BW, EperSymb = env.BW, env.EperSym   
    for featureName in env.InputFeaturesType.names:
        if featureName == 'lat':
            lat = env.StateUsers[featureName]
        elif featureName == 'data':
            dat = env.StateUsers[featureName]
        elif featureName == 'imp' :
            imp = env.StateUsers[featureName]
        elif featureName == 'dist':
            dist = env.StateUsers[featureName]
        elif featureName == 'absH_2':
            gain = env.StateUsers[featureName]
    minData = np.min(env.Classes['data'][:-1])#Last class is considered the NULL
    if 'gain' in locals():
        #Case of Full CSI
        rate =  np.log2(1 + env.ConstLoss_Noise*gain*(dist**-env.PathLoss)*EperSymb )
        bw_needed = dat / np.maximum(rate, 1e-5) *1.0001 #To avoid imprecisions with floating number
                                                     # we increase slightly the given bw
        state = torch.FloatTensor( np.stack( (lat, gain, dat/minData ,  dist, imp, bw_needed/BW), axis=1) ).to(device)
    else:
        #Case of Statistical CSI
        # PreviousActions contains the previous actions as a deque containing Tensors (for every previous timestep)
        currentState = torch.FloatTensor( np.stack( (lat, dat/minData , dist, imp), axis=1) ).to(device)
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
        self.W = Resource


    def forward(self, values, weights):
        batchSize, Kusers = values.shape
        assert( list(weights.size()) == [batchSize, Kusers]  and (values>=0).all())
        #Compare everu user value with the rest. A binary matrix Kusers*Kusers shows at a cell i*j if j user is more valuable than i.       
        VperW_diff = values.unsqueeze(dim=1).detach() - values.unsqueeze(dim=2).detach()
        assert( list(VperW_diff.shape) == [batchSize, Kusers, Kusers]  )
        Better_j_than_i = 1.0* (VperW_diff >=0)
        #A vector of Kusers shows for every user if there are enough resources to satisfy it
        Satisfying_Constr = (self.W - torch.matmul(Better_j_than_i, weights.unsqueeze(dim=2)).squeeze() )>=0
        assert( list(Satisfying_Constr.shape) == [batchSize, Kusers] )
        return Satisfying_Constr*weights


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.3):
        #without Bias linear NoisyNet, taken from https://github.com/qfettes/
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))

    def forward(self, inp):
        if self.training:
            return F.linear(inp, self.weight_mu + self.weight_sigma * self.weight_epsilon)
        else:
            return F.linear(inp, self.weight_mu)


class GraphConvLayer(Module):
    '''Implementing a layer satisfying permutating equivariance between K components with each having In_feat_dim attributes/features'''

    def __init__(self, in_feat_dim, out_feat_dim, K, typeLayer):
        super(GraphConvLayer, self).__init__()
        self.K = K
        self.type = typeLayer
        self.register_buffer('adj', (torch.ones(K, K) - torch.eye(K))/self.K)
        self.register_buffer('adj_v2', torch.ones(K, K)/self.K)
        self.register_buffer('direct_adj', torch.eye(K) - (torch.ones(K, K) - torch.eye(K))/(self.K-1))
        self.bias = nn.parameter.Parameter(torch.zeros(1,out_feat_dim))
        self.linear_others = nn.Linear(in_feat_dim, out_feat_dim, bias=False)
        self.linear_self = nn.Linear(in_feat_dim, out_feat_dim, bias=False)
        self.NoisyLinear_others = NoisyLinear(in_feat_dim, out_feat_dim)
        self.NoisyLinear_self = NoisyLinear(in_feat_dim, out_feat_dim)

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
        elif self.type == 'noisy':
            feat_others = torch.matmul(self.adj_v2, self.NoisyLinear_others(x))
            feat_self = self.NoisyLinear_self(x)
            out = feat_others+feat_self
        elif self.type == 'simple':
            feat_others = torch.matmul(self.adj_v2,  self.linear_others(x))
            feat_self = self.linear_self(x)
            out = feat_others+feat_self        
        return out




#==========================
#++++++++ Actor +++++++++++
#==========================
class Actor(nn.Module):
    def __init__(self, InputStateInfo, Resources, ExploreStatInfo):
        super(Actor, self).__init__() 
        self.BW, self.EperSymb = Resources['BW'], Resources['EnergyPerSymbol'] 
        self.Explore_TypeAndInfo = ExploreStatInfo #Not exploited yet
        self.CSIestimation = InputStateInfo[3] #Full or Statistical
        self.N_MemorizedPrevActions = InputStateInfo[2]        
        self.Kusers = InputStateInfo[1]
        self.N_in_Features = InputStateInfo[0] 
                
        self.conv1 = nn.Conv1d(self.N_in_Features, 10, 1)
        self.conv2 = nn.Conv1d(10, 10, 1)#OutputDim = [BatchSize, 10, Kusers]
        self.softplus = nn.Softplus()
        self.PermutLayer0 = GraphConvLayer(10, 10, self.Kusers, 'simple_simple') 
        self.PermutLayer1 = GraphConvLayer(10, 1, self.Kusers, 'simple_simple') #OutputDim = [BatchSize, Kusers, 1]
        '''      
        #If needed to change Deep Sets with Linear
        self.LinearLayer0 = nn.Linear(10*self.Kusers, 10*self.Kusers)  
        self.LinearLayer1 = nn.Linear(10*self.Kusers, 1*self.Kusers)  
        '''
        self.Allocator = AllocatingLayer(self.BW)


    def forward(self, state, explore=False):           
        if self.CSIestimation == "Full":
            bw_req = state[:,-1,:] * self.BW#ASSUMPTION THAT THE LAST IS THE REQUIRED bw AND THE PREVIOUS imp
            mask_activeUsers = (state[:,-2,:]>0) #users of positive importance
        elif self.CSIestimation == "Statistical":
            mask_activeUsers = (state[:,3,:]>0) #users of positive importance
        if explore:
            #perturbate weights
            snr_noise_1 = 0.3
            snr_noise_2 = 0.4
            noise_1 = snr_noise_1 * self.conv1.weight.data * torch.randn_like(self.conv1.weight.data)
            self.conv1.weight.data += noise_1
            noise_2 = snr_noise_2 * self.conv2.weight.data * torch.randn_like(self.conv2.weight.data)
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
        x_out_Conv = F.relu6(x)

        #Deep Sets:
        x = self.PermutLayer0(x_out_Conv.transpose(1,2))             
        x = self.PermutLayer1(F.relu(x))  
        '''
        #If needed to change Deep Sets with Linear
        x = self.LinearLayer0(x_out_Conv.flatten(start_dim=1))       
        x = self.LinearLayer1(F.relu(x))  
        '''

        #Normalization & SoftPlus   
        x = x.squeeze() * mask_activeUsers 
        N_actUsers = torch.sum(mask_activeUsers,dim=1,keepdim=True)+1e-5#avoiding dividing by zero
        mu = torch.sum(x,dim=1, keepdim=True)/N_actUsers
        var =  torch.sum(x**2,dim=1, keepdim=True)/N_actUsers - mu**2
        x = (x - mu)/(torch.sqrt(var)+1e-8)
        x = self.softplus(x) * mask_activeUsers

        #Last Manipulation Depending on CSI
        if self.CSIestimation == "Full":
            x = x*self.BW /(bw_req + ~mask_activeUsers*self.BW*10)    #~mask_activeUsers*self.BW*10 -> To avoid division with 0  
            bw_alloc = self.Allocator(x.detach(), bw_req)  
        elif self.CSIestimation == "Statistical":                                
            x = x/(torch.sum(x,dim=1, keepdim=True) + 1e-8)
            x = x * mask_activeUsers# to nullify the chance of dividing previously with 1e-8 and now having some huge value
            bw_alloc = x*self.BW
        return bw_alloc.detach(), x



#==========================
#++++++++ Critic ++++++++++
#==========================
class DistrQN(nn.Module):    
    def __init__(self, InputStateInfo, action_size, Resources, Nquantiles):
        super(DistrQN, self).__init__()
        self.BW, self.EperSymb = Resources['BW'], Resources['EnergyPerSymbol']   
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
        '''
        #If needed to change Deep Sets with Linear
        self.LinearLayer1 = nn.Linear(10*self.Kusers, 10*self.Kusers)  
        self.LinearLayer2_value = nn.Linear(10*self.Kusers, 1)
        self.LinearLayer2_distr = nn.Linear(10*self.Kusers, self.Nquantiles)
        '''

    def forward(self, state, action):        
        state_action_in = torch.cat((state, action.unsqueeze(1)),dim=1)
        x_personal = F.relu(self.conv1( state_action_in ))
        x_personal = F.relu6(self.conv2( x_personal )).transpose(1,2)        
        x_total = self.PermutLayer1(x_personal) 
        #Mean Value
        output_value = self.PermutLayer2_value(x_total).squeeze().mean(1)        
        #Distribution
        output_distr = self.PermutLayer2_distr(x_total).mean(1)
        '''
        #If needed to change Deep Sets with Linear
        x_total = F.relu(self.LinearLayer1(x_personal.flatten(start_dim=1)))
        output_value = self.LinearLayer2_value(x_total).squeeze()
        output_distr = self.LinearLayer2_distr(x_total)
        '''
        final_output = output_value.unsqueeze(dim=1) + output_distr - output_distr.mean(1,keepdim=True)
        
        return  final_output, output_value,  output_distr.mean()
        