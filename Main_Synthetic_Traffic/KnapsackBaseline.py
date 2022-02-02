import math
import numpy as np
from ortools.algorithms import pywrapknapsack_solver
import TestMethods



class BaselineKnapsackMethod():
    '''Used for the full-CSI case (and since channel is known either the Base station allocates the 
    necessary resources or not at all. So no retransmission is ever needed)  One-step-greedy can be 
    formulated as a Knapsack problem where the weights are how much exactly bandwidth is needed for a user 
    and the value is the importanceof the class that user belongs to.

    knapsack Solver https://github.com/google/or-tools/blob/stable/ortools/algorithms/knapsack_solver.h
    '''

    def __init__(self, BW, EperSymb):
        # Quantization accuracy since float numbers will become integers (the weights will be approximated upwards   
        # so as to remain bigger than required bandwidth)
        self.Accuracy = 1e-3
        self.capacity = [math.floor(1/self.Accuracy)]
        self.BW = BW
        self.EperSymb = EperSymb
        # This knapsack solver only accepts integer values of weights,values and capacities. So a quantization with "self.Accuracy" is performed
        self.solver = pywrapknapsack_solver.KnapsackSolver( pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_CBC_MIP_SOLVER, 'AllocatorFullCSI')

    
    def __call__(self, env):   
        '''This alogrithm does:
        1)compute the required bandwidth for every user when env.step() is called
        2)compute gains/values by satisfying every user
        3)make the floats integers         
        '''
        Ealloc = self.EperSymb*np.ones((env.N_parallel_env,env.Kusers), dtype=float) 
        Walloc = np.zeros_like(Ealloc)        
        #Below 1e-5 is added  so as not to have numerical problem 
        Weights = env.StateUsers['data'] /np.log2(1 + env.ConstLoss_Noise*env.StateUsers['absH_2']*(env.DistUsersBS**-env.PathLoss)*env.EperSym ) + 1e-5 
        for sample in range(env.N_parallel_env):   
            ind_ImpossibleToSatisfy = Weights[sample]>self.BW#This steps is avoiding getting negative number because huge float numbers when quantized need 
                                                             #more than 32bits integer. Anything bigger than capacity self.BW by default is not being satisfied
            Weights[sample][ind_ImpossibleToSatisfy] = 2*self.BW 
            values = env.StateUsers[sample]['imp']    
            RescaledW, RescaledV = self.QuantizeWeightsValues(Weights[sample], self.BW, values, self.Accuracy)
            RescaledW = [RescaledW]#the Knapsack google solver needs it in this form
            #Solving
            self.solver.Init(RescaledV, RescaledW, self.capacity)
            self.solver.Solve()
            Walloc[sample,:] = Weights[sample]
            for user in range(env.Kusers):
                if not self.solver.BestSolutionContains(user):
                    Walloc[sample,user] = 0
        return Walloc, Ealloc


    def QuantizeWeightsValues(self, weights, BW, values, Accuracy):
        '''Takes weights and values in numpy array form and returns the quantized and
        rescaled version of them in the form of lists quantize BW in "1/Accuracy" resolution
        '''
        quantW = BW * Accuracy
        RescaledWeights = np.ceil(weights/quantW).astype(int).tolist()#round up to be sure that you will succeed the transmission
        quantV = np.max(values) * Accuracy
        if quantV == 0:
            return (RescaledWeights, np.zeros_like(RescaledWeights))
        RescaledValues = np.ceil(values/quantV).astype(int).tolist()
        return (RescaledWeights, RescaledValues)


   
def BaselineTest_Knapsack(env, Resources):
    assert( env.CSIestimation == "Full" )
    BW, EperSymb = Resources['BW'], Resources['EnergyPerSymbol']
    Scheduler = BaselineKnapsackMethod(BW, EperSymb)
    TestMethods.test_Scheduler("Knapsack", Scheduler, env)
