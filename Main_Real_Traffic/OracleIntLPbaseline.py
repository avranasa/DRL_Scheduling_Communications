import numpy as np
import pulp
import cplex #fast!
import TestMethods


class BaselineIntLPoracleMethod():
    '''It is applied for the case of Full-CSI and uses an oracle to know the channels and the user traffic of the next
    T_steps in te future. The problem  can be formulated as Integer (binary) Linear Programming where the variables 
    to be optimized are   1 if scheduler wants to serve the user and 0 if not. Don't forget that since the necessary 
    resources are precisely known due to Full CSI the decision is only binary (serve or not)
    '''

    def __init__(self, Resources, env):
        self.BW = Resources['BW']  
        self.BwBlock = Resources['ResourceBlock']
        self.N_blocks = self.BW / self.BwBlock
        self.Tgreedy = env.OracleT+1
        self.K = env.Kusers  
        self.ILPsolver = pulp.LpProblem("Oracle_Scheduler",pulp.LpMaximize)
        self.x = pulp.LpVariable.matrix('Scheduled',(list(range(self.K)),list(range(self.Tgreedy))) , lowBound = 0, upBound = 1, cat = pulp.LpInteger)
            

    def Create_UsersList(self, sample, StateUsers, futUserImp, futUserLat):
        # Outputs a list Users containing for every active user: 
        # [a ranking number, in how many time steps will appear, in how many time steps he will no longer exist, importance of its class]
        Users = []
        for k in range(self.K):
            t_start = 0
            t_end = min(StateUsers['lat'][sample,k],self.Tgreedy)
            imp = StateUsers['imp'][sample,k] 
            if imp > 0:
                Users.append([k ,t_start, t_end, imp])
            while t_end<self.Tgreedy:
                t_start = t_end
                assert(futUserLat[sample, k, t_start-1]>0)
                t_end = min(t_start+futUserLat[sample, k, t_start-1],self.Tgreedy)
                imp = futUserImp[sample, k, t_start-1]
                if imp > 0:
                    Users.append([k, t_start, t_end, imp])
        return Users
            

    def __call__(self, env):  
        Walloc = np.zeros((env.N_parallel_env,env.Kusers),dtype=float)
        Xsol = np.zeros_like(Walloc)
        futUserLat, futUserImp, Wth = env.Provide_ILPoracle_matrices()
        for sample in range(env.N_parallel_env):   
            Users = self.Create_UsersList( sample, env.StateUsers, futUserImp, futUserLat)  
            self.ILPsolver.setObjective(   sum( Users[u][3]*self.x[Users[u][0]][t] for u in range(len(Users)) for t in range(Users[u][1],Users[u][2]) ))
            self.ILPsolver.constraints.clear()          
            #constraints
            for u in range(len(Users)):
                self.ILPsolver += sum(self.x[Users[u][0]][t] for t in range(Users[u][1],Users[u][2]))<=1    
            for t in range(self.Tgreedy):
                self.ILPsolver += sum(Wth[sample][k][t]*self.x[k][t] for k in range(self.K)) <= self.N_blocks
                    
            self.ILPsolver.solve(pulp.CPLEX_PY( msg=False))
            Xsol[sample,:] = np.array([1 if self.x[k][0].value() >=0.99 else 0  for k in range(self.K)])
        Walloc = Wth[:,:,0]*Xsol 
        return Walloc

   
def BaselineTest_intLPoracle(env, Resources):
    assert( env.CSIestimation == "Full" )
    Scheduler = BaselineIntLPoracleMethod(Resources, env)
    TestMethods.test_Scheduler("IntegerLPoracle", Scheduler, env)  

