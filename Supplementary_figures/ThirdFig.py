import torch
import numpy as np
import torch.distributions as distr
from torch import optim 
import matplotlib.pyplot as pl
import scipy.stats as stats

N_q = 50
cdf_arange = ((2*torch.arange(N_q)+1)/(2.0*N_q))


#==================================   
#Pick-up the favorite distributions
#================================== 
conc, rate = 1.0, 1.0
mix = distr.Categorical(torch.tensor([0.5, 0.5]))
comp = distr.Normal(torch.tensor([  0.0, 4.0]),torch.tensor([1.0,1.0]))
GenGaussian_2 = distr.MixtureSameFamily(mix, comp)
mix = distr.Categorical(torch.tensor([1.0, 1.0, 1.0])/3)
comp = distr.Normal(torch.tensor([ -4.0, 0.0, 4.0]),torch.tensor([1.0,0.50,1.0]))
GenGaussian_3 = distr.MixtureSameFamily(mix, comp)
Gen_matrix = {'Gaussian':distr.Normal(0,1),
            'Exp':distr.Exponential(1.0),
            'Gamma':distr.Gamma(conc,rate),
            'Mix_2_Gaussian':GenGaussian_2,
            'Mix_3_Gaussian':GenGaussian_3
            } 
N_samples_Stop_matrix = {'Gaussian':[1500],
            'Exp':[2000],
            'Gamma':[2000],
            'Mix_2_Gaussian':[2500],
            'Mix_3_Gaussian':[4500]
            } 
initializations = {'Gaussian': [torch.ones(size=(),dtype=float), torch.ones(size=(N_q,),dtype=float)],
            'Exp':[torch.randn(size=(),dtype=float), torch.randn(size=(N_q,),dtype=float)],
            'Gamma':[torch.randn(size=(),dtype=float), torch.randn(size=(N_q,),dtype=float)-1],
            'Mix_2_Gaussian':[torch.rand(size=(),dtype=float), torch.rand(size=(N_q,),dtype=float)],
            'Mix_3_Gaussian':[torch.rand(size=(),dtype=float), torch.rand(size=(N_q,),dtype=float)]
            }

#Parameters for plotting
Color = ['plum','green','black']
Labels = ['Distr. & Dueling without $ \mathcal{L}_{shape}$', 'Distr. & Dueling', 'Real CDF'] 
fig, axs = pl.subplots(ncols=5, figsize=[16.6,3.6])

#===================================================
#Set-up the hyperparameters and initializations ones
#===================================================
LR = 0.01
N_methods = 3#Distr, Distr+Dueling, Real
plt_i = 0 
for Name_distr, Gen in Gen_matrix.items():
    N_samples_Stop = N_samples_Stop_matrix[Name_distr]
    InitCommon_mean, InitCommon_Distr = initializations[Name_distr]
    N_samples_max = N_samples_Stop[-1]
       
    print(Name_distr)
   

    #Distributional and Dueling without explicit separation
    ParCentDistrDuel =InitCommon_Distr.clone().detach().requires_grad_(True)
    ParMeanDuel = InitCommon_mean.clone().detach().requires_grad_(True)

    #Distributional and Dueling with explicit separation
    ParCentDistrDuel_sep =InitCommon_Distr.clone().detach().requires_grad_(True)
    ParMeanDuel_sep = InitCommon_mean.clone().detach().requires_grad_(True)

    #Optimizers
    opt1 = optim.Adam([ParCentDistrDuel,ParMeanDuel],lr = LR)
    opt2 = optim.Adam([ParCentDistrDuel_sep,ParMeanDuel_sep],lr = LR)


    #=======   
    #Running
    #=======
    Mean_Est = torch.empty(size=(N_methods,N_samples_max),dtype=float)
    x_min = 1.0e8
    x_max = -1.0e8
    for i in range(N_samples_max):
        x = Gen.sample() 
        assert x.dim()==0
        x_d = x.repeat(N_q)
        x_min = x if x<x_min else x_min
        x_max = x if x>x_max else x_max

        #Distributional and Dueling without explicit separation
        opt1.zero_grad()
        DuelDistr = ParMeanDuel + ParCentDistrDuel - ParCentDistrDuel.mean() 
        diffDistr1 = x_d.unsqueeze(-1) - DuelDistr #Dim: [N_q, N_q]
        loss1 = diffDistr1*(cdf_arange - (diffDistr1.detach()<0).float())
        loss1.mean().backward()
        opt1.step()

        #Distributional and Dueling with explicit separation
        opt2.zero_grad()
        DuelDistr = ParMeanDuel_sep + ParCentDistrDuel_sep - ParCentDistrDuel_sep.mean() 
        diffDistr2 = x_d.unsqueeze(-1) - DuelDistr #Dim: [N_q, N_q]
        lossDistrDuel = diffDistr2*(cdf_arange - (diffDistr2.detach()<0).float())
        loss2 = lossDistrDuel.mean() + ParCentDistrDuel_sep.mean()**2
        loss2.backward()
        opt2.step()

        #====
        #Plot
        #====
        if i+1 not in N_samples_Stop:
            continue
        x = torch.arange(start=x_min, end=x_max, step=(x_max-x_min)/100)
        distr = ParMeanDuel+ ParCentDistrDuel
        axs[plt_i].plot(distr.data, cdf_arange.data, Color[0],label=Labels[0], linewidth=2.0)
        distr_sep = ParMeanDuel_sep+ ParCentDistrDuel_sep
        axs[plt_i].plot(distr_sep.data, cdf_arange, Color[1],label=Labels[1], linewidth=2.0)
        if Name_distr == 'Gamma':
            axs[plt_i].plot(x,stats.gamma.cdf(x.numpy(),a=conc,scale=1/rate),Color[-1],label=Labels[-1], linewidth=2.0)
        else:
            axs[plt_i].plot(x,Gen.cdf(x),Color[-1],label=Labels[-1], linewidth=2.0)
        axs[plt_i].set_title(str(i+1)+' samples')
    plt_i += 1


lines_labels = [fig.axes[-1].get_legend_handles_labels()] 
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
pl.subplots_adjust(left=0.03, bottom=0.09, right=0.99, top=0.78, wspace=0.14, hspace=0.38)
fig.legend(lines, labels, loc='right', bbox_to_anchor=(0.5, 0.9, 0.48, 0.045), fontsize='x-large', ncol=3)
pl.savefig('Appendix_fig3.eps', format='eps')
pl.show()