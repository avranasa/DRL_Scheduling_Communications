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
            #'Exp':distr.Exponential(1.0),
            'Gamma':distr.Gamma(conc,rate),
            'Mix_2_Gaussian':GenGaussian_2,
            #'Mix_3_Gaussian':GenGaussian_3,
            } 
N_samples_Stop_matrix = {'Gaussian':[100,400,1000],
            #'Exp':[100,400,1000],
            'Gamma':[100,400,1000],
            'Mix_2_Gaussian':[100,400,1000],
            #'Mix_3_Gaussian':[200,500,800,1500],
            } 
N_col = len(Gen_matrix)
N_row = len(N_samples_Stop_matrix['Gaussian'])

#Parameters for plotting
Color = ['blue','green','black']
Labels = ['Distr.', 'Distr. & Dueling', 'Real CDF'] 
fig, axs = pl.subplots(nrows=N_row , ncols=N_col, figsize=[20*N_col/5,8])


#===================================================
#Set-up the hyperparameters and initializations ones
#===================================================
LR = 0.01
InitCommon_mean = torch.zeros(size=(),dtype=float)
InitCommon_Distr = torch.zeros(size=(N_q,),dtype=float)
N_methods = 3#Distr, Distr+Dueling, Real
Plot_column = 0
for Name_distr, Gen in Gen_matrix.items():
    N_samples_Stop = N_samples_Stop_matrix[Name_distr]
    N_samples_max = N_samples_Stop[-1]
    plt_i = 0    
    print(Name_distr)
   

    #Only Distributional:
    ParDistr = InitCommon_Distr.clone().detach().requires_grad_(True)

    #Distributional and Dueling with explicit separation
    ParCentDistrDuel_sep = InitCommon_Distr.clone().detach().requires_grad_(True)
    ParMeanDuel_sep = InitCommon_mean.clone().detach().requires_grad_(True)

    #Optimizers
    opt1 = optim.Adam([ParDistr], lr = LR)
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

        #Only Distributional:
        opt1.zero_grad()    
        diffDistr1 = x_d.unsqueeze(-1) - ParDistr #Dim: [N_q, N_q]
        loss1 = diffDistr1*(cdf_arange - (diffDistr1.detach()<0).float())
        loss1.mean().backward()
        opt1.step()

        #Distributional and Dueling with explicit separation
        opt2.zero_grad()
        DuelDistr = ParMeanDuel_sep + ParCentDistrDuel_sep - ParCentDistrDuel_sep.mean() 
        diffDistr2 = x_d.unsqueeze(-1) - DuelDistr #Dim: [N_q, N_q]
        lossDistrDuel = diffDistr2*(cdf_arange - (diffDistr2.detach()<0).float())
        loss3 = lossDistrDuel.mean() + ParCentDistrDuel_sep.mean()**2
        loss3.backward()
        opt2.step()

        #====
        #Plot
        #====
        if i+1 not in N_samples_Stop:
            continue
        x = torch.arange(start=x_min, end=x_max, step=(x_max-x_min)/100)
        axs[plt_i,Plot_column].plot(ParDistr.data, cdf_arange.data, Color[0],label=Labels[0], linewidth=2.0)
        distr_sep = ParMeanDuel_sep+ ParCentDistrDuel_sep
        axs[plt_i,Plot_column].plot(distr_sep.data, cdf_arange, Color[1],label=Labels[1], linewidth=2.0)
        if Name_distr == 'Gamma':
            axs[plt_i,Plot_column].plot(x,stats.gamma.cdf(x.numpy(),a=conc,scale=1/rate),Color[-1],label=Labels[-1], linewidth=2.0)
        else:
            axs[plt_i,Plot_column].plot(x,Gen.cdf(x),Color[-1],label=Labels[-1], linewidth=2.0)
        axs[plt_i,Plot_column].set_title(str(i+1)+' samples')
        axs[plt_i,Plot_column].set_ylabel("CDF and approx. of CDF of $Z_{}$".format(Plot_column+1))
        axs[plt_i,Plot_column].set_xlabel("Domain of Random Variable $Z_{}$".format(Plot_column+1))
        plt_i += 1
    Plot_column += 1

lines_labels = [fig.axes[-1].get_legend_handles_labels()] 
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
pl.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.94, wspace=0.14, hspace=0.38)
fig.legend(lines, labels, loc='right', bbox_to_anchor=(0.5, 0.96, 0.48, 0.045), fontsize='x-large', ncol=3)

fig.subplots_adjust(
    left  = 0.1,  # the left side of the subplots of the figure
    right = 0.98,    # the right side of the subplots of the figure
    bottom = 0.02,   # the bottom of the subplots of the figure
    top = 0.92,      # the top of the subplots of the figure
    wspace = 0.27,   # the amount of width reserved for blank space between subplots
    hspace = 0.47)   # the amount of height reserved for white space between subplots


pl.savefig('Appendix_fig1.eps', format='eps')
pl.show()
