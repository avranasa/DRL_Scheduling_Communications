# DRL_Scheduling_Communications
Contains the code related to our paper on scheduling and resource allocation in wireless communications ("Deep Reinforcement Learning for Resource Constrained Multiclass Scheduling with Random Service Rates").

There are three folders.\
-The folders "Main_Synthetic_Traffic" and "Main_Real_Traffic" contain the code used in almost all the experiments. The first one is for the experiments in which the traffic and the channel realizations are generated solely using models and the second depend on real measurements. Both folders contain a file named Main.py where the characteristic of the problem (i.e. traffic and channel model) can be defined. Also there is selected the choice of the algorithm and its hyperparameters. Please to run experiments set in Main.py the desired global parameters that determine the characteristics of the scheduling problem and the algorithm and then run that file. For example the first choice is setting the global variable of the method used for the scheduling and can be easily chosen by changing the index of the list below:
```python
#~~~~~~~ METHOD/Algorithm ~~~~~~~~~  
methodAvailable = ["Knapsack","ExpRule", "IntegerLPoracle",  "DeepScheduler"]
MethodUsed = methodAvailable[1] #Define your method
```

The output will give a short summary of the main characteristics of the problem, the algorithm used and its performance. 

```
Method Used:   Knapsack 
Channel corr.:  0.0
BW:  2000000.0 
Protocol Used:  Full CSI

===========>..........step:  400 ...........<===========
Total positive per user reward:  0.46082978791465407  and lost reward:  0.034099996801126006
and the relative positive reward:  0.9311013443640124  and lost reward:  0.06889865563598759
Success rate per class:
    Class 0 :  0.9847658755612572
    Class 1 :  0.8863270777479892
```

Additionally, for the "Deep Scheduler", Tensorboard is used to record and plot the progress of the deep reinforcement learning algorithm.

-The last one named "Supplementary_figures" contains the code that was used to generate some additional plots (for example the Figure 3).


****************************
The link to our paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10250919

In case you want to use either the github repository for your research project please use the following citation:

@Article{DRL_Scheduling_Communications ,
author={Avranas, Apostolos and Ciblat, Philippe and Kountouris, Marios},
journal={IEEE Transactions on Machine Learning in Communications and Networking},
title={Deep Reinforcement Learning for Resource Constrained Multiclass Scheduling in Wireless Networks},
year={2023},
volume={1},
number={},
pages={225-241},
doi={10.1109/TMLCN.2023.3314705}}
*****************************
