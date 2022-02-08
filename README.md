# DRL_Scheduling_Communications
Contains the code related to our paper on scheduling and resource allocation in wireless communications ("Deep Reinforcement Learning for Resource Constrained Multiclass Scheduling with Random Service Rates").

There are three folders.\
-The folders "Main_Synthetic_Traffic" and "Main_Real_Traffic" contain the code used in almost all the experiments. The first one is for the experiments in which the traffic and the channel realizations are generated solely using models and the second depend on real measurements. Both folders contain a file named Main.py where the characteristic of the problem (i.e. traffic and channel model) can be defined. Also there is selected the choice of the algorithm and its hyperparameters. Please to run experiments choose in Main.py the characteristics of the scheduling problem and the algorithm and then run that file.\
-The last one named "Supplementary_figures" contains the code that was used to generate some additional plots (for example the Figure 3).



Method Used:   Knapsack 
Channel corr.:  0.0
BW:  5000000.0 /// Power:  0.003125
Protocol Used:  Full
100
200
300
400

===========>..........step:  400 ...........<===========
Total positive per user reward:  0.49482980089916506  and lost reward:  0.00016056518946692356
and the relative positive reward:  0.9996756195666276  and lost reward:  0.00032438043337225896
Success rate per class:
    Class 0 :  0.995920365535248
    Class 1 :  0.996679875763093
    Class 2 :  0.0
