# DRL_Scheduling_Communications
Contains the code related to our paper ( Journal_DRL_Scheduling.pdf  ) on scheduling and resource allocation in wireless communications.

There are three folders. 
-The folders "Main_Synthetic_Traffic" and "Main_Real_Traffic" contain the code used in almost all the experiments. The first one is for the experiments in which the traffic and the channel realizations are generated solely using models and the second depend on real measurements. Both folders contain a file named Main.py where the characteristic of the problem (i.e. traffic and channel model) can be defined. Also there is selected the choice of the algorithm and its hyperparameters. Please to run experiments choose in Main.py the characteristics of the scheduling problem and the algorithm and then run that script.
-The last one named "Supplementary_figures" contains the code that was used to generate some additional plots (for example the Figure 3).
