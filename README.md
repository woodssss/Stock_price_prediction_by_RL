# Stock_price_prediction_by_Reinforcement_learning
The main goal of this project is to train trader agents by RL.
# Method 1: Q learning table
# Running code
Usage of code: you can define paramter as you want. The first parameter is size of sliding windows in time; \
second parameter is number of level, which discretizes state into different levels; third parameter is number \
of epsisode. \
For example
```
python run_this.py ^GSPC 5 6 2000
```

#Method 2: Policy Gradient
# Running code
Usage of code: you can define paramter as you want. The first parameter is size of sliding windows in time; \
second parameter is number of epsisode. \
For example
```
python run_this_pg.py ^GSPC 5 2000
```


# Conclusion and Future work
In fact, these are just some toy models, and performances are poor. In Q learning table experiment, the performance highly depends on the fineness of mesh of state space. In Policy gradient experiment, the training process is hard to converge. \
I plan to 

# Reference
[Reinforcement_Learning_For_Stock_Prediction](https://github.com/llSourcell/Reinforcement_Learning_for_Stock_Prediction.git)\
[Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow.git)
