#AdaptiveLunarLander
AdaptiveLunarLander is a software that learns how to play LunarLander-v2 by OpenAIGym.\
Three different network architectures are implemented to solve the task:
- Deep Q Networks
- Deep Q Networks with fixed Q-targets
- Double Deep Q Networks

To run the program open the terminal and type:
``` 
python main.py
```

Then there are two possibilities: train an agent or play an already trained agent to see how it plays in practice.\ 
By default he following question will be asked:
```
Do you want to play a trained agent or to train? ["play", "train"]:
```

If you want to train, you can also choose the agent type:
```
What network do you want to use? ["dqn", "dqn_fixed", "ddqn"]:
```
If you want to see some statistics:
```
tensorboard --logdir (logs|play_logs)
```