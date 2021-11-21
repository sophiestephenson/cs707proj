# CS707: LiDAR Project

To run, enter 

```bash
python3 blackbox.py
```

Currently runs an infinite loop with dummy reinforcement learning.
1. Grabs info about the RBG perspective of each camera (rate of change, object size, direction)
2. Makes predictions for when each camera should fire based on the scene info and parameters
3. Throws that data at the simulator (TO DO: currently just makes up simulated distances)
4. Saves the simulated results and updates the parameters for prediction based on the results


Also, on first run, it shows the calculation of the scene details for each camera.

