# circuit_model
A proof of concept that demonstrates modeling transistor circuits using linear regression and neural network.

The program uses PySpice as simulator. 
https://github.com/FabriceSalvaire/PySpice

Predictive Technology Model (PTM) from Arizona State University are used as the transistor models.
http://ptm.asu.edu/

The run_sim.py creates the differential amplifier in PySpice and run transient simulations. The circuit inputs are randomized and the inputs and outputs are saved to csv files.
Two seeds are run, one for generating test data, one for generating train data.

The regression.py uses the train data to train the regression model. The trained regression model is then used to predict the outputs of the test data.
The output from the regression model is then compared against the results produced by the simulator. The square error is then calculated.

The nn.py is similar to regression.py, but instead of using regression model, it uses neural network. Right now, only the fully connected neural network is used.
Recurrent neural network might provides better predictions but is currently under construction.

# Regression result
![Regression result](/figures/regression.png)

# Simple NN result
![Simple NN result](/figures/simple_nn.png)
