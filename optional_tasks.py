################  O P T I O N A L  ##############


# [E] Building an RNN layer
# Objective: Gaining a computational understanding of an RNN

# The equations of an RNN are
#
# a[t] = W x[t] + U a[t-1]
# o[t] = B a[t]
#
# All the multiplications here are matrix-vector multiplications (i.e., W, U and B
# are matrices, and x[t] and a[t] are vectors). 
# To implement an RNN layer, these equations need to be implemented. The values
# that define the matrices need to be passed to the layer. 
# 

# And implementation in R may look as follows:
#
# rnn_layer = function(w, list_of_sequences, sigma=plogis) {
#   # 1. Setup
# 	W = matrix(w[1:9],3,3)
# 	U = matrix(w[1:9 + 9], 3, 3)
# 	B = matrix(w[1:3+9+9],1,3)
#
# 	nr_sequences = length(list_of_sequences)
# 	outputs = rep(NA, nr_sequences)
#
#   # 2. Iterate over sequences
# 	for (i in 1:nr_sequences) {
# 		# get i-th sequence
# 		X = list_of_sequences[[i]]
# 		# initialize hidden state to 0
# 		a = 0 * X[1,]
# 		# Iterate over the time points
# 		for (j in 1:nrow(X)) {
# 			a = W %*% X[j,] + U %*% a
# 		}
# 		# store RNN output for i-th sequence
# 		outputs[i] = B %*% a
# 	}
# 	outputs
# }

# In this implementation 
# w:                 is a single vector containing the flattened weights for the matrices  W ,  U , and  B .
# list_of_sequences: is a list where each element is a sequence represented as a matrix ( X ), 
#                    where rows are time steps and columns are input features.
# sigma:             An optional activation function (default is the sigmoid function plogis).
# 1. Setup: Splits the vector w into three matrices ( W, U, B ).
#       and determines the number of sequences to process.
# 2. Iterate Over Sequences: For each sequence in list_of_sequences 
#      the sequence matrix  X is extracted and processed by
#      • first initializing the hidden state  a  to zero (vector of the same size as a row of  X).
#      • then, for each time step (each row in  X ) the hidden state  a[t] is updated using the RNN 
#        formula:  a[t] = W  x[t] + U a[t-1].
# 4. Compute Output: After processing all time steps of the sequence, the output value is
#      computed using final hidden state using the the equation o = B a[T] and is
#      stored as the predicted output value for this sequence.
# The return value is the vector of RNN output values (one value for each sequence in list_of_sequences).



# Task 10: Translate this function into Python (by hand!)

# Your code here:
# -----------------------------------------------
def rnn_layer(w: np.array, list_of_sequences: list[np.array], sigma=sigmoid ) -> np.array:
    pass # Your code

# Test
np.random.seed(10)
list_of_sequences = [np.random.normal(size=(5,3)) for _ in range(100)]
wstart = np.random.normal(size=(3*3 + 3*3 + 3)) 
o = rnn_layer(wstart, list_of_sequences)
o.shape == (100,) and o.mean().round(3) == 16.287 and o.std().astype(int) == 133
# -----------------------------------------------




# [F] Defining a loss function
# Objective: define the least squares loss function suitable for minimizing with scipy.optimize.minimize

# We want to predict the target value y from each of the sequences using our RNN. We'll do
# that with the minimize function from the scipy.optimize module. But we'll need to implement
# a loss function.

# In R the loss function may look as follows:
#
# rnn_loss = function(w, list_of_sequences, y) {
# 	pred = rnn_layer(w, list_of_sequences)
# 	sum((y-pred)^2)
# }




# Task 11: translate the above loss function into Python

# Your code here:
# -----------------------------------------------
def rnn_loss(w: np.array, w, list_of_sequences: list[np.array], y: np.array) -> np.float64:
    pass # Your code

# Test:
y = np.array([(X @ np.arange(1,4))[0] for X in list_of_sequences])
o = rnn_loss(wstart, list_of_sequences, y)
o.size == 1 and o.round(3) == 17794.733
# -----------------------------------------------




# [G] Fitting the RNN with minimize for the scipy.optmize module
# Objective: fit your RNN on real data

# The data that we will fit is a macroeconomics data set. We'll try to predict inflation ('infl')
# from the consumer price index ('cpi') and unemployment rate ('unemp').
# First, load the data set:
from statsmodels.datasets import macrodata

data = macrodata.load_pandas().data
X = np.hstack([np.ones((len(data),1)), data[['cpi','unemp']].values]) # Features: CPI and unemployment
y = data['infl'].values # Target: inflation

# Next we want to prepare a dataset for training sequence-based models like RNNs. We create 
# input-output pairs where each input is a sequence of seq_len time steps from X, and the output 
# is the corresponding target value y at the next time step after the sequence.

seq_len = 7 # Define the length of each input sequence (we choose 7 consecutive time steps).

# Create a list of tuples:
data_pairs = [(X[i:i+seq_len], y[i+seq_len]) for i in range(len(X)-seq_len)]
# - First element: a slice of `X` of length `seq_len` (the input sequence).
# - Second element: the target value `y` corresponding to the step after the sequence.
# Example: If seq_len=4, for i=0, pair is (X[0:4], y[4]).

# We need the input sequences and target values in a separate list. A trick to do this is this:

list_of_sequences, yy = list(zip(*data_pairs))

# Here, the zip(*...) is used for transposing a list of tuples. It splits the tuple pairs into 
# two separate lists:
# First list: all input sequences (X[i:i+seq_len])
# Second list: all target values (y[i+seq_len])
# The * operator in Python unpacks the list elements into separate arguments for the zip() 
# function. E.g., func(*[2,4,5]) is the same as func(2,4,5). 

# Now we are ready to fit the RNN to the data set. We need to load the optimization routine 
# 'minimize' from the scipy.optimize module

from scipy.optimize import minimize

# fit the RNN (this may take a minute)
fit = minimize(rnn_loss, wstart, args=(list_of_sequences, yy), method='BFGS')
print(fit)

# The 'success' component in fit may be false, and this is due to a loss of computational 
# precision. For now we'll just settle for the weights it has found so far. 

# To evaluate the fit we can compute the correlation between the values predicted by the
# RNN and the true values
pred = rnn_layer(fit['x'], list_of_sequences)
np.corrcoef(pred,yy)

# How good is this? To gage the performance of the RNN we'll compare it to a linear 
# regression with the same data
Z = X[:len(yy)] # features corresponding to elements in yy at the previous time step
linreg_coefs = np.linalg.lstsq(Z, yy, rcond=None)[0] # rcond=None suppresses warning message
linreg_pred = Z @ linreg_coefs
np.corrcoef(linreg_pred, yy)

# The correlation of the RNN predicted values is substantially higher! But it also has
# many more parameters, and so is more flexible. 

# To visualize the difference in performance we plot the true values and predicted values
import matplotlib.pyplot as plt

plt.plot(yy)
plt.plot(pred)
plt.plot(linreg_pred)
plt.legend(['Truth','RNN','LinReg'])


