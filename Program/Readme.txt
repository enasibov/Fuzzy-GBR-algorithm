### "fyzzyGBR.py" – a basic gradient boosting regression module with fuzzy target values (v1.1.0) ###
### The other "load_prepare.py" module is used to load, normalize the datasets and fuzzyfy targets ### 


#The constructor of the FuzzyGBR model
FuzzyGBR(defuz_method = "MOM", opt_index=0.5, distance="D1", 
                 boost_iterations = 100, learning_rate = 0.1, tree_depth = 1)

"""  Parameters
	
defuz_method - the defuzzification method: 

        "MOM" - mean of maxima, 
	
        "COG" - center of gravity, 
	
        "WABL" - weighted average based ol levels.
	
max_left_spread - the maximum left side spread of the simulating target fuzzy numbers,

max_right_spread - the maximum right side spread of the simulating target fuzzy numbers,

distance - the fuzzy distance measures between the FNs A = (a[0],a[1],a[2]) and B = (b[0],b[1],b[2]):

        D1(A, B) = max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[2]-b[2])) 
	
        D2(A, B)  = abs(a[0]-b[0])+max(abs(a[1]-b[1]),abs(a[2]-b[2])) 
	
        D3(A, B)  = abs(defuz(A)-defuz(B)) 
	
        D4(A, B)  = abs(defuz(fuzSubtr(A,B))) 
	
M - the number of boosting iterations,

learning_rate - the learning rate of boosting iterations,

tree_depth - the depth of the stump trees.

"""
The Anaconda Python JupyterLab should be tested for running the fuzzyGBR notebook.
The model contains the following functions:

# initialization of the model
model = FuzzyGBR(defuz_method = "WABL", opt_index=0.5, distance = "D4", boost_iterations = 100, 
               learning_rate = 0.1, tree_depth = 1)

# print model parameters
model.print_parameters()

# model fitting according to the fuzzy targets
model.fit(self, X_train,y_train)

# predict of fuzzyGBR results for input X
predicted = model.predict(X)

#==================================================================
# Functions to perform operations containing fuzzy numbers:

# Defuzzification of th FN A. cc is the optimizm parameter.
defuz(defuz_method,A,cc)

# The RMSE value of the predictions
fuzRMSE(defuz_method,distance,a,b,cc)

# The MAPE value of the predictions
fuzMAPE(defuz_method,distance,a,b,cc)

# The R-squared value of the predictions
fuzR2(defuz_method,distance,a,b,ave,cc)

# Defuzzification for Fuzzy Number
# only the required defuzzification method should be uncommented
defuz(defuz_method,A,cc)

# Center of Area defuzzification for Fuzzy Number
COG(A)

# WABL defuzzification for Fuzzy Number
WABL(A,cc=0.5)

# Mean of the Maxima defuzzification for Fuzzy Number
MOM(A)

# fuzzy average of fuzzy numbers
def fuzAve(a,cc):

# distance between fuzzy numbers
def fuzDist(defuz_method,distance,a,b,cc):

# Calculating of RMSE value according to fuzzy numbers
def fuzRMSE(defuz_method,distance,a,b,cc):

# Calculating of MAPE value according to fuzzy numbers
def fuzMAPE(defuz_method,distance,a,b,cc):

# Calculating of R-squared value according to given average of fuzzy numbers
def fuzR2(defuz_method,distance,a,b,ave,cc):

# Fuzzy Subtraction A-B of fuzzy numbers
def fuzSubtr(a,b,cc):

# Fuzzy Addition A+B of fuzzy numbers
def fuzAdd(a,b,cc):

# Multiplication of fuzzy number A by scalar b
def fuzMultBy(a,b,cc):
