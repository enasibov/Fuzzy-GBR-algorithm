# Fuzzy-GBR-algorithm
FyzzyGBR – a gradient boosting regression software with fuzzy target values


The Anaconda Python JupyterLab should be used for running the notebook.
The fuzzyGBR() function on the main module FuzzyGBR.ipynb of the program should be called as follows:

fuzzyGBR(dataset, defuz_method = "MOM", max_left_spread = 0.2, max_right_spread = 0.2, 
         distance = "D4", M = 201, learning_rate = 0.1, tree_depth = 1)
	

"""  Parameters

dataset - is the dataset to handle ("iris", "cars", "diabetes", "boston", "penguins", "planets", 
        "diamonds", "mpg", "tips", taxis")
	
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

