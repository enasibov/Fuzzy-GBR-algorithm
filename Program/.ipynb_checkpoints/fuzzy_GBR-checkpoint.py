# Gradient Boosting Regression Model 
# with target values as triangular fuzzy numbers
# using different fuzzy distances and different defuzzification methods
# by Resmiye Nasiboglu and Efendi Nasibov
# September, 2022, v1.1.0

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import math

from warnings import filterwarnings
filterwarnings('ignore')

"""
Parameters of the model:

dataset - is the dataset to handle ("iris", "cars", "diabetes", "boston", "penguins", "planets", 
        "diamonds", "mpg", "tips", taxis")
defuz_method - the defuzzification method: 
        "MOM" - mean of maxima, 
        "COG" - center of gravity, 
        "WABL" - weighted average based ol levels.
opt_index - optimizm index in case of the WABL defuzzification,
max_left_spread - the maximum left side spread of the simulating target fuzzy numbers,
max_right_spread - the maximum right side spread of the simulating target fuzzy numbers,
distance - the fuzzy distance measures between the FNs A = (a[0],a[1],a[2]) and B = (b[0],b[1],b[2]):
        D1(A, B) = max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[2]-b[2])) 
        D2(A, B) = abs(a[0]-b[0])+max(abs(a[1]-b[1]),abs(a[2]-b[2])) 
        D3(A, B) = abs(defuz(A)-defuz(B)) 
        D4(A, B) = abs(defuz(fuzSubtr(A,B))) 
boost_iterations - the number of boosting iterations,
learning_rate - the learning rate of algorithm,
tree_depth - the depth of the stump trees.
"""

class FuzzyGBR:
    def __init__(self, defuz_method = "MOM", opt_index=0.5, distance="D1", 
                 boost_iterations = 100, learning_rate = 0.1, tree_depth = 1):
        
        self.defuz_method=defuz_method
        self.opt_index=opt_index
        self.distance=distance
        self.boost_iterations=boost_iterations
        self.learning_rate=learning_rate
        self.tree_depth=tree_depth
        self.max_leaf=2**self.tree_depth  # maximum leaf number of the stump trees
        self.trres=[]
        self.gamma=[[[0,0,0] for i in range(self.max_leaf)] for j in range(self.boost_iterations)]
        self.fuz_ave=[0,0,0]  

    # model fitting according to the fuzzy targets
    def fit(self, X_train,y_train):
        boost_iterations=self.boost_iterations
        c=self.opt_index
        max_leaf=self.max_leaf
        defuz_method=self.defuz_method
        tree_depth=self.tree_depth
        learning_rate=self.learning_rate
        f_ave=fuzAve(y_train,self.opt_index)
        self.fuz_ave=f_ave 

        F=[[[0,0,0] for i in range(len(X_train))] for j in range(boost_iterations)]

        # F[i] is the fuzzy outputs of the model after i.th iteration
        F[0]=[f_ave for _ in range(len(X_train))]

        # gamma is the predicted fuzzy output (as a Fuzzy Number) according to the leaf 
        gamma=[[[0,0,0] for i in range(max_leaf)] for j in range(boost_iterations)]
        trees=[]

        # boosting iterations
        for m in range(1,boost_iterations):    
            rrr=[fuzSubtr(y_train[i],F[m-1][i],c) for i in range(len(X_train))]

            # stump tree is constructed up to the defuzzified values of the FNs
            r1=[defuz(defuz_method,rrr[i],c) for i in range(len(rrr))]

            # constructing of the stump tree
            tree = DecisionTreeRegressor(random_state=0,max_depth=tree_depth)
            tree.fit(X_train, r1)
            trees.append(tree)

            # h is the list of the indices of the leafs
            h=tree.apply(X_train)   

            # h1 is the list of the distinct leaf indices 
            h1=list(set(h))

            for l in range(len(h1)):
                leaf_l=[j for j in range(len(r1)) if h[j]==h1[l]] 
                ss=[rrr[j] for j in leaf_l]
                ss1=np.reshape(ss,(-1,3))
                
                gamma[m][l]=fuzAve(ss1,c) #for each leaf node
                for k in leaf_l:
                    F[m][k]=fuzAdd(F[m-1][k],fuzMultBy(gamma[m][l],learning_rate,c),c) 

        self.gamma=gamma
        self.trees=trees
        self.F=F

    # predict of fuzzyGBR results for input X
    def predict(self, X):
        boost_iterations=self.boost_iterations
        learning_rate=self.learning_rate
        c=self.opt_index
        trees=self.trees
        gamma=self.gamma
        ave=self.fuz_ave
        FM=[ave for _ in range(len(X))]

        for m in range(1,boost_iterations):
            h=trees[m-1].apply(X)
            h1=list(set(h))
            for l in range(len(h1)):
                leaf_l=[j for j in range(len(X)) if h[j]==h1[l]] 
                for k in leaf_l:
                    FF=fuzAdd(FM[k],fuzMultBy(gamma[m][l],learning_rate,c),c)
                    FM[k]=FF    #for each xi of each leaf node 
     
        return FM
    
    def print_parameters(self):
        print("=== Model parameters: ===")
        print("defuz_method = ",self.defuz_method)
        print("opt_index = ",self.opt_index)
        print("distance = ",self.distance)
        print("boost_iterations = ",self.boost_iterations)
        print("learning_rate = ",self.learning_rate)
        print("tree_depth = ",self.tree_depth)
        print("=========================")

# functions for fuzzy operations

# Defuzzification for Fuzzy Number
# only the required defuzzification method should be uncommented
def defuz(defuz_method,A,cc):
    if defuz_method == "WABL":
        return WABL(A,cc)
    elif defuz_method == "COG":
        return COG(A)
    elif defuz_method == "MOM":
        return MOM(A)
    else:
        print("DEFUZZIFICATION ERROR !\nTHE PARAMETER HAVE TO BE IN ['WABL','COG','MOM']")
        raise SystemExit

# Center of Area defuzzification for Fuzzy Number
def COG(A):
    s=1 # shape convexity parameter 
    b=(A[0]+s*((A[0]-A[1])+(A[0]+A[2])))/(2*s+1)
    return b

# WABL defuzzification for Fuzzy Number
def WABL(A,cc=0.5):
    #cc=1.0 # optimism parameter
    s=1   # shape convexity parameter 
    k=0   # increasing speed of level importances
    b=cc*((A[0]+A[2])-((k+1)/(k+s+1))*A[2])+(1-cc)*((A[0]-A[1])+((k+1)/(k+s+1))*A[1])
    return b

# Mean of the Maxima defuzzification for Fuzzy Number
def MOM(A):
    return A[0]

# fuzzy average of fuzzy numbers
def fuzAve(a,cc):
    fuz=[0,0,0]
    for i in range(len(a)):
        fuz=fuzAdd(a[i],fuz,cc)
    fuz=[fuz[0]/len(a),fuz[1],fuz[2]]    
    return fuz

# distance between fuzzy numbers
def fuzDist(defuz_method,distance,a,b,cc):
    if distance=="D1":
        fuz=max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[2]-b[2])) # D1 distance
    elif distance=="D2":
        fuz=abs(a[0]-b[0])+max(abs(a[1]-b[1]),abs(a[2]-b[2])) # D2=D5 distance
    elif distance=="D3":
        fuz=abs(defuz(defuz_method,a,cc)-defuz(defuz_method,b,cc)) # D3 distance
    elif distance=="D4":  # distance=="D4":
        fuz=abs(defuz(defuz_method,fuzSubtr(a,b,cc),cc)) # D4 distance
    else:
        print("DISTANCE ERROR !\nTHE PARAMETER HAVE TO BE IN ['D1','D2','D3','D4']")
        raise SystemExit
    return fuz

# Calculating of RMSE value according to fuzzy numbers
def fuzRMSE(defuz_method,distance,a,b,cc):
    fuz=0
    for i in range(len(a)):
        fuz+=fuzDist(defuz_method,distance,a[i],b[i],cc)**2
    fuz=math.sqrt(fuz/len(a))
    return fuz

# Calculating of MAPE value according to fuzzy numbers
def fuzMAPE(defuz_method,distance,a,b,cc):
    fuz=0
    for i in range(len(a)):
        fuz+=fuzDist(defuz_method,distance,a[i],b[i],cc)/a[i][0]*100
    fuz=fuz/len(a)
    return fuz

# Calculating of R-squared value according to given average of fuzzy numbers
def fuzR2(defuz_method,distance,a,b,ave,cc):
    fuz1,fuz2=0,0
    for i in range(len(a)):
        fuz1+=fuzDist(defuz_method,distance,a[i],b[i],cc)**2
        fuz2+=fuzDist(defuz_method,distance,a[i],ave,cc)**2
    fuz=1-(fuz1/fuz2)
    return fuz

# fuzzy Subtraction A-B of fuzzy numbers
def fuzSubtr(a,b,cc):
    fuz=[a[0]-b[0],max(a[1],b[1]),max(a[2],b[2])]
    #fuz=[a[0]-b[0],a[1]+b[2],a[2]+b[1]]
    return fuz

# fuzzy Addition A+B of fuzzy numbers
def fuzAdd(a,b,cc):
    fuz=[a[0]+b[0],max(a[1],b[1]),max(a[2],b[2])]
    #fuz=[a[0]+b[0],a[1]+b[1],a[2]+b[2]]
    return fuz

# Multiplication of fuzzy number A by scalar b
def fuzMultBy(a,b,cc):
    fuz=[b*a[0],a[1],a[2]]
    return fuz




    
