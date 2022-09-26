# library for fuzzy operations

import math

# Defuzzification for Fuzzy Number
# only the required defuzzification method should be uncommented
def defuz(defuz_method,A,cc):
    if defuz_method == "WABL":
        return WABL(A,cc)
    elif defuz_method == "COG":
        return COG(A)
    else:
        return MOM(A)

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
def fuzDist(defuz_method,a,b,cc):
    # You can select the required distance by deleting the comment

    #fuz=max(abs(a[0]-b[0]),abs(a[1]-b[1]),abs(a[2]-b[2])) # D1 distance
    #fuz=abs(a[0]-b[0])+max(abs(a[1]-b[1]),abs(a[2]-b[2])) # D2=D5 distance
    #fuz=abs(defuz(a,cc)-defuz(b,cc)) # D3 distance
    fuz=abs(defuz(defuz_method,fuzSubtr(a,b,cc),cc)) # D4 distance
    return fuz

# Calculating of RMSE value according to fuzzy numbers
def fuzRMSE(defuz_method,a,b,cc):
    fuz=0
    for i in range(len(a)):
        fuz+=fuzDist(defuz_method,a[i],b[i],cc)**2
    fuz=math.sqrt(fuz/len(a))
    return fuz

# Calculating of MAE value according to fuzzy numbers
def fuzMAE(defuz_method,a,b,cc):
    fuz=0
    for i in range(len(a)):
        fuz+=fuzDist(defuz_method,a[i],b[i],cc)
    fuz=fuz/len(a)
    return fuz

# Calculating of R-squared value according to given average of fuzzy numbers
def fuzR2(defuz_method,a,b,ave,cc):
    fuz1,fuz2=0,0
    for i in range(len(a)):
        fuz1+=fuzDist(defuz_method,a[i],b[i],cc)**2
        fuz2+=fuzDist(defuz_method,a[i],ave,cc)**2
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
