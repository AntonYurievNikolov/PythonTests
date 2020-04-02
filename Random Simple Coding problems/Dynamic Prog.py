import pandas as pd
def isSubsetSum(set,n, sum) : 
	if (sum == 0) : 
		return True
	elif (n == 0 and sum != 0) : 
		return False
	elif (set[n - 1] > sum) : 
		return isSubsetSum(set, n - 1, sum);  
	return isSubsetSum(set, n-1, sum) or isSubsetSum(set, n-1, sum-set[n-1]) 


test = [3, 34, 4, 12, 5, 2, 120] 
set = pd.array(test) 
sum = 127
n = len(set)
if (isSubsetSum(set, n, sum) == True) : 
	print("Yes") 
else : 
	print("No") 


import pandas as pd
class A:
    data = []
    def __init__(self):
        self.data = [3, 34, 4, 12, 5, 2, 120]  
 
class B(A):
    def __init__(self):
        A.__init__(self)
        self.betterdata = pd.array(self.data)*2  - 1
        
def test(t):
    for i in range(len(t)-1):
        print (t[i])
    
x = A()  
test(x.data)
y= B()
test(y.betterdata)
