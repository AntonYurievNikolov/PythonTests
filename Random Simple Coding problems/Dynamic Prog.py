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
