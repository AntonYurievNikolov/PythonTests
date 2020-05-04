import numpy as np
import random 
import seaborn as sns

success = 0
#Number of tests  
n = 100000
#Target value
threshold = 5
#What we reroll . 3 will reroll 1,2 and 3
reroll = 1
#Number of Dice used in the experiment
dice = 2
data = np.zeros([n, dice+1], dtype=int)
for i in range(n):
    total = 0
    for d in range(dice):
        roll = random.randint(1, 6) 
    
        if roll <= reroll:
            roll = random.randint(1, 6)
        data[i,d] = roll
        data[i,dice] = data[i,dice] + roll
    
    if (data[i,dice]>=threshold):
        success=success+1
            
#sns.distplot(data[:,dice])

print(success/n)
