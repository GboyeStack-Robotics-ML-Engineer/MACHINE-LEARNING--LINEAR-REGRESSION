import pandas as pd

import numpy as np

import random

id=np.array(range(100))

id=np.array(id)

df_id=pd.DataFrame(id,columns=['id'])

df_id['index']=df_id

survived=[]

for c in range(len(id)):
    
    value=random.randint(0,2)
    
    survived.append(value)
    
df_survived=pd.DataFrame(survived, columns=['SURVIVED'])

print(df_survived)

#df_survived['index']=df_id.values

sex=[]

for c in range(len(id)):
    
    value=random.randint(0,10)
    
    if value>=5:
        
        sex.append('MALE')
        
    elif value<5:
        
        sex.append('FEMALE')
        
sex=np.array(sex)

df_sex=pd.DataFrame(id, columns=['SEX'])

age=[]

for c in range(len(id)):
    
    value=random.randint(1,90)
    
    age.append(value)
    
age=np.array(age)

df_age=pd.DataFrame(id, columns=['AGE'])

p_class=[]

for c in range(len(id)):
    
    value=random.randint(0,4)
    
    p_class.append(value)
    
p_class=np.array(p_class)

df_p_class=pd.DataFrame(id, columns=['P CLASS'])

df=pd.concat([[df_id,df_survived,p_class,df_sex,df_age]])

print(df)


