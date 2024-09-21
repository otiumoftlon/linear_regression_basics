

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Aux_Function.Save_images import *
from Aux_Function.PCAM import *
sns.set_theme(style="whitegrid")

df = pd.read_csv('Linear Regression/Ecommerce_Customers.csv')

features_num = ['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership', 'Yearly Amount Spent']

max_spent = df['Yearly Amount Spent'].max()
min_spent = df['Yearly Amount Spent'].min()

corr_matrix = df[features_num].corr()

columns = df[features_num].columns

mean = df[features_num] - np.mean(df[features_num] , axis=0)

cov_matrix = (mean.T@mean)/(np.shape(mean)[0])

tab_1,data_pca,f_1 = pca(df[features_num],corr_matrix,columns)
tab_2,data_fa,f_2 = fa(df[features_num],corr_matrix,columns)
print('tabla:','/n',tab_2)
print(f_2)

plt.figure(figsize=(8, 6))
"""
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
            center=0,  # Center the colormap at 0 for better visualization
            fmt='.2f',  # Format of the annotations
            linewidths=0.5,  # Line width of the grid lines
            linecolor='gray',  # Color of the grid lines
            square=True
            ) # Make cells square-shaped)
ax.tick_params(axis='both', which='major', labelsize=8)  # Ajusta el tamaño de las etiquetas de los ejes

plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()
"""

df['Max_Spent'] = 0
df.loc[df['Yearly Amount Spent'] >= (2*((max_spent - min_spent) / 4)+min_spent), 'Max_Spent'] = 1

sns.scatterplot(x=data_fa['F_1'], y=data_fa['F_2'], hue=df['Max_Spent'])

plt.show()
# sns.scatterplot(x='Length of Membership', y='Yearly Amount Spent', hue='Max_Spent', data=df)  
# 

