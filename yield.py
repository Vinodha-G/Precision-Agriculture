
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
df_cotton=pd.read_csv('C:/Users/Admin/Desktop/New folder (2)/Unprocessed Data.csv')
print(df_cotton)

"""data preprocessing"""

print(df_cotton.isna().sum())
print(df_cotton.groupby('State').mean())

df_cotton=df_cotton.fillna(df_cotton.groupby('State').transform('mean'))
print(df_cotton)

print(df_cotton.isna().sum())


print(df_cotton.info())

fig, ax = plt.subplots()

colors = {'Alabama':'#E50800',
          'Arizona':'#D50713',
          'Arkansas':'#C60626',
          'California':'#597AA1',
          'Georgia':'#A7044C',
          'Louisiana':'#98045F',
          'Mississippi':'#880372',
          'Missouri':'#790285',
          'New Mexico':'#690198',
          'North Carolina':'#5A00AB',
          'Oklahoma':'#4B00BF',
          'South Carolina':'#8FD900',
          'Tennessee':'#2FD500',
          'Texas':'#00B9CA'
           }

ax.scatter(df_cotton['Year'], df_cotton['Lint Yield (Pounds/Harvested Acre)'], c=df_cotton['State'].apply(lambda x: colors[x]),)

plt.show()

sns.boxplot(df_cotton['Lint Yield (Pounds/Harvested Acre)'])
plt.title("Boxplot of Yield vs State")

plt.figure(figsize=(12,10))
plt.subplot(231) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Nitrogen (%)'], color='r', bins=100, hist_kws={'alpha': 0.4})
plt.title('Nitrogen Area %')

plt.subplot(232) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Phosphorous (%)'], color='b', bins=100, hist_kws={'alpha': 0.4})
plt.title('Phosphorous Area %')

plt.subplot(233) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Potash (%)'], color='g', bins=100, hist_kws={'alpha': 0.4})
plt.title('Potash Area %')

plt.subplot(234) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Nitrogen (Pounds/Acre)'], color='r', bins=100, hist_kws={'alpha': 0.4})
plt.title('Nitrogen (Pounds/Acre)')

plt.subplot(235) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Phosphorous (Pounds/Acre)'], color='b', bins=100, hist_kws={'alpha': 0.4})
plt.title('Phosphorous (Pounds/Acre)')

plt.subplot(236) #2 is row, 2 is column and 1 is position
sns.distplot(df_cotton['Potash (Pounds/Acre)'], color='g', bins=100, hist_kws={'alpha': 0.4})
plt.title('Potash (Pounds/Acre)')


plt.figure(figsize=(22,6))
GraphData=df_cotton.groupby(['State'])['Lint Yield (Pounds/Harvested Acre)'].sum().nlargest(10)


GraphData.plot(kind='bar')
plt.ylabel('Lint Yield')
plt.xlabel('State Name')

def plot_pie(nitrogen,phosphorous,potash,title):
    labels = ['Nitrogen Area','Phosphorous Area','Potash Area']
    sizes = [nitrogen,phosphorous,potash]
    color= ['#42275a','#dd5e89','#f7bb97']
    explode = []

    for i in labels:
        explode.append(0.05)
    
    plt.figure(figsize= (15,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode = explode,colors = color)
    centre_circle = plt.Circle((0,0),0.70,fc='white')

    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title(title,fontsize = 20)
    plt.axis('equal')  
    plt.tight_layout()
    
States = df_cotton['State'].unique().tolist()
States

state_df = pd.DataFrame()

for state in States:
    one_state_df = df_cotton.loc[df_cotton['State'] == state,:]
    state_df = pd.concat([state_df,pd.DataFrame(one_state_df.iloc[-1,:]).T],axis = 0)
    phosphorous = one_state_df['Phosphorous (%)'].values[-1]
    potash = one_state_df['Potash (%)'].values[-1]
    nitrogen = df_cotton['Nitrogen (%)'].values[-1]
    plot_pie(nitrogen,phosphorous,potash,state)
    
sns.jointplot((df_cotton['Lint Yield (Pounds/Harvested Acre)'], df_cotton['Nitrogen (Pounds/Acre)']), kind='kde')
sns.jointplot((df_cotton['Lint Yield (Pounds/Harvested Acre)'],df_cotton['Phosphorous (Pounds/Acre)']),kind='kde')
sns.jointplot((df_cotton['Lint Yield (Pounds/Harvested Acre)'],df_cotton['Potash (Pounds/Acre)']),kind='kde')

print(df_cotton['State'].unique())
mapping = ({'Alabama':1,
'Arizona':2,
'Arkansas':3,
'California':4,
'Georgia':5,
'Louisiana':6,
'Mississippi':7,
'Missouri':8,
'New Mexico':9,
'North Carolina':10,
'Oklahoma':11,
'South Carolina':12,
'Tennessee':13,
'Texas':14,
           })
df_cotton=df_cotton.replace({'State': mapping})
print(df_cotton)

x=df_cotton.drop('Lint Yield (Pounds/Harvested Acre)',axis=1)
y=df_cotton['Lint Yield (Pounds/Harvested Acre)']

print(x)
print(y)

 

cor1, _ = pearsonr(df_cotton['State'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Year'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Nitrogen (%)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Nitrogen (Pounds/Acre)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Phosphorous (%)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Phosphorous (Pounds/Acre)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Potash (%)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)



cor1, _ = pearsonr(df_cotton['Potash (Pounds/Acre)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Area Planted (acres)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)

cor1, _ = pearsonr(df_cotton['Harvested Area (acres)'], df_cotton['Lint Yield (Pounds/Harvested Acre)']) 
print('Pearsons correlation: %.3f' % cor1)


import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(10, activation='softmax'))
model.summary()
#create a correlation heatmap
sns.heatmap(df_cotton.corr(),annot=True,cmap='terrain',linewidth=5)
fig=plt.gcf() #method to make heatmap
fig.set_size_inches(15,10)
#No need to drop any columns since the Pearson Correlations are upwards 0.2 (medium relations)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) #80% for Training and 20% for Testing
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

from sklearn import ensemble
yield_predict = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'squared_error')
yield_predict.fit(x_train, y_train)

yield_predict_test=yield_predict.predict(x_test)
yield_predict_train=yield_predict.predict(x_train)
pd.DataFrame({'actual unseen data':y_train,'predicted unseen data':yield_predict_train})

scores = cross_val_score(yield_predict, x_test, y_test, cv=5)
print(scores)

predictions = cross_val_predict(yield_predict, x_test, y_test, cv=5)
accuracy = metrics.r2_score(y_test, predictions)
print(accuracy)

x_ax = range(len(y_test))
plt.figure(figsize=(20,6))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, yield_predict_test, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

print('MAE= ',metrics.mean_absolute_error(y_test,yield_predict_test))
print('MSE= ',metrics.mean_squared_error(y_test,yield_predict_test))
print('R2 value= ',yield_predict.score(x_test,y_test))
print('Adjusted R2 value= ',1 - (1 - (yield_predict.score(x_test,y_test))) * ((756 - 1)/(756-10-1)))
print('RMSE (train)= ',np.sqrt(mean_squared_error(y_train,yield_predict_train)))
print('RMSE (test)= ',np.sqrt(mean_squared_error(y_test,yield_predict_test)))

print(df_cotton['Lint Yield (Pounds/Harvested Acre)'].max() - df_cotton['Lint Yield (Pounds/Harvested Acre)'].min())