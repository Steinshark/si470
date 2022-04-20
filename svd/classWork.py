import pandas as pd
import numpy
from pprint import pp
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error



foodFrame=pd.read_csv('foodsAndMovies.csv.gz')
foodFrame = foodFrame[['Broccoli', 'Mushrooms', 'Beef Tacos', 'Salads', 'Black Licorice', 'Steak', 'Grilled Chicken', 'Mayonnaise', 'Candy Corn', 'Pulled Pork', 'Spicy Mustard', 'Raw Oysters', 'Bananas', 'Avocado', 'Eggs', 'Olives', 'Tofu', 'Cottage Cheese']]
means=foodFrame.mean()

for col in foodFrame.columns:
    foodFrame[col]=foodFrame[col].fillna(means[col])
foodArray=foodFrame.to_numpy()


# Calc the SVD of the array
U, S, V = numpy.linalg.svd(foodArray)


## plot everything
plt.plot(range(len(S)),S)
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
plt.show()



# Reshape the matrices and reconstruct A
print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
keep= int(input("keep: "))
S[keep:]=numpy.zeros(len(S)-keep)
Sigma=numpy.zeros((U.shape[0],len(S)))
Sigma[0:len(S),0:len(S)]=numpy.diag(S)
newArr = U@Sigma@V




# Find errors:
mse = list()
kept = list(range(1,len(S)-1))
for i,keep in enumerate(kept):
    U, S, V = numpy.linalg.svd(foodArray)
    S[keep:]=numpy.zeros(len(S)-keep)
    Sigma=numpy.zeros((U.shape[0],len(S)))
    Sigma[0:len(S),0:len(S)]=numpy.diag(S)
    newArr = U@Sigma@V
    mse.append(0)
    count = 0
    for row,row_O in zip(newArr,foodArray):
        count += 1
        mse[i] += mean_squared_error(row,row_O)
    mse[i] /= count
print(f"{kept}")
print(f"{mse}")
plt.plot(kept,mse)
plt.show()
