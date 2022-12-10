# Import the necessary modules
from osgeo import gdal
import numpy as np
import pandas as pd

# Open the GeoTIFF files using GDAL
datasetTrainingGT = gdal.Open('E:/Ceng463/Proje_Gibraltar/S2A_MSIL1C_20220516_Train_GT.tif')

# Read the data from the first GeoTIFF file into a NumPy array
trainGT2d = datasetTrainingGT.ReadAsArray()
trainGT2d = np.swapaxes(trainGT2d, 0, 1)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
trainGT1d = trainGT2d.reshape(trainGT2d.shape[0] * trainGT2d.shape[1], 1)

# Convert the combined array into a Pandas DataFrame
dfTrainLabels = pd.DataFrame(trainGT1d)

# Export the DataFrame as a CSV file
# dfTrainLabels.to_csv('train.csv', index=False)
np.save('train_gt.npy', trainGT1d)

datasetTraining = gdal.Open('E:/Ceng463/Proje_Gibraltar/S2A_MSIL1C_20220516_TrainingData.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTraing = datasetTraining.ReadAsArray()
dataTraing = np.swapaxes(dataTraing, 0, 2)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTraining1d = dataTraing.reshape(dataTraing.shape[0] * dataTraing.shape[1], -1)
dfTrain = pd.DataFrame(dataTraining1d)

final_data = pd.concat([dfTrainLabels, dfTrain])

train_label_data = pd.concat([dfTrainLabels, dfTrain], axis=1)
train_label_data.columns=['Code', 'Blue', 'Green', 'Red', 'NIR']
train_label_data.to_csv('train.csv')

np.save('train.npy', dataTraining1d)

datasetTestGT = gdal.Open('E:/Ceng463/Proje_Gibraltar/S2B_MSIL1C_20220528_Test_GT.tif')

# Read the data from the first GeoTIFF file into a NumPy array
testGT2d = datasetTestGT.ReadAsArray()
testGT2d = testGT2d[1:, :]
testGT2d = np.swapaxes(testGT2d, 0, 1)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
testGT1d = testGT2d.reshape(testGT2d.shape[0] * testGT2d.shape[1], 1)

# Convert the combined array into a Pandas DataFrame
df = pd.DataFrame(testGT1d)

# Export the DataFrame as a CSV file
df.to_csv('test_gt.csv')
np.save('test_gt.npy', testGT1d)

datasetTest = gdal.Open('E:/Ceng463/Proje_Gibraltar/S2B_MSIL1C_20220528_Test.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTest2d = datasetTest.ReadAsArray()
dataTest2d = np.swapaxes(dataTest2d, 0, 2)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
np.save('test_all.npy', dataTest1d)
# Convert the combined array into a Pandas DataFrame
dfTest = pd.DataFrame(dataTest1d)
dfTest.columns=['Blue', 'Green', 'Red', 'NIR']
# Export the DataFrame as a CSV file
dfTest.to_csv('test.csv')


from sklearn.model_selection import train_test_split
X_Test, X_Val, y_test, y_val = train_test_split(dataTest1d, testGT1d, stratify=testGT1d, test_size=0.30)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Create the KNN classifier with k=1
clf = KNeighborsClassifier(n_neighbors=1)

# Use cross-validation to evaluate the model's accuracy
# scores = cross_val_score(clf, dataTraining1d, np.ravel(trainGT1d), cv=5)
# acc = scores.mean()

# Fit the classifier to the data
clf.fit(dataTraining1d, np.ravel(trainGT1d))

# Predict labels for new data
predictions = clf.predict(dataTest1d)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(testGT1d, predictions)
print(f'Accuracy: {accuracy:.2%}')

# Compute the confusion matrix
labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

cm = confusion_matrix(testGT1d, predictions)
print(classification_report(testGT1d, predictions,target_names=labels))
# print(cm)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()

df = pd.DataFrame(predictions)
df.columns=['Code']
# Export the DataFrame as a CSV file
df.to_csv('submission.csv')