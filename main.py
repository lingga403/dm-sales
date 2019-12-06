

from pandas import read_csv
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

#Loading the dataset
series = read_csv('C:/Users/admin/Desktop/TimeSeriesForecastingModels/monthly-car-sales.csv',header=0,index_col=0)

#display first few rows
#print(series.head(5))

#line plot of dataset
#series.plot()
#pyplot.show()

#Walk forward validation method
#Root men squared error : penalizes large errors and the scores

#Optimized PErsistence Forecast
#Uses previous observation to predict the next time step

#Prepare data
X = series.values
train, test = X[0:-24],X[-24:]
persistence_values = range(1,25)
scores = list()

for p in persistence_values:
	#walk forward validation
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# make prediction
		yhat = history[-p]
		predictions.append(yhat)
		#observation
		history.append(test[i])
	#report performance
	rmse = sqrt(mean_squared_error(test,predictions))
	scores.append(rmse)
	print('p=%d RMSE:%.3f' %(p,rmse))

#plot scores over persistence values
pyplot.plot(persistence_values,scores)
pyplot.show()

