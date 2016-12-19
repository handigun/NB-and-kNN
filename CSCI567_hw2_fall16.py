from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import operator 
import time
import itertools

def normalise(df_train,mu_train, sigma_train):
	df_train = (df_train - mu_train)/ sigma_train
	df_train.insert(0,'x0',1.0)
	return df_train


def calc_theta(df_train, target):
	transpose_matrix = df_train.transpose()
	f = np.linalg.pinv(np.dot(transpose_matrix, df_train))
	s = np.dot(transpose_matrix, target)
	theta = np.dot(f,s)
	return theta
	
def calc_regression_theta(df_train, target, lmda):
	transpose_matrix = df_train.transpose()
	x = np.dot(transpose_matrix, df_train)
	iden = np.matrix(np.identity(len(df_train.columns)))
	iden[0][0] = 0
	x += lmbda * iden
	f = np.linalg.pinv(x)
	s = np.dot(transpose_matrix, target)
	theta = np.dot(f,s)
	return theta

def predict(df_train, target, theta):
	pred = [0] * len(df_train)
	for index in range(len(df_train.index)):
		pred[index] = np.dot(df_train.iloc[index], theta)
	return pred

def calc_mse(df_train, target, theta):
	pred = predict(df_train, target, theta)
	mse = 0
	for index in range(len(df_train.index)):
		mse += (pred[index] - target[index])**2 
	mse /= len(df_train)
	return mse

def calc_pearson_coefficient(source, target):
	# print(source.as_matrix().mean())
	source -= source.mean()
	target -= target.mean()
	sigma_src = source.std()
	sigma_tgt = target.std()
	numerator = np.dot(source,target)
	#print(numerator/(sigma_tgt * sigma_src * len(source)))
	return numerator/(sigma_tgt * sigma_src * len(source))


def linear_regression(df_train, df_test, train_target, test_target):
	theta = calc_theta(df_train, train_target)
	return (calc_mse(df_train, train_target, theta),calc_mse(df_test, test_target, theta))
	


def find_highest(df, target):
	dic_pearson = {}
	column = df.columns
	# print(column)
	# print(df)
	for col in column:
		dic_pearson[col] = abs(calc_pearson_coefficient(df[col], target))
	sorted_feature = sorted(dic_pearson.items(), key=operator.itemgetter(1), reverse = True)
	#sorted_feature = sorted_feature[:4]
	feature_list = []
	for item in sorted_feature:
		feature_list.append(item[0])
	return feature_list
def call_linear_regression(feature_list, df_train, df_test, train_target, test_target):
	temp_train = pd.DataFrame()
	temp_test = pd.DataFrame()
	for feature in feature_list:
		temp_train[feature] = df_train[feature]
		temp_test[feature] = df_test[feature]
	mu_train = temp_train.mean()
	sigma_train = temp_train.std()
	df_train_norm = normalise(temp_train, mu_train, sigma_train)
	df_test_norm = normalise(temp_test, mu_train, sigma_train)
	return linear_regression(df_train_norm, df_test_norm, train_target, test_target)

def histogram(df):
	for col in list(df.columns):
		plt.xlabel(col)
		plt.ylabel("Frequency")
		plt.hist(df[col], bins =10)
		plt.show()



if __name__ == "__main__":
	boston = load_boston()
	df = pd.DataFrame(boston.data)
	df.columns = boston.feature_names
	column = list(df.columns)
	target = boston.target
	df_total = pd.DataFrame()

	''' Taking all '''
	# df_total = df_total.append(df)
	# df_total['MEDV'] = boston.target
	# column.append("MEDV")


	#train data
	#pd.DataFrame(df.loc[i] for i in range(len(df.index)) if i%7).to_csv("train.txt",header = None)
	df_train = pd.DataFrame(df.loc[i] for i in range(len(df.index)) if i%7)
	train_target = list(target[i] for i in range(len(df.index)) if i%7)
	mu_train = df_train.mean()
	sigma_train = df_train.std()
	df_train_norm = normalise(df_train, mu_train, sigma_train)
	# theta = calc_theta(df_train, train_target)
	# print("=======LINEAR REGRESSION=======")
	# print "Training data MSE: ", predict(df_train, train_target, theta)

	
	
	#test data
	#pd.DataFrame(df.loc[i] for i in range(len(df.index)) if i%7==0).to_csv("test.txt",header = None)
	df_test = pd.DataFrame(df.loc[i] for i in range(len(df.index)) if i%7==0)
	test_target = list(target[i] for i in range(len(df.index)) if i%7==0)
	df_test_norm = normalise(df_test, mu_train, sigma_train)
	print '=======HISTOGRAM======='
	histogram(df)
	# print "Testing data MSE: ",predict(df_test, test_target, theta)
	print "=======PEARSON COEFFICIENT======="
	for col in column:
		print "For ",col,": ",calc_pearson_coefficient(np.array(df_train[col]), np.array(train_target))
	print '\n'
	print("=======LINEAR REGRESSION=======")
	train_mse, test_mse = linear_regression(df_train_norm, df_test_norm, train_target, test_target)
	print "Training data MSE: ",train_mse
	print "Testing data MSE: ",test_mse
	
	print '\n'
	print "=======RIDGE REGRESSION========="
	for lmbda in [0.01, 0.1, 1.0]:
		print "FOR LAMBDA: %s" %(str(lmbda)) 
		regression_theta = calc_regression_theta(df_train_norm, train_target,lmbda)
		print "Training data MSE: ", calc_mse(df_train_norm, train_target, regression_theta)
		print "Testing data MSE: ",calc_mse(df_test_norm, test_target, regression_theta)
	

	# index = list(range(len(df)))
	# np.random.shuffle(index) 
	# lst = np.array_split(index, 10)
	''' taking only training data'''
	df_total = df_total.append(df_train.ix[:,df_train.columns != "x0"])
	df_total['MEDV'] = train_target
	column.append("MEDV")
	np.random.seed(0)
	print '\n'
	print "=======RIDGE REGRESSION WITH CROSS VALIDATION======="
	for lmbda in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
		print "FOR LAMBDA: %s" %(str(lmbda)) 
		lst = np.array_split(np.random.permutation(df_total),10)
		#train_mse = 0.0
		test_mse = 0.0
		for i in range(len(lst)):
			df_test_r = pd.DataFrame(lst[i],columns = column)
			df_train_r = pd.DataFrame()
			
			for j in range(len(lst)):
				if j != i:
					df_temp = pd.DataFrame(lst[j])
					df_train_r = df_train_r.append(df_temp)
			df_train_r.columns = column		
			train_target_r = list(df_train_r['MEDV'])
			mu_train = df_train_r.ix[:,df_train_r.columns != "MEDV"].mean()
			sigma_train = df_train_r.ix[:,df_train_r.columns != "MEDV"].std()
			df_train_r = normalise(df_train_r.ix[:,df_train_r.columns != "MEDV"], mu_train, sigma_train)
			
			test_target_r = list(df_test_r['MEDV'])
			df_test_r = normalise(df_test_r.ix[:,df_test_r.columns != "MEDV"], mu_train, sigma_train)
			regression_theta = calc_regression_theta(df_train_r, train_target_r,lmbda)
			#train_mse += predict(df_train, train_target, regression_theta)
			test_mse += calc_mse(df_test_r, test_target_r, regression_theta)
		#print "Training data MSE: ", train_mse/len(lst)
		print "Testing data MSE: ", test_mse/len(lst)
	feature_list = find_highest(df,target)
	print '\n'
	print("=======FEATURE SELECTION=======")
	print "Top 4 features",feature_list[:4]
	train_mse, test_mse = call_linear_regression(feature_list[:4],df_train, df_test, train_target, test_target)
	print "Training data MSE: ",train_mse
	print "Testing data MSE: ",test_mse
	
	highest = []
	temp_train = pd.DataFrame()
	temp_train = df_train
	while len(highest) < 4:
		highest.append(feature_list[0])
		temp_train = temp_train.drop(feature_list[0], 1)
		train = pd.DataFrame()
		for high in highest:
			train[high] = df_train[high]
		theta = calc_theta(train, train_target)
		pred = predict(train, target, theta)
		residue = [0] * len(train)
		for index in range(len(train.index)):
			residue[index] = pred[index] - train_target[index]
		residue = np.asarray(residue)
		feature_list = find_highest(temp_train,residue)

	
	train_mse, test_mse = call_linear_regression(highest,df_train, df_test, train_target, test_target)
	print '\n'
	print "=======RESIDUE BASED FEATURE SELECTION======="
	print(highest)
	print "Training data MSE: ",train_mse
	print "Testing data MSE: ",test_mse
	'''Brute Force '''
	combi_lst = list(itertools.combinations(column[:-1], 4))
	dict_combi = {}
	start = time.time()
	for combi in combi_lst:
		train_mse, test_mse = call_linear_regression(combi,df_train, df_test, train_target, test_target)	
		dict_combi[combi] = [train_mse,test_mse]	
	end = time.time()
	sorted_feature = sorted(dict_combi.items(), key=operator.itemgetter(1))
	print '\n'
	print "=======BRUTE FORCE======="
	print list(sorted_feature[0][0])
	print "Training data MSE: ",dict_combi[sorted_feature[0][0]][0]
	print "Testing data MSE: ",dict_combi[sorted_feature[0][0]][1]
	
	'''Feature Expansion'''
	combi_lst = list(itertools.combinations(column[:-1], 2))
	df_exp = pd.DataFrame()
	df_exp = df_train_norm.ix[:,df_train_norm.columns != "x0"]
	df_exp_test = pd.DataFrame()
	df_exp_test = df_test_norm.ix[:,df_test_norm.columns != "x0"]
	for combi in combi_lst:
	 	df_exp[combi] = df_train_norm.ix[:,combi[0]] * df_train_norm.ix[:,combi[1]]
	 	df_exp_test[combi] = df_test_norm.ix[:,combi[0]] * df_test_norm.ix[:,combi[1]]
	for col in df_train.columns:
		combi = (col, col)
		df_exp[combi] = df_train_norm.ix[:,combi[0]] * df_train_norm.ix[:,combi[1]]
		df_exp_test[combi] = df_test_norm.ix[:,combi[0]] * df_test_norm.ix[:,combi[1]]
	mu = df_exp.mean()
	sd = df_exp.std()
	df_exp = normalise(df_exp, mu, sd)
	df_exp_test = normalise(df_exp_test, mu, sd)
	train_mse, test_mse = linear_regression(df_exp, df_exp_test, train_target, test_target)
	print '\n'
	print "=======FEATURE EXPANSION======= "
	print "Training data MSE: ",train_mse
	print "Testing data MSE: ",test_mse
