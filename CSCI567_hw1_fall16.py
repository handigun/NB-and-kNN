import pandas as pd
import math
from operator import itemgetter
import time

def naive(train,test):
	print "=======NAIVE BAYES======="
	df = pd.DataFrame(pd.read_csv(train,header = None))
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	df.columns = column
	lst = df.values.tolist()
	mean_dic = {}
	var_dic = {}
	prior = [0]*8
	for c in (list(df.Class.unique())):
		mean_dic.setdefault(c,{})
		var_dic.setdefault(c,{})
		prior[c] = float(df.loc[df["Class"] == c]["Id"].count()) / df["Id"].count()
		for col in column[1:-1]: 
			mean_dic[c][col] = df.loc[df["Class"] == c][col].mean()
			var_dic[c][col] = df.loc[df["Class"] == c][col].var()
	t_a = test_accuracy(test, df, prior, mean_dic, var_dic)
	t_t = test_accuracy(train, df, prior, mean_dic, var_dic)
	print "Testing Accuracy", t_a
	print "Traing Accuracy", t_t 



def test_accuracy(test, df, prior, mean_dic, var_dic):
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	df_test =pd.DataFrame(pd.read_csv(test,header = None))
	df_test.columns = column
	test_lst = df_test.values.tolist()
	test_result = []
	for index in range(len(df_test.index)):
		pr_lst = list(prior)
		for c in (list(df.Class.unique())):
			for col in column[1:-1]:
				pr_lst[c] *= normal(mean_dic[c][col], var_dic[c][col], df_test.loc[index][col])

		test_result.append(pr_lst.index(max(pr_lst[1:])))
	##print(test_result)
	actual_result = list(df_test["Class"])
	n = 0.0
	for i in range(len(test_result)):
		if test_result[i] == actual_result[i]:
			n += 1
	##print(n / len(actual_result))
	return n / len(actual_result)
def normal(mu, sigma_sqr, x):
	if sigma_sqr != 0:
		first = 1.0/(math.sqrt(2 * math.pi) * math.sqrt(sigma_sqr))
		expo = (-1.0 / (2 * sigma_sqr)) * ((x - mu) ** 2)
		second = math.exp(expo)
		return first * second
	else:
		if x == mu:
			return 1
		else:
			return 0
def knn(train,test):
	print "=======kNN======="
	df = pd.DataFrame(pd.read_csv(train,header = None))
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	df.columns = column
	lst = df.values.tolist()
	mu = {}
	sigma = {}
	for col in column:
		mu[col] = df[col].mean()
		sigma[col] = df[col].std()
	knn_accuracy_test(test,df,0, mu, sigma)
	knn_accuracy_test(train,df,1, mu, sigma)

def knn_accuracy_test(filename, df, n, mu, sigma):
	df_test = pd.DataFrame(pd.read_csv(filename,header = None))
	column = ['Id','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
	df_test.columns = column
	manhattan = [[0]*len(df) for i in range(len(df_test))]
	eucledian = [[0]*len(df) for i in range(len(df_test))]
	for index_test in range(len(df_test)):
		for index_train in range(len(df)):
			if not n:
				# test data
				manhattan[index_test][index_train] = calc_manhattan(df_test.loc[index_test], df.loc[index_train],column, mu, sigma)
				eucledian[index_test][index_train] = calc_eucledian(df_test.loc[index_test], df.loc[index_train],column, mu, sigma)
			else:
				# train data
				if index_test != index_train:
					manhattan[index_test][index_train] = calc_manhattan(df_test.loc[index_test], df.loc[index_train],column, mu, sigma)
					#rint dist
					if manhattan[index_test][index_train]['distance'] == 0.0:
						manhattan[index_test][index_train] = {"distance" : float("inf"), "class" : int(df.loc[index_train]['Class'])}
					eucledian[index_test][index_train] = calc_eucledian(df_test.loc[index_test], df.loc[index_train],column, mu, sigma)
					if eucledian[index_test][index_train]['distance'] == 0:
						eucledian[index_test][index_train] = {"distance" : float("inf"), "class" : int(df.loc[index_train]['Class'])}
				else:
					manhattan[index_test][index_train] = {"distance" : float("inf"), "class" : int(df.loc[index_train]['Class'])}
					eucledian[index_test][index_train] = {"distance" : float("inf"), "class" : int(df.loc[index_train]['Class'])}

	#print(manhattan[0])
	if not n:
		print "kNN Testing accuracy "
		accuracy_test(df_test, manhattan, "Manhattan")
		accuracy_test(df_test, eucledian, "Euclidean")
	else:
		print "kNN Training accuracy "
		accuracy_test(df_test, manhattan, "Manhattan")
		accuracy_test(df_test, eucledian, "Euclidean")

def accuracy_test(df_test, series, string):
	class_lst = [[0] * len(df_test) for i in range(0,4)]
	##print(class_lst)
	for index in range(len(df_test)):
		for i in range(1,8,2):
			lst = get_k_nearest(series[index],i)
			count_lst = [0] * 8
			class_var = 0
			for k in range(len(lst)):
				count_lst[lst[k]['class']] += 1
			##print(count_lst)
			max_count_lst = [k for k, j in enumerate(count_lst) if j == max(count_lst[1:])]

			if len(max_count_lst) == 1:
				class_var = max_count_lst[0]
			else:
				for item in lst:
					if item['class'] in max_count_lst:
						class_var = item['class'] 
						break
			class_lst[int(math.floor(i/2))][index] = class_var
	print "For", string
	for i in range(len(class_lst)):
		acc = compare(class_lst[i], list(df_test["Class"]))
		acc = acc * 1.0/ len(df_test)
		print "Accuracy = ", acc, "For k =", i * 2 + 1
	

def compare(lst1, lst2):
	n = 0
	for i in range(len(lst1)):
		if lst1[i] != lst2[i]:
			n += 1
	return len(lst1) - n

def calc_manhattan(test_series, train_series,column, mu, sigma):
	dist = 0
	for col in column[1:-1]:
		dist += abs(((test_series[col] - mu[col]) / sigma[col])  - ((train_series[col] - mu[col])/sigma[col]))
	return {"distance" : dist, "class" : int(train_series['Class'])} 

def calc_eucledian(test_series, train_series,column, mu, sigma):
	dist = 0
	for col in column[1:-1]:
		dist += (((test_series[col] - mu[col]) / sigma[col]) - ((train_series[col] - mu[col])/sigma[col])) ** 2 
	return {"distance": math.sqrt(dist), "class" : int(train_series['Class'])}

def get_k_nearest(lst, n):
	lst = sorted(lst, key=itemgetter('distance')) 
	return lst[:n]
	
if __name__ == '__main__':
	start =  time.time()
	naive("train.txt","test.txt")
	end = time.time()
	print end - start
	start =  time.time()
	knn("train.txt", "test.txt")
	end = time.time()
	print end - start