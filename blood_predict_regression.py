import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



def fill_na(variable,blood_data):
	data_age_mean = blood_data[variable].mean()
	data_age_std = blood_data[variable].std()
	data_age_null_size = blood_data[variable].isnull().sum()
	data_age_random = np.random.randint(data_age_mean - data_age_std, data_age_mean + data_age_std,
	                                    size=data_age_null_size)
	if data_age_mean - data_age_std>=0:
		blood_data[variable] = blood_data[variable].fillna(np.random.randint(data_age_mean - data_age_std, data_age_mean + data_age_std))
	else:
		blood_data[variable] = blood_data[variable].fillna(np.random.randint(0, data_age_mean + data_age_std))
	return blood_data


def preprocess(blood_data):
	blood_data["d1"] = blood_data["d1"].fillna(0)
	blood_data["d2"] = blood_data["d2"].fillna(0)
	blood_data["d3"] = blood_data["d3"].fillna(0)
	blood_data["d4"] = blood_data["d4"].fillna(0)
	blood_data["d5"] = blood_data["d5"].fillna(0)
	blood_data["d6"] = blood_data["d6"].fillna(0)
	blood_data["d7"] = blood_data["d7"].fillna(0)
	blood_data["d8"] = blood_data["d8"].fillna(0)
	blood_data["d9"] = blood_data["d9"].fillna(0)
	blood_data["d10"] = blood_data["d10"].fillna(0)
	blood_data["d11"] = blood_data["d11"].fillna(0)
	blood_data["d12"] = blood_data["d12"].fillna(0)
	blood_data["d13"] = blood_data["d13"].fillna(0)
	blood_data["d14"] = blood_data["d14"].fillna(0)
	blood_data["d15"] = blood_data["d15"].fillna(0)

	blood_data["redcell"] = pd.to_numeric(blood_data["redcell"])
	blood_data["weight"] = pd.to_numeric(blood_data["weight"])
	blood_data["albumin"] = pd.to_numeric(blood_data["albumin"])
	blood_data["temperature"] = pd.to_numeric(blood_data["temperature"])
	blood_data["low_pressure"] = pd.to_numeric(blood_data["low_pressure"])
	blood_data["high_pressure"] = pd.to_numeric(blood_data["high_pressure"])
	blood_data["upperbody"] = pd.to_numeric(blood_data["upperbody"])
	blood_data["lowerbody"] = pd.to_numeric(blood_data["lowerbody"])
	blood_data["d1"] = pd.to_numeric(blood_data["d1"])
	blood_data["d12"] = pd.to_numeric(blood_data["d12"])

	blood_data.loc[(blood_data["hospital"]=='shangyu'),"redcell"]=blood_data[blood_data["hospital"]=="shangyu"]["redcell"]/100

	# 统计每天的总用血量
	# day_three = 0
	# day_all = 0
	# for i in range(3):
	# 	column = "d%d" % (i + 1)
	# 	day_three += blood_data[column].sum()
	# amount_list = []
	# for i in range(15):
	# 	column = "d%d" % (i + 1)
	# 	day_all += blood_data[column].sum()
	# 	amount_list.append(blood_data[column].sum())
	#
	# print(day_three/day_all)
	# fig1 = plt.figure(2)
	# rects =plt.bar(x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],height = amount_list,width = 0.8,align="center",yerr=0.000001)
	# plt.title("blood amount per day")
	# plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],["d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15"])
	# plt.show()



	# ------------------------------------------------------------------------

	blood_data.dropna(subset=["heart_rate","low_pressure","high_pressure","hemoglobin","redcell"],axis=0,how="all",inplace=True)

	numerical_list = ["age","weight","heart_rate","low_pressure","high_pressure","temperature","hemoglobin","redcell","albumin"]
	for temp in numerical_list:
		blood_data = fill_na(temp,blood_data)

	# blood_data["shock"] = blood_data["heart_rate"]/blood_data["high_pressure"]
	blood_data["penetrate"]=blood_data["penetrate"].fillna("N")
	blood_data["pleural"]=blood_data["pleural"].fillna("N")
	blood_data["ascites"]=blood_data["ascites"].fillna("N")
	blood_data["upperbody"]=blood_data["upperbody"].fillna("N")
	blood_data["lowerbody"]=blood_data["lowerbody"].fillna("N")
	blood_data["chest"]=blood_data["chest"].fillna("N")
	blood_data["abdomen"]=blood_data["abdomen"].fillna("N")
	blood_data["pelvis"]=blood_data["pelvis"].fillna("N")
	blood_data["c1"]=blood_data["c1"].fillna("N")
	blood_data["c2"]=blood_data["c2"].fillna("N")
	blood_data["c3"]=blood_data["c3"].fillna("N")



	# print(blood_data.info())
	# 预处理数据，删除不包含 心率、舒张压、收缩压、血红蛋白、红细胞压积的数据， 在范围内随机生成年龄和体重， 腹部受伤缺失数据填充"N"
	blood_data["penetrate"] = [1.0 if i == "Y" else 0.0 for i in blood_data["penetrate"]]
	blood_data["chest"] = [1.0 if i == "Y" else 0.0 for i in blood_data["chest"]]
	blood_data["abdomen"] = [1.0 if i == "Y" else 0.0 for i in blood_data["abdomen"]]
	blood_data["pelvis"] = [1.0 if i == "Y" else 0.0 for i in blood_data["pelvis"]]
	blood_data["c1"] = [1.0 if i == "Y" else 0.0 for i in blood_data["c1"]]
	blood_data["c2"] = [1.0 if i == "Y" else 0.0 for i in blood_data["c2"]]
	blood_data["c3"] = [1.0 if i == "Y" else 0.0 for i in blood_data["c3"]]
	blood_data["pleural"] = [1.0 if i == "Y" else 0.0 for i in blood_data["pleural"]]
	blood_data["ascites"] = [1.0 if i == "Y" else 0.0 for i in blood_data["ascites"]]
	blood_data["upperbody"] = [1.0 if i == "Y" else 0.0 for i in blood_data["upperbody"]]
	blood_data["lowerbody"] = [1.0 if i == "Y" else 0.0 for i in blood_data["lowerbody"]]

	blood_data.drop(labels=["blood_type"], axis=1, inplace=True)
	# blood_data.drop(labels=["patient_id","blood_type"], axis=1, inplace=True)

	blood_data["d1d2"] = blood_data[["d1", "d2"]].apply(lambda x: x["d1"] + x["d2"], axis=1)
	blood_data["d1d2d3"] = blood_data[["d1d2", "d3"]].apply(lambda x: x["d1d2"] + x["d3"], axis=1)

	blood_data = blood_data.drop(
		labels=["d1", "d2", "d3", "d1d2d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13",
		        "d14", "d15"], axis=1)

	return blood_data


def detect_outliers(df, features):
	outlier_indices = []
	for c in features:
		# 1st quartile:
		Q1 = np.percentile(df[c], 25)
		# 3rd quartile:
		Q3 = np.percentile(df[c], 75)
		# IQR:
		IQR = Q3 - Q1
		# Outlier step:
		outlier_step = IQR * 1.5
		# detect outlier and their indices:
		outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

		# store indices:
		outlier_indices.extend(outlier_list_col)
	outlier_indices = Counter(outlier_indices)
	multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)

	return multiple_outliers




def bar_plot(variable,blood_data):
	"""
	input: variable, example: "Sex"
	output: bar plot & value count
	"""
	# getting the feature
	var = blood_data[variable]
	# counting the number of categorical variables (value or sample)
	varValue = var.value_counts()
	# visualizing
	plt.figure(figsize=(9, 3))
	plt.bar(varValue.index, varValue)
	plt.xticks(varValue.index, varValue.index.values)
	plt.ylabel("Frequency")
	plt.title(variable)
	plt.show()
	print("{}: \n {}".format(variable, varValue))


def plot_hist(variable,name):
	plt.figure(figsize = (9,3))
	plt.hist(variable, bins =  80)
	plt.xlabel(name)
	plt.ylabel("Frequency")
	plt.title("{} distribution with hist".format(name))
	plt.show()




# ------------------------------------------------------------------------
# category1 = ["sex","penetrate","chest","abdomen","pelvis","pleural","ascites"]
# for c in category1:
# 	bar_plot(c)
# category2 = ["age","weight","heart_rate","low_pressure","high_pressure","temperature","hemoglobin","redcell","albumin"]
# for c in category2:
# 	plot_hist(blood_data[c],c)

# 1. 多例样本体温、心率血压等基本数据为0，处理为null
# 2. 上虞医院数据中的红细胞压积明显与其他医院单位不符,按百分比处理
# 3. 白蛋白有一例405，应该是小数点遗漏，处理为补上小数点40。5


# 样本分布      基本呈正态分布
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# heatmap of correlation
#
# f,ax = plt.subplots(figsize=(36,36))
# sns.heatmap(blood_data.corr(), annot=True)
# plt.show()
# 单变量关系    输血量与 腹部受伤、骨盆受伤、血红蛋白、心率、舒张压diastolic_pressure 相关度较高
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# for c in category1:
# 	print(c+":")
# 	print(blood_data[[c,"blood_amount"]].groupby([c], as_index=False).mean().sort_values(by="blood_amount",ascending=False))
# 平均值判断     性别、穿透伤、躯干受伤、胸腔积液不明显，    头颈受伤、腹部受伤、骨盆受伤、腹腔积液 明显
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# print(blood_data.loc[detect_outliers(blood_data,["heart_rate","low_pressure","high_pressure","hemoglobin","redcell"])])






#
#
# # 拆分出不需要用血的人，用血量为0样本过多，需要进行均衡
# blood_need = train_df[~train_df['d1d2'].isin([0])].copy()
# blood_no_need = train_df[train_df['d1d2'].isin([0])].copy()
#
#
#
#
#
# print(train_df.info())
# print("need blood amount:%d,no need blood amount:%d"%(len(blood_need),len(blood_no_need)))
# # blood_no_need = blood_no_need.sample(n=len(blood_need))
# # train_df = blood_need.append(blood_no_need)
# # print(len(train_df))
# # scale所有非离散特征，使其更符合正态分布
# scale_attributes=["age","weight","heart_rate","low_pressure","high_pressure","temperature","hemoglobin","redcell","albumin"]
# # for column in scale_attributes:
# #     if train_df[column].dtype == np.float64:
# #         plt.figure(figsize = (20, 3))
# #         sns.boxplot(x = train_df[column])
# #         plt.show()
#
# train_df[scale_attributes] = np.log1p(train_df[scale_attributes])

# 查看数据是否符合正态分布
# plt.rcParams['figure.figsize'] = (12.0, 6.0)
# fig = plt.figure()
# res = stats.probplot(train_df["d1d2"], plot=plt)
# plt.show()

#
#
#
#

#
#

#
#
#
# # -----------------------------------线性方案-----------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ----------------------------------------------------------------------
# def rmse_r2(model, y, y_predict):
# 	print('average loss is {}'.format(sum((y-y_predict))/len(y)))
# 	rmse = (np.sqrt(mean_squared_error(y, y_predict)))
# 	r2 = r2_score(y, y_predict)
# 	print('RMSE is {}'.format(rmse))
# 	print('R2 score is {}'.format(r2))
#
#
#
#
# def score_rmse_r2(model, X, y):
# 	rmse = cross_val_score(clf, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
# 	r2 = cross_val_score(clf, X, y, cv=5, scoring='r2').mean()
# 	print('RMSE is {}'.format(rmse))
# 	print('R2 score is {}'.format(r2))
#
#
#
#
# def get_predict(model, X_train, y_train, X_test, y_test):
# 	print('\nThe model performance for training set')
# 	print('--------------------------------------')
# 	y_predict = model.predict(X_train)
# 	rmse_r2(model, y_train, y_predict)
# 	print('\nThe model performance for testing set')
# 	print('--------------------------------------')
# 	y_predict = model.predict(X_test)
# 	rmse_r2(model, y_test, y_predict)
#
#
#
#
#
# def get_score_predict(model, X_train, y_train, X_test, y_test):
# 	print('\nThe model performance for training set')
# 	print('--------------------------------------')
# 	score_rmse_r2(model, X_train, y_train)
# 	print('\nThe model performance for validation set')
# 	print('--------------------------------------')
# 	score_rmse_r2(model, X_test, y_test)
#
#
#
#
#
# def test_score(model, X, y):
# 	print('\nThe model performance for testing set')
# 	print('--------------------------------------')
# 	score_rmse_r2(model, X, y)
#
#
#
# def get_model_grid_search(model, parameters, X, y, pipeline):
# 	# X = pipeline.fit_transform(X)
#
# 	random_search = RandomizedSearchCV(model,
# 	                                   param_distributions=parameters,
# 	                                   scoring='r2',
# 	                                   verbose=1, n_jobs=-1,
# 	                                   n_iter=1000)
#
# 	grid_result = random_search.fit(X, y)
#
# 	print('Best R2: ', grid_result.best_score_)
# 	print('Best Params: ', grid_result.best_params_)
#
# 	return random_search.best_estimator_
#
#
#
#
# def get_model_random_search(model, parameters, X, y, pipeline=None):
# 	# X = pipeline.fit_transform(X)
# 	clf = GridSearchCV(model, parameters, scoring='r2', cv=5, verbose=1, n_jobs=-1)
# 	grid_result = clf.fit(X, y)
#
# 	print('Best R2: ', grid_result.best_score_)
# 	print('Best Params: ', grid_result.best_params_)
#
# 	return clf.best_estimator_
#
#
#
#
# def k_fold_score(model, X, y, pipeline):
# 	kf = KFold(n_splits=5)
# 	average_list = []
# 	rmse_list = []
# 	r2_list = []
# 	xaxis=[]
# 	for train_index, test_index in kf.split(X, y):
# 		xaxis.append(len(xaxis)+1)
# 		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
# 		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
# 		# X_train = pipeline.fit_transform(X_train)
# 		# X_test = pipeline.transform(X_test)
#
# 		model.fit(X_train, y_train)
# 		y_predict = model.predict(X_test)
#
# 		average = sum((y_test-y_predict))/len(y_test)
# 		rmse = (np.sqrt(mean_squared_error(y_test, y_predict)))
# 		r2 = r2_score(y_test, y_predict)
# 		average_list.append(average)
# 		rmse_list.append(rmse)
# 		r2_list.append(r2)
# 	average_list = np.array(average_list)
# 	rmse_list = np.array(rmse_list)
# 	r2_list = np.array(r2_list)
#
# 	print('--------------------------------------')
# 	print('average is {}'.format(average_list.mean()))
# 	print('RMSE is {}'.format(rmse_list.mean()))
# 	print('R2 score is {}'.format(r2_list.mean()))
# 	plt.plot(xaxis, average_list, color='red', linewidth=2.0, linestyle='--')
# 	plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.2)
# #
# # # Linear Regressor----------------------------------------
# # #
# # lin_model = LinearRegression()
# # lin_model.fit(X_train, y_train)
# # get_predict(lin_model,X_train,y_train,X_test,y_test)
# #
# # params = {
# #     'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
# #     'l1_ratio':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
# # }
# #
# # en = ElasticNet()
# # en_model = get_model_grid_search(en, params, data_gs, target_gs,None)
# # k_fold_score(en_model, data_cv, target_cv, None)
# #
# # en_rs_model = get_model_random_search(en, params, data_gs, target_gs, None)
# # k_fold_score(en_rs_model, data_cv, target_cv, None)
# #
# #
# #
#
# # # SVR-------------------------------------------
# # print("basic svr------------------------------------")
# # svr = SVR(kernel='rbf')
# # svr.fit(X_train, y_train)
# # get_predict(svr,X_train,y_train,X_test,y_test)
# #
# #
# # print("grid svr------------------------------------")
# # # SVR gridsearch的结果最好
# # params = {  'C': [0.1, 1, 100, 1000],
# #             'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
# #             'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
# #         }
# #
# # svr = SVR(kernel='rbf')
# # svr_model = get_model_grid_search(svr, params, data_gs, target_gs, None)
# # k_fold_score(svr_model, data_cv, target_cv, None)
#
# # print("random svr------------------------------------")
# # params = {  'C': [0.1, 1, 100, 1000],
# #             'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
# #             'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
# #         }
# #
# # svr = SVR(kernel='rbf')
# # svr_rs_model = get_model_random_search(svr, params, data_gs, target_gs, None)
# # k_fold_score(svr_rs_model, data_cv, target_cv, None)
#
#
#
#
# # Decision Tree Regressor-------------------------------------------
# # print("basic DT------------------------------------")
# # tree = DecisionTreeRegressor()
# # tree.fit(X_train, y_train)
# # get_predict(tree, X_train, y_train, X_test, y_test)
# #
# # print("grid DT------------------------------------")
# # params = {'min_samples_split': range(2, 10)}
# # tree = DecisionTreeRegressor()
# # tree_model = get_model_grid_search(tree, params, data_gs, target_gs, None)
# # k_fold_score(tree_model, data_cv, target_cv, None)
# #
# # print("random DT------------------------------------")
# # params = {'min_samples_split': range(2, 10)}
# #
# # tree = DecisionTreeRegressor()
# # tree_rs_model = get_model_random_search(tree, params, data_gs, target_gs, None)
# # k_fold_score(tree_rs_model, data_cv, target_cv, None)
#
#
#
#
#
# # -----------------------------------分类方案-----------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ------------------------------------------------------------------------
# # ----------------------------------------------------------------------
def amount_map(blood_amount):
    if blood_amount ==0:
        return 0
    elif blood_amount > 0 and blood_amount <= 10:
        return 1
    else:
        return 2

# def amount_map(blood_amount):
#     if blood_amount ==0:
#         return 0
#     else:
#         return 1

def sum_caculate(result_list,class_num):
	amount_level_5 = [0, 3.41,15.7]            # 分级的倍数
	amount_level_2 = [0,5]
	final=0
	if class_num==2:
		for item in result_list:
			final+=amount_level_2[item]
	if class_num==5:
		for item in result_list:
			final+=amount_level_5[item]
	return final


def predict(X_train, X_test, y_train, y_test):
	y_train = y_train.apply(amount_map)
	logreg = XGBClassifier()
	logreg.fit(X_train, y_train)
	predictions=logreg.predict(X_test)
	Y_pred_prob = logreg.predict_proba(X_test)
	gt = y_test
	y_test = y_test.apply(amount_map)
	return gt.sum()/len(y_test),sum_caculate(y_test,2)/len(y_test),sum_caculate(predictions,2)/len(y_test)


def learning(X,Y):
	y_train = Y.apply(amount_map)
	logreg = XGBClassifier()
	logreg.fit(X, y_train)
	return logreg

def run():
	# # ------------------------------------------------------------------------
	# 读取训练数据
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width',None)

	# blood_file_path = "all_hospital_v1_v2.csv"
	blood_file_path = "./all_hospital_v1_v2_for_article.xls"
	learning_data = pd.read_excel(blood_file_path)
	learning_data = preprocess(learning_data)

	# 浙二数据和其他医院数据分开
	# learning_data = learning_data[learning_data['hospital'].isin(["zheer2018","zheer2019","zheer2020"])].copy()
	learning_data.drop(labels=["hospital"], axis=1, inplace=True)

	# other_data = blood_data[~blood_data['hospital'].isin(["zheer2018","zheer2019","zheer2020"])].copy()

	# 删除部分outlinear数据
	outliers = detect_outliers(learning_data,["heart_rate","low_pressure","high_pressure","hemoglobin","redcell","albumin"])
	learning_data = learning_data.drop(outliers, axis = 0).reset_index(drop=True)

	# ------------------------------------------------------------------------
	# 训练模型

	X = learning_data.copy().drop(labels = ["d1d2"], axis = 1)
	Y = learning_data["d1d2"].copy()
	model = learning(X,Y)

	# ------------------------------------------------------------------------
	# 使用模型

	#读取测试数据
	# test_file_path = "zheer.csv"
	# test_file_path = input("输入测试数据文件:")
	# randomnumber = int(input("输入随机选取人数:"))
	num = int(input("测试次数:"))
	test_file_path = './all_hospital_v1_v2_for_article.xls'
	print("读取的测试数据：%s"%test_file_path)

	test_data = pd.read_excel(test_file_path)
	test_data = preprocess(test_data)

	ground_truth, predict_truth = [], []
	for i in range(num):
		randomnumber = np.random.randint(100)
		test_data_use = test_data.sample(n=randomnumber)
		test_data_gt = test_data_use["d1d2"].copy()
		test_data_use.drop(labels=["hospital","d1d2"], axis=1, inplace=True)

		predictions = model.predict(test_data_use)
		predict_result = sum_caculate(predictions, 5)
		
		ground_truth.append(test_data_gt.sum())
		predict_truth.append(predict_result)
		print("测试数据的规模:%d人   测试数据的实际用血量:%s   模型预测用血量:%s"%(len(test_data_gt), test_data_gt.sum(), predict_result))

	mse = mean_squared_error(ground_truth, predict_truth) ** 0.5
	print("%d次实验，mse误差为: %f"%(num, mse))


# def temp():
# 	precision_score_list=[]
# 	recall_score_list=[]
# 	f1_score_list=[]
#
# 	transform_loss_list = []
# 	real_loss_list = []
# 	x_axis = []
#
# 	# y_train = train_df["d1d2"]
# 	# X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)
#
# 	r2_y_mean=0
#
# 	r2_y=[]
# 	r2_f=[]
#
#
# 	epoch=1
# 	for i in range(epoch):
# 		x_axis.append(i)
# 		y_train = train_df["d1d2"]
# 		X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)
# 		gt,class_tf,final = predict(X_train, X_test, y_train, y_test)
# 		transform_loss_list.append((gt-class_tf))
# 		real_loss_list.append((gt-final))
# 		r2_y_mean+=gt
# 		r2_y.append(gt)
# 		r2_f.append(class_tf)
#
# 	y_mean=r2_y_mean/epoch
# 	print(y_mean)
# 	print(r2_y)
# 	print(r2_f)
# 	sst=0
# 	sse=0
# 	ssr=0
# 	for i in range(epoch):
# 		sst += (r2_y[i]-y_mean)*(r2_y[i]-y_mean)
# 		sse += (r2_y[i] - r2_f[i]) * (r2_y[i] - r2_f[i])
# 		ssr += (r2_f[i] - y_mean)*(r2_f[i] - y_mean)
#
# 	r2 = 1-sse/sst
#
#
# 	print(r2)
#
# 	plt.plot(x_axis,transform_loss_list,color='red',linewidth=2.0,linestyle='--')
# 	plt.plot(x_axis,real_loss_list,color='blue',linewidth=3.0,linestyle='-.')
# 	plt.show()




# -----------------------------------找寻最优量化-----------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ----------------------------------------------------------------------
# def sum_caculate(result_list,amount_level):
# 	final=0
# 	for item in result_list:
# 		final+=amount_level[item]
# 	return final
#
#
#
#
# amount_level = [0, 3.41, 15.7]            # 分级的倍数
# loss_list=[]

# x_axis=[]
# for i in range(50):
# 	x_axis.append(i)
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 	y_class = y_test.apply(amount_map)
# 	gt = y_test.sum()
# 	predict = sum_caculate(y_class,amount_level)
# 	average_loss = abs(gt-predict)/len(y_test)
# 	loss_list.append(average_loss)

# train_df["class"] = y.apply(amount_map)
# print(train_df[train_df["class"]==2]["d1d2"].sum()/len(train_df[train_df["class"]==2]))
#
# plt.plot(x_axis,loss_list,color='red',linewidth=2.0,linestyle='--')
# plt.show()

if __name__ == "__main__":
	run()