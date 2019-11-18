### Description:
这是Kaggle的入门赛， 通过这个比赛，可以学会一些处理数据的基本流程和方法，包括如何理解数据， 特征相关性分析，缺失值处理，特征组合等，然后建立模型进行预测等，学会用机器学习方法处理问题的基本流程。
> 文件说明：
>> * 这里面共有四个HousePrice笔记本，第一个版本是拿到赛题之后，如何分析的基础版，第二个版本是加入了特征工程和模型的提升版，第三个和第四个是根据别人的
思路进行改进的一个版本
>> * 还有两个知识补充笔记本，这两个是在学习比赛的过程中进行的一个知识的整理补充。

赛题链接： [房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)
>思路分享(有以下问题要解决）：
>> 1. 数据明确 -> 查看数据集和数据的描述，理解意思 -> 解决问题的思路
>> 2. 数据明确之后，要想这是个什么问题？ （回归做预测）
>> 3. 这时候要想起哪些算法可以做回归（线性回归、决策树、XGB、SVM等）
>> 4. 数据中是否含有字符串或者缺失值？ 如何把它们变成数值型的？
>> 5. 数据特征工程的思路， 数据的EDA(pandas_profling)探索性数据分析、特征选择、再就是数据特征组合或者特征分割等
>> 6. 然后就是算法的选择
>
> 思路展开： 数据探索， 做一点点的修改 -> 数据清洗（空值的填充） --> 数据预处理（数据归一化，标准化） --> 模型的构建 --> 训练预测 -- 保存提交

下面就开始一步一步的进行分析：[根据我整理的一个板子](https://blog.csdn.net/wuzhongqiang/article/details/103116953)
### 第一步：定义问题
这个问题很显然是一个回归预测问题，这时候，就要先联想到处理回归的问题可以使用的一些模型。 然后看看最后赛题要求提交的最后的格式，一个文件，评估方式是均方误差损失。
明白了一些基本的问题之后，就可以导入包和数据集。

```python
"""导入包"""
import pandas as pd
import numpy as np
import pandas_profiling as ppf

# 数据预处理
from sklearn.preprocessing import RobustScaler, StandardScaler#去除异常值与数据标准化
from sklearn.preprocessing import Imputer
from scipy.stats import skew  # for some statistics  偏度
from scipy.stats import boxcox_normmax
from scipy.special import boxcox1p
from mlxtend.regressor import StackingCVRegressor

# 模型选择
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

# 创建模型
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV, LassoCV, RidgeCV, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 管道机制
from sklearn.pipeline import Pipeline, make_pipeline

# 集成技术
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
```

```python
"""导入数据"""
train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
test_ID = test['Id']
train_size = train.shape[0]
```
### 第二步：理解数据
这一步很重要，真正的理解了数据之后才能进行正确的处理，这里对目标变量进行分析

```python
f,ax = plt.subplots(1,2,figsize=(16,6))
sns.distplot(train['SalePrice'],fit=norm,ax=ax[0])
sns.boxplot(train['SalePrice'])
plt.show()

#skewness and kurtosis
print("Skewness: {}".format(train['SalePrice'].skew()))
print("Kurtosis: {}".format(train['SalePrice'].kurt()))
print("--------------------------------------")
print(train['SalePrice'].describe())
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118102825625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
Observations
> * 目标变量不是正态分布
> * 它高度偏斜
> * 平均售价为180921美元，拉向离群值的上端。
> * 中位数163000美元，低于平均值。
> * 它的上端有几个异常值

下面分析特征与目标变量之间的相关性

```python
f,ax = plt.subplots(1,2,figsize=(16,4))
sns.boxplot(train['GrLivArea'],ax=ax[0])
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel('GrLiveArea')
plt.ylabel('SalePrice')
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118103119990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
上面略去了几步相似的分析，具体的可以见GitHub代码和分析，上面有几个更全的思路。
```python
"""Finding numeric features"""
numeric_cols = train.select_dtypes(exclude='object').columns
numeric_cols_length = len(numeric_cols)  

fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 120))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)

# skiped Id and saleprice feature
for i in range(1,numeric_cols_length-1):
    feature = numeric_cols[i]
    plt.subplot(numeric_cols_length, 3, i)
    sns.scatterplot(x=feature, y='SalePrice', data=train)
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
           
plt.show()
```
Observations:
> * MSSubClass,MoSold,YrSold—模式显示它作为一个类别和描述的意思是一样的
> * OverallQual, OverallCond—有序值(如评级)
>* BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,Fireplaces,BedroomAbvGr,KitchenAvbGr - discrete value(no. of bathrooms)

```python
corr = train.select_dtypes(include='number').corr()
plt.figure(figsize=(16,6))
corr['SalePrice'].sort_values(ascending=False)[1:].plot(kind='bar')
plt.tight_layout()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111810332572.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
上面画出了特征与目标变量相关性排名
```python
"""Correlation of top 10 feature with saleprice"""
corWithSalePrice = train.corr().nlargest(10,'SalePrice')['SalePrice'].index
f , ax = plt.subplots(figsize = (18,12))
ax = sns.heatmap(train[corWithSalePrice].corr(), annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()
```
下面这个图是特征和特征之间的相关性，根据这个图，就可以选择出相关性强的，删除掉冗余特征。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118103411970.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
Observations:
> * Overallqual、GrLivArea、GarageCars、GarageArea和TotalBsmtSF与销售价格有很强的相关性。
> * GarageCas和GarageArea是紧密相关的，这是因为车库的车停在GarageArea。
> * TotRmsAbvGrd和GrLivArea是密切相关的。当地面面积增加时，房间的数量也增加。
> * TotalBsmtSF和1stFlrSF是密切相关的。

```python
#Log - transformation
y = np.log1p(train['SalePrice'])

f,ax = plt.subplots(1,2,figsize=(16,4))
sns.distplot(y,fit=norm,ax=ax[0])
stats.probplot(y,plot=plt)
plt.show()

#skewness and kurtosis
print("Skewness: {}".format(y.skew()))
print("Kurtosis: {}".format(y.kurt()))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118103714825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
这里通过一系列处理之后，发现目标变量的分布呈现正态分布。 可以和第一张图片进行一个对比。
### 第三步：数据的准备
1. 填充缺失值

```python
#visualize missing data
missing_value = all_features.isnull().sum().sort_values(ascending=False) / len(all_features) * 100
missing_value = missing_value[missing_value != 0]
missing_value = pd.DataFrame({'Missing value' :missing_value,'Type':missing_value.index.map(lambda x:all_features[x].dtype)})
missing_value.plot(kind='bar',figsize=(16,4))
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118104013137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
```python
print("Total No. of missing value {} before Imputation".format(sum(all_features.isnull().sum())))
def fill_missing_values():
 
    fillSaleType = all_features[all_features['SaleCondition'] == 'Normal']['SaleType'].mode()[0]
    all_features['SaleType'].fillna(fillSaleType,inplace=True)

    fillElectrical = all_features[all_features['Neighborhood']=='Timber']['Electrical'].mode()[0]
    all_features['Electrical'].fillna(fillElectrical,inplace=True)

    exterior1_neighbor = all_features[all_features['Exterior1st'].isnull()]['Neighborhood'].values[0]
    fillExterior1 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior1st'].fillna(fillExterior1,inplace=True)

    exterior2_neighbor = all_features[all_features['Exterior2nd'].isnull()]['Neighborhood'].values[0]
    fillExterior2 = all_features[all_features['Neighborhood'] == exterior1_neighbor]['Exterior1st'].mode()[0]
    all_features['Exterior2nd'].fillna(fillExterior2,inplace=True)

    bsmtNeigh = all_features[all_features['BsmtFinSF1'].isnull()]['Neighborhood'].values[0]
    fillBsmtFinSf1 = all_features[all_features['Neighborhood'] == bsmtNeigh]['BsmtFinSF1'].mode()[0]
    all_features['BsmtFinSF1'].fillna(fillBsmtFinSf1,inplace=True)

    kitchen_grade = all_features[all_features['KitchenQual'].isnull()]['KitchenAbvGr'].values[0]
    fillKitchenQual = all_features[all_features['KitchenAbvGr'] == kitchen_grade]['KitchenQual'].mode()[0]
    all_features['KitchenQual'].fillna(fillKitchenQual,inplace=True)
        
    all_features['MSZoning'] = all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
       
    all_features['LotFrontage'] = all_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2','PoolQC']:
        all_features[col] = all_features[col].fillna('None')
    
    categorical_cols =  all_features.select_dtypes(include='object').columns
    all_features[categorical_cols] = all_features[categorical_cols].fillna('None')
    
    numeric_cols = all_features.select_dtypes(include='number').columns
    all_features[numeric_cols] = all_features[numeric_cols].fillna(0)
    
    all_features['Shed'] = np.where(all_features['MiscFeature']=='Shed', 1, 0)
    
    #GarageYrBlt -  missing values there for the building which has no Garage, imputing 0 makes huge difference with other buildings,
    #imputing mean doesn't make sense since there is no Garage. So we'll drop it
    all_features.drop(['GarageYrBlt','MiscFeature'],inplace=True,axis=1)
    
    all_features['QualitySF'] = all_features['GrLivArea'] * all_features['OverallQual']

fill_missing_values()

print("Total No. of missing value {} after Imputation".format(sum(all_features.isnull().sum())))
```
2. 修复偏斜特征

```python
# converting some numeric features to string
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)


# Filter the skewed features
numeric = all_features.select_dtypes(include='number').columns
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)

# Normalize skewed features using boxcox
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))
```
3. 特征组合和创造

```python
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']

all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']
all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])

# Exponential features 

all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
```
4. 特征转换

```python
# There is a natural order in their values for few categories, so converting them to numbers gives more meaning
quality_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
quality_cols = ['BsmtQual', 'BsmtCond','ExterQual', 'ExterCond','FireplaceQu','GarageQual', 'GarageCond','KitchenQual','HeatingQC']
for col in quality_cols:
    all_features[col] = all_features[col].replace(quality_map)

all_features['BsmtExposure'] = all_features['BsmtExposure'].replace({"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3})

all_features["PavedDrive"] =all_features["PavedDrive"].replace({"N" : 0, "P" : 1, "Y" : 2})

bsmt_ratings = {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
bsmt_col = ['BsmtFinType1','BsmtFinType2']
for col in bsmt_col:
    all_features[col] = all_features[col].replace(bsmt_ratings)

    
all_features["OverallScore"]   = all_features["OverallQual"] * all_features["OverallCond"]
all_features["GarageScore"]    = all_features["GarageQual"] * all_features["GarageCond"]
all_features["ExterScore"]     = all_features["ExterQual"] * all_features["ExterCond"]
```
6. 划分数据集

```python
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]

outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
train_labels = train_labels.drop(y.index[outliers])
```
### 第四步：训练模型

```python
kf = KFold(n_splits=12, random_state=42, shuffle=True)
# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006,random_state=42)

# StackingCVRegressor 
stackReg = StackingCVRegressor(regressors=(xgboost, svr, ridge, gbr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True,random_state=42)
```
### 第五步：预测和提交结果

```python
pred = np.floor(np.expm1(blended_predictions(X_test, [0.15, 0.2,0.1,0.15,0.1,0.3])))

result = pd.DataFrame({'Id':test_ID, 'SalePrice':pred})
result.to_csv("result/submission5.csv",index=False)

"""这个误差率0.11436"""
```
### 总结
这是房价预测的处理路程和思路，看了好几个大佬的思路，基本上处理流程都是这个流程，只不过，特征预处理的方式都各有不同，比如缺失值的填充方式，特征的组合方式等，然后模型训练的时候，大部分都是采用集成或者融合的技术。 通过这次比赛学习到了很多知识，上面的代码，我也只是抽取来很简单的一部分进行说明，下面总结一下，学习到的一些知识或者经验，我觉得这个对我来说才更重要。
> * 首先是机器学习的板子[https://blog.csdn.net/wuzhongqiang/article/details/103116953](https://blog.csdn.net/wuzhongqiang/article/details/103116953)
> * 学习到了一些比较好用的可视化的工具，seaborn，数据EDA分析的工具pandas_profiling
> * 处理数据的方式： 从目标变量的研究出发，去研究目标变量的分布和特征与特征，特征与目标变量之间的相关性，然后去进行筛选特征，特征刷选完之后再进行数据清洗等操作，比一上来就进行数据清洗操作要好的多
> * 数据转换成正态还是很重要的，学习到了一系列操作，比如标签转换成对数，或者是修改偏斜度（boxcox）等，特征也可以进行对数操作等
> * 填充缺失值的时候，可以分数据类型进行填充，字符的用None，数字的用0或者中位数或者众数等
>  * 有一些特征虽然是数字型的，但是没有数字的意义，可以适当转成LabelEncoder
>  * 一些特征有很强的相关性，这时候可以进行组合
>  模型的构建， 现在大多数使用模型集成或者堆叠的方式，即使XGB也不太好单独使用，这一块需要重点的学学。[https://www.aboutyun.com/thread-27205-1-1.html](https://www.aboutyun.com/thread-27205-1-1.html)
