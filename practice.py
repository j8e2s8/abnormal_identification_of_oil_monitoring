import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from scipy.stats import ranksums
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping
# import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier




# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project3/data_week3.csv


def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
	

kde(df)

kde(df[df['unknown1']=='type1'])
kde(df[df['unknown1']=='type2'])
kde(df[df['unknown1']=='type3'])
kde(df[df['unknown1']=='type4'])


# 데이터 정보 확인
df.info()

for i in df.columns:
	print(f'{i}컬럼의 unique 개수 :',len(df[i].unique()))
      

cols = ['unknown1','unknown4','unknown5','unknown15','target']
for i in cols:
	print(f'{i}컬럼의 unique :', np.sort(df[i].unique()))


df['target'] = df['target'].astype('boolean')



df.describe()


num_df = df.select_dtypes(include=('number'))

#
ranksum_p = []
variable = df.select_dtypes(include=('number')).columns

for i in variable:
	temp = ranksums(df.loc[()])
	







# 이상치 여부 컬럼 만들기
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)

    return pd.concat([lower_bound, upper_bound], axis = 1).T


num_df = df.select_dtypes(include=('int', 'float'))
variable = num_df.columns

for col in variable:
	df[f'{col}_outlier'] = np.where((df[col]<IQR_outlier(num_df).loc[0,col])|(df[col]>IQR_outlier(num_df).loc[1,col]),1,0)


for col in variable:
	group_df = df.groupby(f'{col}_outlier').agg(불량수 = ('target','sum'), 데이터수 = ('target','count'))
	group_df['불량 비율'] = np.round(group_df['불량수']/group_df['데이터수'],2)
	print("\n",group_df,"\n","-"*40)


# outlier 0 인 것만 뽑아서 하면, 결국엔 이상치 제거하고 나서 돌리는 거랑 마찬가지


# 모든 컬럼에 대해서 이상치인 부분은 0개임.
outlier_df = df[(df['unknown2_outlier']==1) & (df['unknown3_outlier']==1) & (df['unknown4_outlier']==1) & \
	(df['unknown5_outlier']==1) & (df['unknown6_outlier']==1) & (df['unknown7_outlier']==1) & \
		(df['unknown8_outlier']==1) & (df['unknown9_outlier']==1) & (df['unknown10_outlier']==1) & \
			(df['unknown11_outlier']==1) & (df['unknown12_outlier']==1) & (df['unknown13_outlier']==1) & \
				(df['unknown14_outlier']==1) & (df['unknown15_outlier']==1) & (df['unknown16_outlier']==1) & \
					(df['unknown17_outlier']==1)]
outlier_df.shape

for col in variable:
	print("이상치 개수 :",df[f'{col}_outlier'].sum())     # group_df에 이미 나와있음


def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌












def cat_count(df, palette='dark'):
    cat_col = df.select_dtypes(include=['object','boolean','bool']).columns.to_list()
    num_col = df.select_dtypes(include=['int','float']).columns.to_list()
    select_num_col = [i for i in num_col if len(df[i].unique())<=15]
    total_col = cat_col + select_num_col
    w_n = len(total_col)
    n = int(np.ceil(w_n/4))
    plt.clf()
    plt.figure(figsize=(6*4, 5*n))
    for i, col in enumerate(total_col, 1): 
        # plt.figure(figsize=(6*4, 5*n))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, i)
        if df[col].dtypes in ['bool','boolean', 'int', 'float']:
            ax = sns.countplot(df[col].astype('str'), order=df[col].value_counts().sort_values().index, palette=palette)
        else:
            ax = sns.countplot(df[col], order=df[col].value_counts().sort_values().index, palette=palette)
        for p in ax.patches:
            plt.text(p.get_width(), p.get_y() + p.get_height()/2 , p.get_width())
        plt.title(f'{col}의 범주별 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
	
cat_count(df)

cat_count(X_train, palette='coolwarm')
cat_count(X_train, palette='pastel')
cat_count(X_train, palette='dark')







import pandas as pd
from scipy.stats import chi2_contingency

# unknown1 에 대해서
# 교차표 생성
contingency_table = pd.crosstab(df['unknown1'], df['target'])

# 카이제곱 검정을 위한 예상 빈도 계산
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 예상 빈도 출력
print("예상 빈도:")
print(expected)

# 각 예상 빈도가 5 이상인지 확인
if (expected < 5).any():
    print("예상 빈도 중 5 미만인 값이 있습니다.")
else:
    print("모든 예상 빈도가 5 이상입니다.")
	

print("p-값 :", p)

# 유의수준 설정 (예: 0.05)
alpha = 0.05

# 귀무가설 채택 여부 판단
if p < alpha:
    print("귀무가설 기각: 독립변수와 타겟 변수는 독립적이지 않다. 관련이 있다.")
else:
    print("귀무가설 채택: 독립변수와 타겟 변수는 독립적이다.")





# type별로 unknown17 분포를 고려해 범주화하는 함수
def categorize_unknown17(row):
    if row['unknown1'] == 'type1':
        # type1에 맞는 구간 나누기
        if row['unknown17'] < 500:
            return 'low'
        elif row['unknown17'] < 1000:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1'] == 'type2':
        # type2에 맞는 구간 나누기
        if row['unknown17'] < 300:
            return 'low'
        elif row['unknown17'] < 750:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1'] == 'type3':
        # type3에 맞는 구간 나누기
        if row['unknown17'] < 200:
            return 'low'
        elif row['unknown17'] < 500:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1'] == 'type4':
        # type4에 맞는 구간 나누기
        if row['unknown17'] < 600:
            return 'low'
        elif row['unknown17'] < 1200:
            return 'medium'
        else:
            return 'high'

# 범주화 적용
df['unknown17_type_n'] = df.apply(categorize_unknown17, axis=1)  # 행에 적용
df.columns



# 모든 범주 컬럼에 대해서
# 교차표 생성
select_col = df.columns.drop(variable).drop('target')
for col in select_col:
	contingency_table = pd.crosstab(df[col], df['target'])
	chi2, p, dof, expected = chi2_contingency(contingency_table)

	# 예상 빈도 출력
	print(f"{col}의 예상 빈도:")
	print(expected)

	# 각 예상 빈도가 5 이상인지 확인
	if (expected < 5).any():
		print(f"{col}의 예상 빈도 중 5 미만인 값이 있습니다.")
	else:
		print(f"{col}의 모든 예상 빈도가 5 이상입니다.")
		
	print(f"{col}의 p-값 :", p)

	# 귀무가설 채택 여부 판단
	if p < 0.05:
		print(f"{col}의 귀무가설 기각: 독립변수와 타겟 변수는 독립적이지 않다. 관련이 있다.", "\n","-"*30)
	else:
		print(f"{col}의 귀무가설 채택: 독립변수와 타겟 변수는 독립적이다.", "\n","-"*30)



cat_count(df)







# 이상치 분석
from sklearn.ensemble import IsolationForest

# Isolation Forest 모델 생성 및 학습
iso_forest = IsolationForest(contamination=0.05)  # 5%의 데이터를 이상치로 간주
df['outlier_flag'] = iso_forest.fit_predict(df[['variable']])

# Isolation Forest는 -1을 이상치로, 1을 정상 데이터로 반환
df['outlier_flag'] = df['outlier_flag'].map({1: 0, -1: 1})

print(df)




