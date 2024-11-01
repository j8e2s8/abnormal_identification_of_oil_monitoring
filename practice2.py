import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, make_scorer, precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project3/data_week3.csv




def hist(df):
	numeric_cols = df.select_dtypes(include=['number']).columns
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.histplot(df[col], stat='density')
		plt.title(f'{col}의 히스토그램 분포', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


hist(df)





# 자료형 바꾸기
df['target'] = df['target'].astype('boolean')


# 데이터와 타겟 변수 설정 (예: X는 설명 변수, y는 타겟 변수)
X = df.drop('target', axis=1)  # 설명 변수들
y = df['target']  # 타겟 변수








# -----------------------------------------   기존 컬럼들 공통 전처리
# 더미코딩 
X['unknown1_original'] = X['unknown1']
X = pd.get_dummies(X, columns=['unknown1'], drop_first=True)


# train, test 80%, 20% split          # X, y, X_train, X_test, y_train, y_test 명 통일하기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print("훈련 데이터 크기:", X_train.shape, y_train.shape)
print("테스트 데이터 크기:", X_test.shape, y_test.shape)



# -----------------------------------------   이상치 여부 컬럼 추가
# 이상치 여부 컬럼 만들기
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df


num_X_train = X_train.select_dtypes(include=('int', 'float'))
variable = num_X_train.columns

for col in variable:
	X_train[f'{col}_outlier'] = np.where((X_train[col]<IQR_outlier(num_X_train).loc[0,col])|(X_train[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)
	X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train).loc[0,col])|(X_test[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)

# 'unknown11' -10.5  , 17.5



# -----------------------------------------   분포 구간화 컬럼 추가
def categorize_unknown17(row):
    if row['unknown1_original'] == 'type1':
        # type1에 맞는 구간 나누기
        if row['unknown17'] < 500:
            return 'low'
        elif row['unknown17'] < 1000:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1_original'] == 'type2':
        # type2에 맞는 구간 나누기
        if row['unknown17'] < 300:
            return 'low'
        elif row['unknown17'] < 750:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1_original'] == 'type3':
        # type3에 맞는 구간 나누기
        if row['unknown17'] < 200:
            return 'low'
        elif row['unknown17'] < 500:
            return 'medium'
        else:
            return 'high'
    elif row['unknown1_original'] == 'type4':
        # type4에 맞는 구간 나누기
        if row['unknown17'] < 600:
            return 'low'
        elif row['unknown17'] < 1200:
            return 'medium'
        else:
            return 'high'

# 범주화 적용
X_train['unknown17_type_n'] = X_train.apply(categorize_unknown17, axis=1)  # 행에 적용
X_train.columns
X_test['unknown17_type_n'] = X_test.apply(categorize_unknown17, axis=1)  # 행에 적용
X_test.columns





# 모든 범주 컬럼에 대해서 검정
# 교차표 생성
num_X_train = X_train.select_dtypes(include=('int', 'float'))
variable = num_X_train.columns
select_col = X_train.columns.drop(variable)
for col in select_col:
	contingency_table = pd.crosstab(X_train[col], y_train)
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







# -----------------------------------------   모델 적용
# X_train, X_test 에서 'unknown1_original' 컬럼 제거하기
X_train = X_train.drop('unknown1_original', axis=1)
X_test = X_test.drop('unknown1_original', axis=1)

# 모델 바꿔서 적용
log_clf = LogisticRegression(class_weight='balanced', random_state=42)
log_clf.fit(X_train, y_train)


# test set 예측 : 1이 나올 확률로 예측
prob_y = log_clf.predict_proba(X_test)[:, 1]



# -----------------------------------------   임계값별 성능평가 -> 임계값 정하기
# 임계값별로 모델 성능 목록 생성
thresholds = np.arange(0, 1.1, 0.1)
results = []

roc_auc = roc_auc_score(y_test, prob_y)
for threshold in thresholds:
    pred_y_threshold = (prob_y >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_y_threshold).ravel()
    
    precision = precision_score(y_test, pred_y_threshold)
    recall = recall_score(y_test, pred_y_threshold)
    f1 = f1_score(y_test, pred_y_threshold)  # F1 Score 계산
    fpr = fp / (fp + tn)
    g_mean = geometric_mean_score(y_test, pred_y_threshold)
    
    # 결과 저장
    results.append({
        'Threshold': threshold,
        'Predicted Positive N': tp,
        'Actual Positive N': tp + fn,
        'Predicted Negative N': tn,
        'Actual Negative N': tn + fp,
        'Precision': precision,
        'Recall': recall,
        'G-Mean': g_mean,
        'F1 Score': f1,
        'FPR': fpr,
        'ROC AUC': roc_auc 
    })

results_df = pd.DataFrame(results)
results_df







#


# VIF 계산 함수
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# VIF가 10을 넘는 변수를 제거하는 함수
def remove_high_vif(X, threshold=10):
    vif_data = calculate_vif(X)
    
    # VIF 값이 threshold를 초과하는 변수들이 존재할 동안 반복
    while vif_data['VIF'].max() > threshold:
        # VIF 값이 가장 높은 변수 제거
        drop_col = vif_data.sort_values('VIF', ascending=False).iloc[0]['Feature']
        print(f'제거된 변수: {drop_col}, VIF: {vif_data.loc[vif_data["Feature"] == drop_col, "VIF"].values[0]}')
        
        # 해당 변수를 제거한 데이터로 다시 VIF 계산
        X = X.drop(columns=[drop_col])
        vif_data = calculate_vif(X)
    
    return X, vif_data

# X_train은 설명 변수들로 구성된 DataFrame이라고 가정
X_cleaned, final_vif = remove_high_vif(X_train, threshold=10)

# 최종 VIF 확인
print(final_vif)






# 다중공선성 확인하는 함수
def calculate_vif(num_X_train):
    # Boolean 컬럼을 int로 변환 (True -> 1, False -> 0)
    X_vif = num_X_train.astype({col: 'int' for col in num_X_train.select_dtypes(include='bool').columns})
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    return vif_data

# 더미변수, boolean 변수, 수치 변수에 대해 다중공선성 확인하기
num_bool_X_train = X_train.select_dtypes(include = ('number','bool'))
vif_df = calculate_vif(num_bool_X_train)
vif_df[vif_df['VIF']>10]['Feature']

# 수치 변수에 대해서만 다중공선성 확인하기
int_X_train = X_train.select_dtypes(include = ('number'))
vif_df2 = calculate_vif(int_X_train)
vif_df2[vif_df2['VIF']>10]['Feature']   




import pandas as pd

# 예시: df는 더미 변수와 수치 변수를 포함한 데이터프레임이라고 가정
# 더미 변수만 선택 (정수형 0, 1 값으로 구성된 변수들)
dummy_vars = X_train[['unknown1_type2' , 'unknown1_type3', 'unknown1_type4']]

# 상관계수 행렬 계산
corr_matrix = dummy_vars.corr()

# 상관관계 확인: VIF > 10인 특정 변수와 다른 변수 간의 상관관계
vif_high_dummy = 'your_vif_high_dummy_variable'  # VIF > 10인 더미 변수명
print(corr_matrix[vif_high_dummy].sort_values(ascending=False))




#---------------------------------
# IV 값을 계산하는 함수
def calculate_iv(df, feature, target):
    # 각 구간의 좋은 사건과 나쁜 사건의 수 계산
    grouped = df.groupby(feature, as_index=False)[target].agg(['count', 'sum'])
    grouped.columns = ['outlier','총건수', '긍정건수']
    grouped['부정건수'] = grouped['총건수'] - grouped['긍정건수']

    # 전체 긍정건수와 부정건수 계산
    total_positive = grouped['긍정건수'].sum()
    total_negative = grouped['부정건수'].sum()

    # 각 구간의 비율 계산
    grouped['긍정비율'] = grouped['긍정건수'] / total_positive
    grouped['부정비율'] = grouped['부정건수'] / total_negative

    # IV 값 계산
    grouped['IV'] = (grouped['긍정비율'] - grouped['부정비율']) * np.log((grouped['긍정비율'] / grouped['부정비율']).astype('float'))
    iv_value = grouped['IV'].sum()
    return iv_value



train = pd.concat([X_train,y_train], axis=1).select_dtypes(include=['object', 'bool'])
train.info()
train = train.astype('object')
train.info()
dummy_train = train.iloc[:,[0,1,2,-1]]
cat_train = train.iloc[:,3:]
cat_train.columns




# 각 수치형 컬럼의 IV 값 계산
iv_results = {}
find = ['target']
for column in cat_train.columns[:-1]:  # 타겟 컬럼 제외
    iv = calculate_iv(cat_train, column, 'target')
    iv_results[column] = iv

# IV 결과를 데이터프레임으로 변환 및 정렬
iv_df = pd.DataFrame(list(iv_results.items()), columns=['컬럼명', 'IV 값'])
iv_df = iv_df.sort_values(by='IV 값', ascending=False)

# 특정 IV 값을 기준으로 중요한 컬럼 선정 (예: IV > 0.1)
important_columns = iv_df[iv_df['IV 값'] > 0.1]


print(iv_df)
print("중요한 컬럼:")
print(important_columns)




# ------------------------

# Shapiro-Wilk 검정
stat, p_value = stats.shapiro(X_train)

print(f"Shapiro-Wilk 검정 통계량: {stat}")
print(f"p-value: {p_value}")

if p_value > 0.05:
    print("p-value가 0.05보다 크므로 귀무가설 채택: 데이터는 정규성을 따른다.")
else:
    print("p-value가 0.05보다 작으므로 귀무가설 기각: 데이터는 정규성을 따르지 않는다.")



# -------------------------


def qcut(X_train, X_test):  # 모든 수치 컬럼을 구간화 함. X_train 구간을 기준으로 X_test도 구간화함
    v = X_train.select_dtypes(include='number').columns
    
    # 각 열에 대해 동일한 구간(bin) 적용
    for i in v:
        # 훈련 데이터에서 구간 생성
        cut_bins = pd.qcut(X_train[i], q=4, duplicates='drop')
        # 구간의 실제 개수에 맞춘 라벨 생성
        num_bins = cut_bins.cat.categories.size
        labels = [f'Q{j+1}' for j in range(num_bins)]
        # qcut으로 훈련 데이터 구간화
        X_train[f'{i}_qcut'] = pd.qcut(X_train[i], q=4, labels=labels, duplicates='drop')
        # 훈련 데이터에서 얻은 구간을 사용하여 테스트 데이터 구간화
        bin_edges = pd.qcut(X_train[i], q=4, retbins=True, duplicates='drop')[1]  # 경계값(bin edges) 추출
        X_test[f'{i}_qcut'] = pd.cut(X_test[i], bins=bin_edges, labels=labels, include_lowest=True)
    return X_train, X_test


X_train, X_test = qcut(X_train, X_test)




from sklearn.preprocessing import LabelEncoder

# 각 qcut 컬럼에 대해 라벨 인코딩 적용
for i in v:
    le = LabelEncoder()
    X_train[f'{i}_qcut'] = le.fit_transform(X_train[f'{i}_qcut'])
    X_test[f'{i}_qcut'] = le.transform(X_test[f'{i}_qcut'])



    import numpy as np
import matplotlib.pyplot as plt
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')




cat_count(X_train, palette='coolwarm')





# 'unknown11' -10.5  , 17.5


# 그림 잘 안나옴
highlight1 = (X_train['unknown11'] < -10.5) | (X_train['unknown11'] > 17.5)
highlight2 = (X_train['unknown11'] > -10.5) & (X_train['unknown11'] < 17.5)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=X_train, x='unknown11', fill=True , palette='coolwarm', alpha=0.5)
y_values = sns.kdeplot(x_values).get_lines()[0].get_data()[1]  # KDE density values
plt.fill_between(X_train['unknown11'], y_values, where=highlight1, color='red', alpha=0.5, label='이상치 구간')
plt.fill_between(X_train['unknown11'], y_values, where=highlight2, color='blue', alpha=0.5, label='이상치 구간')
plt.title('unknown11의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌




# 그림 잘 나옴

plt.clf()
plt.figure(figsize=(6, 4))
# Plot the KDE
kde = sns.kdeplot(data=X_train, x='unknown11', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = (x_kde < -10.5) | (x_kde > 17.5)
    highlight2 = (x_kde > -10.5) & (x_kde < 17.5)

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 2)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='이상치 구간 [1]')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='정상 구간 [0]')
plt.text(20, 0.025, 0, ha='center', va='bottom', size=15)
plt.text(80, 0.00035, 1, ha='center', va='bottom', size=15)
plt.title('unknown11의 확률밀도', fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()




plt.clf()
plt.figure(figsize=(6, 4))
# Plot the KDE
kde = sns.kdeplot(data=X_train, x='unknown15', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = (x_kde < -10.5) | (x_kde > 17.5)
    highlight2 = (x_kde > -10.5) & (x_kde < 17.5)

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 2)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='이상치 구간 [1]')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='정상 구간 [0]')
plt.text(20, 0.025, 0, ha='center', va='bottom', size=15)
plt.text(80, 0.00035, 1, ha='center', va='bottom', size=15)
plt.title('unknown15의 확률밀도', fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()




plt.clf()
plt.figure(figsize=(8, 6))

# Plot the KDE
kde = sns.kdeplot(data=X_train, x='unknown11', fill=False, color='black', alpha=0.5)

# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = (x_kde < -10.5) | (x_kde > 17.5)
    highlight2 = (x_kde > -10.5) & (x_kde < 17.5)

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 2)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='이상치 구간 [1]')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='정상 구간 [0]')

# x축에 -10.5와 17.5 표시
xticks = [tick for tick in plt.xticks()[0] if tick != 0]  # 기존 tick에서 0을 제거
xticks += [-10.5, 17.5]  # -10.5와 17.5 추가
plt.xticks(sorted(xticks))  # tick 정렬
plt.text(20, 0.025, 0, ha='center', va='bottom', size=15)
plt.text(80, 0.00065, 1, ha='center', va='bottom', size=15)
plt.title('unknown11의 확률밀도', fontsize=20)
plt.tight_layout()
plt.legend()
plt.show()




type_17df = X_train[X_train['unknown1_original']=='type1']
plt.clf()
plt.figure(figsize=(4, 4))
# Plot the KDE
kde = sns.kdeplot(data=type_17df, x='unknown17', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = x_kde < 500
    highlight2 = (x_kde > 500) & (x_kde < 1000)
    highlight3 = x_kde > 1000

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 3)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='low')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='medium')
    plt.fill_between(x_kde, y_kde, where=highlight3, color=palette[2], label='high')
plt.title('type1_unknown17의 확률밀도', fontsize=15)
plt.tight_layout()
plt.legend()
plt.show()


type_17df = X_train[X_train['unknown1_origincaal']=='type2']
plt.clf()
plt.figure(figsize=(4, 4))
# Plot the KDE
kde = sns.kdeplot(data=type_17df, x='unknown17', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = x_kde < 300
    highlight2 = (x_kde > 300) & (x_kde < 750)
    highlight3 = x_kde > 750

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 3)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='low')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='medium')
    plt.fill_between(x_kde, y_kde, where=highlight3, color=palette[2], label='high')
plt.title('typ2_unknown17의 확률밀도', fontsize=15)
plt.tight_layout()
plt.legend()
plt.show()



type_17df = X_train[X_train['unknown1_original']=='type3']
plt.clf()
plt.figure(figsize=(4, 4))
# Plot the KDE
kde = sns.kdeplot(data=type_17df, x='unknown17', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = x_kde < 200
    highlight2 = (x_kde > 200) & (x_kde < 500)
    highlight3 = x_kde > 500

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 3)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='low')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='medium')
    plt.fill_between(x_kde, y_kde, where=highlight3, color=palette[2], label='high')

plt.title('typ3_unknown17의 확률밀도', fontsize=15)
plt.tight_layout()
plt.legend()
plt.show()



type_17df = X_train[X_train['unknown1_original']=='type4']
plt.clf()
plt.figure(figsize=(4, 4))
# Plot the KDE
kde = sns.kdeplot(data=type_17df, x='unknown17', fill=False, color='black', alpha=0.5)
# Ensure that the kde plot has lines
if len(kde.get_lines()) > 0:
    # Get the density and corresponding x-values from the KDE plot
    kde_data = kde.get_lines()[0].get_data()
    x_kde = kde_data[0]  # x-values of the KDE
    y_kde = kde_data[1]  # density values of the KDE

    # Define the mask for highlighting (values outside the range -10.5 to 17.5)
    highlight1 = x_kde < 600
    highlight2 = (x_kde > 600) & (x_kde < 1200)
    highlight3 = x_kde > 1200

    # Fill between the correct x and y KDE values
    palette = sns.color_palette("coolwarm", 3)
    plt.fill_between(x_kde, y_kde, where=highlight1, color=palette[0], label='low')
    plt.fill_between(x_kde, y_kde, where=highlight2, color=palette[1], label='medium')
    plt.fill_between(x_kde, y_kde, where=highlight3, color=palette[2], label='high')
plt.title('typ4_unknown17의 확률밀도', fontsize=15)
plt.tight_layout()
plt.legend()
plt.show()




Q1 = X_train['unknown16'].quantile(0.25)
Q3 = X_train['unknown16'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR) 
upper_bound = Q3 + (1.5 * IQR)



plt.figure(figsize=(4, 14))
sns.boxplot(X_train['unknown16'], color=sns.color_palette("coolwarm", 3)[0])
plt.text(0.25, 72.2, "Q1", color=sns.color_palette("coolwarm", 5)[4], ha='center', va='bottom', size=15)
plt.text(0.25, 137.1, "Q3", color=sns.color_palette("coolwarm", 5)[4], ha='center', va='bottom', size=15)
plt.text(0.25, 234.45, "상한 234.45", color=sns.color_palette("coolwarm", 5)[4], ha='center', va='bottom', size=15)
plt.text(0.25, -25.15, "상한 -25.15", color=sns.color_palette("coolwarm", 5)[4], ha='center', va='bottom', size=15)
plt.xlabel('unknown11의 박스 그래프', fontsize=15)


plt.figure(figsize=(4, 14))
sns.boxplot(X_train['unknown16'], color=sns.color_palette("coolwarm", 3)[0])
plt.text(0.25, 72.2, "Q1", color='red', ha='center', va='bottom', size=15)
plt.text(0.25, 137.1, "Q3", color='red', ha='center', va='bottom', size=15)
plt.text(0.25, 234.45, "상한 234.45", color='red', ha='center', va='bottom', size=15)
plt.text(0.25, -25.15, "상한 -25.15", color='red', ha='center', va='bottom', size=15)
plt.xlabel('unknown16의 박스 그래프', fontsize=15)



cor_df = X_train.drop(['unknown1_original','unknown1_type2','unknown1_type3','unknown1_type4'], axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(X_train['unknown17_type_n'])

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder(categories=[['low','medium','high']])
cor_df['unknown17_type_n'] = encoder.fit_transform(cor_df[['unknown17_type_n']])


correlation_matrix = cor_df.corr(method='spearman')  # 스피어만 상관계수 행렬 구하기

# 히트맵 시각화
plt.figure(figsize=(12, 10))  # 그림 크기 설정
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정

plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()



correlation_matrix = cor_df.corr()  # 스피어만 상관계수 행렬 구하기

# 히트맵 시각화
plt.figure(figsize=(12, 10))  # 그림 크기 설정
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 8})  # annot_kws로 글씨 크기 조정

plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()



cor_df.columns
cor_df[['unknown2','unknown3', 'unknown17','unknown8','unknown16']]
correlation_matrix = cor_df[['unknown2','unknown3', 'unknown17','unknown8','unknown16']].corr(method='spearman')  # 스피어만 상관계수 행렬 구하기

# 히트맵 시각화
plt.figure(figsize=(8, 6))  # 그림 크기 설정
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 12})  # annot_kws로 글씨 크기 조정

plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()





cor_df[['unknown2','unknown3', 'unknown17','unknown8','unknown16']]
correlation_matrix = cor_df[['unknown2','unknown3', 'unknown17','unknown8','unknown16']].corr()  # 스피어만 상관계수 행렬 구하기

# 히트맵 시각화
plt.figure(figsize=(8, 6))  # 그림 크기 설정
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
            cbar_kws={"shrink": .8}, annot_kws={"size": 12})  # annot_kws로 글씨 크기 조정

plt.title('Correlation Matrix', fontsize=14)  # 제목 설정 (작게)
plt.xticks(rotation=45, ha='right', fontsize=10)  # x축 라벨 회전 및 크기 조정
plt.yticks(rotation=0, fontsize=10)  # y축 라벨 수평 및 크기 조정
plt.tight_layout()  # 레이아웃 최적화
plt.show()








def rel_nx_ny(df, numeric_col, y):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	plt.title(f'{numeric_col}과 {y}컬럼의 관계')
	sns.scatterplot(data=df, x=numeric_col, y=y)
	plt.tight_layout()
	plt.show()
     

def num_scatter(df, palette='dark', alpha=0.5):
    import itertools
    numeric_cols = df.select_dtypes(include=['number']).columns 
    combinations=list(itertools.combinations(numeric_cols, 2))
    n = int(np.ceil(len(combinations)/4))
    plt.clf()
    plt.figure(figsize=(5*4, 4*n))
    for index, col in enumerate(combinations, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, index)
        sns.scatterplot(data=df, x=col[0], y=col[1] , palette=palette, alpha=alpha)
        plt.title(f'{col}의 확률밀도', fontsize=20)
    plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
     



num_scatter(df)


imp_df= df[['unknown2','unknown3', 'unknown17','unknown8','unknown16']]
kde(imp_df)




def rel_cat_nx_ny1(df, category_col, numeric_col, y):
	plt.clf()
	plt.rcParams['font.family'] = 'Malgun Gothic'
	plt.rcParams['axes.unicode_minus'] = False
	plt.title(f'{category_col}범주별로 {numeric_col}와 {y} 관계 비교')
	cats = df[category_col].value_counts().sort_values(ascending=False).index  # 갯수가 많은 범주 그래프부터 바탕에 낄려고
	for i in range(len(cats)):
		a = df[df[category_col] == cats[i]][[numeric_col, y]]
		n = np.random.choice(np.arange(len(colors)), len(colors), replace = False)
		n = n.tolist()[i]
		sns.scatterplot(data=a, x=numeric_col, y=y, color=colors[n], label=cats[i])
	plt.legend()
	plt.tight_layout()
	plt.show()
     




col1 = 'unknown2'
col2 = 'unknown3'


import itertools
imp_cols = ['unknown2','unknown3', 'unknown17','unknown8','unknown16']
combinations=list(itertools.combinations(imp_cols, 2))
n = int(np.ceil(len(combinations)/4))
plt.figure(figsize=(5*4, 4*n))
for index, col in enumerate(combinations, 1):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(n, 4, index)
    
    sns.scatterplot(data=df, x=col[0], y=col[1], hue='target', palette='coolwarm')
    plt.title(f'범주별 {col[0]}와 {col[1]}', fontsize=20)
    plt.legend()
plt.tight_layout()
plt.show()


def hue_num_scatter(df, hue ,palette='dark', alpha=0.5):
    import itertools
    num_cols = df.select_dtypes(include='number').columns
    combinations=list(itertools.combinations(num_cols, 2))
    n = int(np.ceil(len(combinations)/4))
    plt.figure(figsize=(5*4, 4*n))
    for index, col in enumerate(combinations, 1):
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, index)    
        sns.scatterplot(data=df, x=col[0], y=col[1], hue=hue, palette=palette, alpha=alpha)
        plt.title(f'범주별 {col[0]}와 {col[1]}', fontsize=20)
        plt.legend()
    plt.tight_layout()
    plt.show()
     
hue_num_scatter(df[['unknown2','unknown3', 'unknown17','unknown8','unknown16','unknown6','target']],'target')



kde(X_train[X_train['unknown1_original']=='type1'])
kde(X_train[X_train['unknown1_original']=='type2'])
kde(X_train[X_train['unknown1_original']=='type3'])
kde(X_train[X_train['unknown1_original']=='type4'])