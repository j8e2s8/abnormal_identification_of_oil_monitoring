import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, confusion_matrix, auc, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder  
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from scipy.stats import boxcox
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
from sklearn.utils import resample
from scipy.stats import chi2_contingency
import seaborn as sns

# 랜덤 시드 설정
np.random.seed(42)

# 데이터 불러오기
df = pd.read_csv("./data/data_week3.csv")

# 공통 전처리 ======================================================================
# 특성과 타겟 정의
X = df.drop("target", axis=1)
y = df['target']

# 더미코딩
X = pd.get_dummies(X, columns=['unknown1'], drop_first=True)

# train/test 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("훈련 데이터 크기:", X_train.shape, y_train.shape)
print("테스트 데이터 크기:", X_test.shape, y_test.shape)

# EDA =============================================================================
# KDE 그래프
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

# 데이터에 있는 모든 범주형 변수 subplot 빈도 그래프   
def cat_count(df, palette='dark'):
    col = df.select_dtypes(include=['object','boolean']).columns
    w_n = sum([len(df[i].unique()) for i in col])
    n = int(np.ceil(w_n/4))
    plt.clf()
    plt.figure(figsize=(6*4, 5*n))
    for i, col in enumerate(col, 1): 
        # plt.figure(figsize=(6*4, 5*n))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(n, 4, i)
        if df[col].dtypes == 'boolean':
            ax = sns.countplot(df[col].astype('str'), order=df[col].value_counts().sort_values().index)
        else:
            ax = sns.countplot(df[col], order=df[col].value_counts().sort_values().index)
        for p in ax.patches:
            plt.text(p.get_width(), p.get_y() + p.get_height()/2 , p.get_width())
        plt.title(f'{col}의 범주별 빈도 그래프', fontsize=20)
    plt.tight_layout( )  #  plt.show() 전에 있어야 적용됨.
    plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


cat_count(df)


plt.figure(figsize=(15, 10))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title('Boxplot for Numerical Columns in data')
# plt.ylim(0, 100000)
plt.show()

# 데이터에 있는 모든 수치형 변수 subplot 빈도 그래프
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

# 이상치 여부 함수 만들기 ===========================================================
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)

    return pd.concat([lower_bound, upper_bound], axis = 1).T


# # 분포 구간화 함수 만들기 =========================================================
# def categorize_unknown17(row):
#     if row['unknown1_type2'] == False & row['unknown1_type3'] == False & row['unknown1_type4'] == False :
#         if row['unknown17'] < 500:
#             return 'low'
#         elif row['unknown17'] < 1000:
#             return 'medium'
#         else:
#             return 'high'
#     elif row['unknown1_type2'] == True:
#         if row['unknown17'] < 300:
#             return 'low'
#         elif row['unknown17'] < 750:
#             return 'medium'
#         else:
#             return 'high'
#     elif row['unknown1_type3'] == True:
#         if row['unknown17'] < 200:
#             return 'low'
#         elif row['unknown17'] < 500:
#             return 'medium'
#         else:
#             return 'high'
#     elif row['unknown1_type4'] == True:
#         if row['unknown17'] < 600:
#             return 'low'
#         elif row['unknown17'] < 1200:
#             return 'medium'
#         else:
#             return 'high'

# X_train['unknown17_type_n'] = X_train.apply(categorize_unknown17, axis=1)  # 행에 적용
# X_train.columns
# X_test['unknown17_type_n'] = X_test.apply(categorize_unknown17, axis=1)  # 행에 적용

# 카이제곱검정 및 타겟변수와 독립적인 변수 제거 ========================================
num_X_train = X_train.select_dtypes(include=('int', 'float'))
variable = num_X_train.columns
select_col = X_train.columns.drop(variable)
for col in select_col:
   contingency_table = pd.crosstab(X_train[col], y_train)
   chi2, p, dof, expected = chi2_contingency(contingency_table)

   print(f"{col}의 예상 빈도:")
   print(expected)

   if (expected < 5).any():
      print(f"{col}의 예상 빈도 중 5 미만인 값이 있습니다.")
   else:
      print(f"{col}의 모든 예상 빈도가 5 이상입니다.")
      
   print(f"{col}의 p-값 :", p)

   if p < 0.05:
      print(f"{col}의 귀무가설 기각: 독립변수와 타겟 변수는 독립적이지 않다. 관련이 있다.", "\n","-"*30)
   else:
      print(f"{col}의 귀무가설 채택: 독립변수와 타겟 변수는 독립적이다.", "\n","-"*30)


num_X_train = X_train.select_dtypes(include=('int', 'float'))
variable = num_X_train.columns

for col in variable:
   X_train[f'{col}_outlier'] = np.where((X_train[col]<IQR_outlier(num_X_train).loc[0,col])|(X_train[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)
   X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train).loc[0,col])|(X_test[col]>IQR_outlier(num_X_train).loc[1,col]),True,False)

# 검정에서 타겟변수와 독립인 변수 제거
X_train = X_train.drop(['unknown11_outlier', 'unknown16_outlier', 'unknown17_outlier'], axis=1)
X_test = X_test.drop(['unknown11_outlier', 'unknown16_outlier', 'unknown17_outlier'], axis=1)

# 모델 생성 및 예측 =================================================================
# # catboost
# scale_pos_weight = (len(y) - y.sum()) / y.sum()

# def objective(trial):
#     param = {
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10),
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'scale_pos_weight': scale_pos_weight,
#         'random_seed': 42,
#         'verbose': 0
#     }

#     catboost_clf = CatBoostClassifier(**param)
#     catboost_clf.fit(X_train, y_train)
    
#     prob_y = catboost_clf.predict_proba(X_test)[:, 1]
#     roc_auc = roc_auc_score(y_test, prob_y)
    
#     return roc_auc

# # Optuna 최적화 수행
# study = optuna.create_study(direction='maximize', sampler=TPESampler())
# study.optimize(objective, n_trials=50)

# # 최적 하이퍼파라미터 출력
# print("Best trial:")
# print("  Value: ", study.best_trial.value)
# print("  Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")

# # 최적 하이퍼파라미터로 최종 모델 학습
# best_params = study.best_trial.params
# catboost_clf_best = CatBoostClassifier(**best_params, scale_pos_weight=scale_pos_weight, random_seed=42)
# catboost_clf_best.fit(X_train, y_train)

# # 최종 모델 예측 및 평가
# prob_y = catboost_clf_best.predict_proba(X_test)[:, 1]

scale_pos_weight = (len(y) - y.sum()) / y.sum()
catboost_clf = CatBoostClassifier(
    learning_rate = 0.008454014889612177,
    depth = 7,
    l2_leaf_reg = 9.76169089164976,
    iterations = 549,
    scale_pos_weight=scale_pos_weight,random_seed=42)
catboost_clf.fit(X_train, y_train)

# 최종 모델 예측 및 평가
prob_y = catboost_clf.predict_proba(X_test)[:, 1].round(6)
prob_y.shape

## 구간별 성능평가표 =================================================================
# 상위 10개 구간 설정
df_prob = pd.DataFrame({'prob_y': prob_y, 'y_test': y_test})
df_prob = df_prob.sort_values(by='prob_y', ascending=False).reset_index(drop=True)

# 구간별 개수 설정
num_bins = 10
bin_size = 282

# 마지막 구간의 크기 조정
last_bin_size = len(df_prob) - (bin_size * (num_bins - 1))  # 마지막 구간의 크기 계산
if last_bin_size < 0:
    raise ValueError("데이터가 구간 수에 비해 부족합니다.")

# 구간별 Precision, Recall, G-Mean, ROC-AUC 계산
table = []
cumulative_tp, cumulative_fp = 0, 0
total_y1 = y_test.sum()

for rank in range(num_bins):
    # 상위 누적합계
    if rank == num_bins - 1:
        cumulative_count = len(df_prob)
    else:
        cumulative_count = (rank + 1) * bin_size

    # 상위 누적합계에 해당하는 데이터에서 Y=1인 개수 계산
    tp = df_prob['y_test'].iloc[:cumulative_count].sum()
    fp = cumulative_count - tp
    tn = y_test.shape[0] - total_y1 - fp

    # Precision, Recall 계산
    precision = tp / cumulative_count if cumulative_count > 0 else 0
    recall = tp / total_y1 if total_y1 > 0 else 0
    
    # Specificity 계산
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # G-Mean 계산 (Recall × Specificity의 제곱근)
    g_mean = np.sqrt(recall * specificity) if (recall + specificity) > 0 else 0
    
    # Cut-off에 해당하는 확률값
    cutoff_prob = df_prob['prob_y'].iloc[cumulative_count - 1] if cumulative_count - 1 < len(df_prob) else 0
    
    # ROC AUC 계산
    roc_auc = roc_auc_score(df_prob['y_test'].iloc[:cumulative_count], df_prob['prob_y'].iloc[:cumulative_count])

    # 결과 추가
    table.append([cumulative_count, tp, fp, cutoff_prob, precision, recall, g_mean, roc_auc])

# 데이터프레임 생성 및 열 이름 지정
result_df = pd.DataFrame(table, columns=[
    '예측 Y=1', 'TP', 'FP', 'Cut-off', 'Precision', 'Recall', 'G-Mean', 'ROC-AUC'
])

result_df

## 임계값 설정 및 G-mean 신뢰구간 ===================================================
# 임계값 선정
threshold = 0.470613
pred_y = (prob_y >= threshold).astype(int)

# 성능 신뢰구간
# 신뢰구간 계산을 위한 부트스트랩
n_iterations = 1000
g_mean_scores = []

for _ in range(n_iterations):
    # 부트스트랩 샘플링
    indices = np.random.choice(len(y_test), len(y_test), replace=True)
    sample_y_test = y_test.iloc[indices]
    sample_pred_y = pred_y[indices]
    
    # G-Mean 계산 및 저장
    g_mean_scores.append(geometric_mean_score(sample_y_test, sample_pred_y))

# 신뢰구간 계산 함수
def calculate_confidence_interval(metric_values, confidence=0.95):
    mean = np.mean(metric_values)
    sem = stats.sem(metric_values)
    margin = sem * stats.t.ppf((1 + confidence) / 2., len(metric_values) - 1)
    return mean, mean - margin, mean + margin

# G-Mean의 신뢰구간 계산
gmean_mean, gmean_lower, gmean_upper = calculate_confidence_interval(g_mean_scores)

# 결과 출력
print(f"G-Mean: {gmean_mean:.6f}, 95% Confidence Interval: [{gmean_lower:.6f}, {gmean_upper:.6f}]")

# 변수 중요도 ======================================================================
# 피처 중요도 가져오기
feature_importances = catboost_clf.get_feature_importance()

# 피처 이름
feature_names = X_train.columns

# 변수 중요도 데이터프레임 생성
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 중요도 기준으로 내림차순 정렬
importance_df = importance_df.sort_values(ascending=False, by='Importance')

# 피처 중요도 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

# 정렬된 변수 중요도 데이터프레임 출력
importance_df
