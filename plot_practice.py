import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats


# 데이터 불러오기
file_dir = input("데이터 파일의 절대 경로를 입력하세요 :")
df = pd.read_csv(file_dir, encoding='cp949')
# C:/Users/USER/Documents/LS 빅데이터 스쿨/project3/data_week3.csv




# 범주별로 수치형과 수치형 산점도
# 범주별로 범주 막대그래프
# 날짜 가로축으로 수치 시계열
# 날짜 가로축으로 범주별로 수치 시계열
# 범주별 수치 kde
# 정규분포 겹치기





info = dict( data_frame = df , col = 'unknown17', palette = 'dark', alpha=1.0)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.histplot(data = info['data_frame'] , x=info['col'], stat='density', palette=info['palette'], alpha = info['alpha'])
plt.title(f'{info['col']}의 히스토그램 분포', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌


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


def hist_bycat(df, cat_col, palette = 'dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.histplot(data=df, x=col, stat='density', hue=cat_col, palette = palette, alpha=alpha)
		plt.title(f'{col}의 히스토그램 분포', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
	
hist_bycat(df, 'unknown1')

df2 = df.copy()
df2['unknown4'] = df2['unknown4'].astype('object')
hist_bycat(df2, 'unknown4')



info = dict(data_frame=df, col='unknown17', palette='dark', alpha=1.0)
plt.clf()
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.kdeplot(data=info['data_frame'], x=info['col'], fill=True , palette=info['palette'], alpha=info['alpha'])
plt.title(f'{info['col']}의 확률밀도', fontsize=20)
plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌



def kde(df, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
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
	

def kde(df, cat_col, palette='dark', alpha=0.5):
	numeric_cols = df.select_dtypes(include=['number']).columns   # number에는 boolean도 포함됨
	n = int(np.ceil(len(numeric_cols)/4))
	plt.clf()
	plt.figure(figsize=(5*4, 4*n))
	for index, col in enumerate(numeric_cols, 1):
		plt.rcParams['font.family'] = 'Malgun Gothic'
		plt.rcParams['axes.unicode_minus'] = False
		plt.subplot(n, 4, index)
		sns.kdeplot(data=df, x=col, fill=True , hue=cat_col, palette=palette, alpha=alpha)
		plt.title(f'{col}의 확률밀도', fontsize=20)
	plt.tight_layout()  #  plt.show() 전에 있어야 적용됨.
	plt.show()  # for문 안에 있으면 그래프 1개씩 보여줌
	

kde(df, 'target')