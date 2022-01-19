### **Kaggle Competitions - Bike Sharing Demand**
#### 링크: https://www.kaggle.com/c/bike-sharing-demand/overview
<br/>

#### 사용 모델: RandomForestRegressor
#### 평가 지표: Root Mean Squared Logarithmic Error
#### 점수: 0.42480 (7년 전 리더보드 기준 상위 16% 정도)


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
# 제출 양식 불러와서 확인하기
sub = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
sub
```


```python
# train 데이터 불러와서 확인하기
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
train
```


```python
train.isna().sum()  # 결측치 없음
```


```python
# test 데이터 불러와서 확인하기
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
test
```


```python
test.isna().sum()  # 결측치 없음
```


```python
# train, test 같이 전처리하기 위해 합치기
alldata = pd.concat([train, test])
alldata
```


```python
# datetime에서 쓸만한 정보들 추출하기
## train은 매달 1일~19일, test는 매달 20일~31일 -> 따라서 '일'은 도움이 되지 않을 것이라 판단
## '분, 초'는 다 0이므로 도움이 되지 않을 것으로 판단
## 그 외 '년, 월, 시'는 도움이 될 수 있을 것이라고 예상

alldata['datetime'] = pd.to_datetime(alldata['datetime'])  # 우선 datetime을 날짜 형식으로 변환
alldata['year'] = alldata['datetime'].dt.year  # 년 추출
alldata['month'] = alldata['datetime'].dt.month  # 월 추출
alldata['hour'] = alldata['datetime'].dt.hour # 시 추출

# week 정보 추가
alldata['week'] = alldata['datetime'].dt.week
# 요일 정보 추가
alldata['weekday'] = alldata['datetime'].dt.weekday
alldata
```


```python
# 그래프로 확인 (x축: 정답에 대한 영향력을 확인하고 싶은 변수, y축: 정답)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(alldata['year'], alldata['count'])  # x축은 카테고리 느낌, y축은 수치일 때 boxplot 활용
```


```python
plt.figure(figsize=(8, 6))
sns.boxplot(alldata['month'], alldata['count'])
```


```python
plt.figure(figsize=(8, 6))
sns.boxplot(alldata['hour'], alldata['count'])
```


```python
# datetime, casual, registered, count 열 삭제
alldata2 = alldata.drop(columns=['datetime', 'casual', 'registered', 'count'])
alldata2
```


```python
# 합친 데이터 다시 분리
train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]
print(train2.shape, test2.shape)
```


```python
# 모델링
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=42)
rfr.fit(train2, np.log(train['count']))
```


```python
result = rfr.predict(test2)
```


```python
# 참고 - 로그 취했을 때 정규분포에 가까워짐
import seaborn as sns
sns.displot(train['count'])
```


```python
sns.displot(np.log(train['count']))
```


```python
# 결과 넣기
sub['count'] = np.exp(result)
sub
```


```python
sub.to_csv('submission.csv', index=False)
```
