{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Kaggle Competitions - Bike Sharing Demand**\n",
    "#### 링크: https://www.kaggle.com/c/bike-sharing-demand/overview\n",
    "<br/>\n",
    "\n",
    "#### 사용 모델: RandomForestRegressor\n",
    "#### 평가 지표: Root Mean Squared Logarithmic Error\n",
    "#### 점수: 0.42480 (7년전 리더보드 기준 상위 16% 정도)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.116537Z",
     "iopub.status.busy": "2022-01-17T08:25:25.115968Z",
     "iopub.status.idle": "2022-01-17T08:25:25.126867Z",
     "shell.execute_reply": "2022-01-17T08:25:25.126018Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.116499Z"
    }
   },
   "outputs": [],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.129009Z",
     "iopub.status.busy": "2022-01-17T08:25:25.128646Z",
     "iopub.status.idle": "2022-01-17T08:25:25.149574Z",
     "shell.execute_reply": "2022-01-17T08:25:25.14878Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.128972Z"
    }
   },
   "outputs": [],
   "source": [
    "# 제출 양식 불러와서 확인하기\n",
    "sub = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')\n",
    "sub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.151415Z",
     "iopub.status.busy": "2022-01-17T08:25:25.151167Z",
     "iopub.status.idle": "2022-01-17T08:25:25.187158Z",
     "shell.execute_reply": "2022-01-17T08:25:25.186397Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.151383Z"
    }
   },
   "outputs": [],
   "source": [
    "# train 데이터 불러와서 확인하기\n",
    "train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')\n",
    "train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.188685Z",
     "iopub.status.busy": "2022-01-17T08:25:25.188451Z",
     "iopub.status.idle": "2022-01-17T08:25:25.19939Z",
     "shell.execute_reply": "2022-01-17T08:25:25.198791Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.188653Z"
    }
   },
   "outputs": [],
   "source": [
    "train.isna().sum()  # 결측치 없음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.201284Z",
     "iopub.status.busy": "2022-01-17T08:25:25.200986Z",
     "iopub.status.idle": "2022-01-17T08:25:25.228998Z",
     "shell.execute_reply": "2022-01-17T08:25:25.228232Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.201249Z"
    }
   },
   "outputs": [],
   "source": [
    "# test 데이터 불러와서 확인하기\n",
    "test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')\n",
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.230494Z",
     "iopub.status.busy": "2022-01-17T08:25:25.230187Z",
     "iopub.status.idle": "2022-01-17T08:25:25.240467Z",
     "shell.execute_reply": "2022-01-17T08:25:25.239833Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.230461Z"
    }
   },
   "outputs": [],
   "source": [
    "test.isna().sum()  # 결측치 없음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.24149Z",
     "iopub.status.busy": "2022-01-17T08:25:25.241311Z",
     "iopub.status.idle": "2022-01-17T08:25:25.266317Z",
     "shell.execute_reply": "2022-01-17T08:25:25.265557Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.241468Z"
    }
   },
   "outputs": [],
   "source": [
    "# train, test 같이 전처리하기 위해 합치기\n",
    "alldata = pd.concat([train, test])\n",
    "alldata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.267881Z",
     "iopub.status.busy": "2022-01-17T08:25:25.267643Z",
     "iopub.status.idle": "2022-01-17T08:25:25.319404Z",
     "shell.execute_reply": "2022-01-17T08:25:25.318624Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.267847Z"
    }
   },
   "outputs": [],
   "source": [
    "# datetime에서 쓸만한 정보들 추출하기\n",
    "## train은 매달 1일~19일, test는 매달 20일~31일 -> 따라서 '일'은 도움이 되지 않을 것이라 판단\n",
    "## '분, 초'는 다 0이므로 도움이 되지 않을 것으로 판단\n",
    "## 그 외 '년, 월, 시'는 도움이 될 수 있을 것이라고 예상\n",
    "\n",
    "alldata['datetime'] = pd.to_datetime(alldata['datetime'])  # 우선 datetime을 날짜 형식으로 변환\n",
    "alldata['year'] = alldata['datetime'].dt.year  # 년 추출\n",
    "alldata['month'] = alldata['datetime'].dt.month  # 월 추출\n",
    "alldata['hour'] = alldata['datetime'].dt.hour # 시 추출\n",
    "\n",
    "# week 정보 추가\n",
    "alldata['week'] = alldata['datetime'].dt.week\n",
    "# 요일 정보 추가\n",
    "alldata['weekday'] = alldata['datetime'].dt.weekday\n",
    "alldata"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.3218Z",
     "iopub.status.busy": "2022-01-17T08:25:25.321567Z",
     "iopub.status.idle": "2022-01-17T08:25:25.507753Z",
     "shell.execute_reply": "2022-01-17T08:25:25.506997Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.321768Z"
    }
   },
   "outputs": [],
   "source": [
    "# 그래프로 확인 (x축: 정답에 대한 영향력을 확인하고 싶은 변수, y축: 정답)\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.boxplot(alldata['year'], alldata['count'])  # x축은 카테고리 느낌, y축은 수치일 때 boxplot 활용"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.509095Z",
     "iopub.status.busy": "2022-01-17T08:25:25.508872Z",
     "iopub.status.idle": "2022-01-17T08:25:25.891758Z",
     "shell.execute_reply": "2022-01-17T08:25:25.891041Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.509064Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 6))\n",
    "sns.boxplot(alldata['month'], alldata['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:25.893914Z",
     "iopub.status.busy": "2022-01-17T08:25:25.89359Z",
     "iopub.status.idle": "2022-01-17T08:25:26.415697Z",
     "shell.execute_reply": "2022-01-17T08:25:26.415036Z",
     "shell.execute_reply.started": "2022-01-17T08:25:25.893872Z"
    }
   },
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 6))\n",
    "sns.boxplot(alldata['hour'], alldata['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:26.417256Z",
     "iopub.status.busy": "2022-01-17T08:25:26.417005Z",
     "iopub.status.idle": "2022-01-17T08:25:26.440735Z",
     "shell.execute_reply": "2022-01-17T08:25:26.440039Z",
     "shell.execute_reply.started": "2022-01-17T08:25:26.417222Z"
    }
   },
   "outputs": [],
   "source": [
    "# datetime, casual, registered, count 열 삭제\n",
    "alldata2 = alldata.drop(columns=['datetime', 'casual', 'registered', 'count'])\n",
    "alldata2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:26.442146Z",
     "iopub.status.busy": "2022-01-17T08:25:26.441906Z",
     "iopub.status.idle": "2022-01-17T08:25:26.447238Z",
     "shell.execute_reply": "2022-01-17T08:25:26.446542Z",
     "shell.execute_reply.started": "2022-01-17T08:25:26.442114Z"
    }
   },
   "outputs": [],
   "source": [
    "# 합친 데이터 다시 분리\n",
    "train2 = alldata2[:len(train)]\n",
    "test2 = alldata2[len(train):]\n",
    "print(train2.shape, test2.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:26.448862Z",
     "iopub.status.busy": "2022-01-17T08:25:26.448457Z",
     "iopub.status.idle": "2022-01-17T08:25:30.693764Z",
     "shell.execute_reply": "2022-01-17T08:25:30.693023Z",
     "shell.execute_reply.started": "2022-01-17T08:25:26.448825Z"
    }
   },
   "outputs": [],
   "source": [
    "# 모델링\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "rfr = RandomForestRegressor(random_state=42)\n",
    "rfr.fit(train2, np.log(train['count']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:30.696991Z",
     "iopub.status.busy": "2022-01-17T08:25:30.69678Z",
     "iopub.status.idle": "2022-01-17T08:25:30.809566Z",
     "shell.execute_reply": "2022-01-17T08:25:30.808957Z",
     "shell.execute_reply.started": "2022-01-17T08:25:30.696967Z"
    }
   },
   "outputs": [],
   "source": [
    "result = rfr.predict(test2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:30.811505Z",
     "iopub.status.busy": "2022-01-17T08:25:30.810617Z",
     "iopub.status.idle": "2022-01-17T08:25:31.178202Z",
     "shell.execute_reply": "2022-01-17T08:25:31.177571Z",
     "shell.execute_reply.started": "2022-01-17T08:25:30.811468Z"
    }
   },
   "outputs": [],
   "source": [
    "# 참고 - 로그 취했을 때 정규분포에 가까워짐\n",
    "import seaborn as sns\n",
    "sns.displot(train['count'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:31.181363Z",
     "iopub.status.busy": "2022-01-17T08:25:31.180642Z",
     "iopub.status.idle": "2022-01-17T08:25:31.551013Z",
     "shell.execute_reply": "2022-01-17T08:25:31.550363Z",
     "shell.execute_reply.started": "2022-01-17T08:25:31.181322Z"
    }
   },
   "outputs": [],
   "source": [
    "sns.displot(np.log(train['count']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:31.553321Z",
     "iopub.status.busy": "2022-01-17T08:25:31.552121Z",
     "iopub.status.idle": "2022-01-17T08:25:31.56566Z",
     "shell.execute_reply": "2022-01-17T08:25:31.564989Z",
     "shell.execute_reply.started": "2022-01-17T08:25:31.553281Z"
    }
   },
   "outputs": [],
   "source": [
    "# 결과 넣기\n",
    "sub['count'] = np.exp(result)\n",
    "sub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-01-17T08:25:31.567244Z",
     "iopub.status.busy": "2022-01-17T08:25:31.566972Z",
     "iopub.status.idle": "2022-01-17T08:25:31.596773Z",
     "shell.execute_reply": "2022-01-17T08:25:31.596165Z",
     "shell.execute_reply.started": "2022-01-17T08:25:31.567207Z"
    }
   },
   "outputs": [],
   "source": [
    "sub.to_csv('submission.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
