# 190509-dl-seminar

parameter의 의미
-----------
> 차원이 많으면 표현력이 뛰어나다고 함.

parameter
----------
> hyper-parameter
* Channel 정보, Stride, Padding 정보 등... (ex 5x5x3)
> parameter
* Weight, Bias 등

batch Gradient Descent
----------
> Full batch : 문제 전체를 다 보고 반영함. <br>
> Stochastic batch : 한 문제만 보고 이를 반영함 --> 값이 이리저리 튀긴 하지만 연산량이 적기 때문에 사용. <br>
> mini batch : 위 두개의 장단점을 조합한 것이 미니배치.

Jupyter도 커맨드 모드가 됐구나?
----------
 <code> 코드 앞에 !(느낌표) 넣으면 된다. </code>

김태영 대표님 Keras 강의
========================================

다층 퍼셉트론 모델 만들어보기
------

> 아래와 같은 데이터가 있다고 하자. <br>

```
 6,148,72,35,0,33.6,0.627,50,1
 1,85,66,29,0,26.6,0.351,31,0
 8,183,64,0,0,23.3,0.672,32,1
```


> 앞의 8개 데이터의 경우에는 인풋, 최 후방의 데이터의 경우에는 Dead or Alive 라고 한다. <br>
> 실습은 google colab 환경에서 진행 하였으며, 코드는 아래와 같다. <br>

```
from google.colab import files
uploaded = files.upload()
```
> 해당 코드의 경우 실행 할 경우 가상의 드라이브(구글 드라이브 X)에 파일을 업로드 할 수 있다.

```
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 랜덤시드 고정시키기
np.random.seed(5)

# 1. 데이터 준비하기
dataset = np.loadtxt("./pima-indians-diabetes.data", delimiter=",")
 # 위에서 파일을 업로드 하였을 경우, 기본 경로 (./)에 파일이 올라간다.

# 2. 데이터셋 생성하기
x_train = dataset[:700,0:8] # 0~699번줄 까지 데이터 -> 배열 0번 부터 7번 까지 데이터를 불러온다.
y_train = dataset[:700,8] # 0~699번줄 까지 데이터 -> 배열 8번의 데이터를 불러온다.
x_test = dataset[700:,0:8] # 700번줄 부터의 데이터 -> ...
y_test = dataset[700:,8] # 700번 줄 부터의 데이터 -> ...

# 3. 모델 구성하기
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 4. 모델 학습과정 설정하기
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=1500, batch_size=64)

# 6. 모델 평가하기
scores = model.evaluate(x_test, y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
```
