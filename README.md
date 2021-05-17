# Pytorch
데이터 마이닝 수업 시간에 파이토치에 대한 과제를 하게 되었습니다. 수업 시간에는 각종 알고리즘들의 수식, 의사코드 같은 이론적인 부분들에 대해서만 배웠는데 이렇게 과제로 파이토치를 갑자기 내버리시니... 따로 공부를 안 할 수가 없네요😂😂 <br>이해가 미흡하지만 여기 저기서 공부한 내용들을 직접 써보면서 익히는 저만의 공부장입니다.
## 1. 파이토치 패키지의 기본 구성
|패키지|구성|
|:---|:---|
|1. torch|main namespace입니다. numpy같은 구조이고 수학 함수 엄청 많아요|
|2. torch.autograd|자동 미분 함수들 있어요.
|3. torch.nn|신경망 구축 위한 데이터 구조, 레이어 있어요. NN, LSTM, ReLU, MSELoss...|
|4. torch.optim|확률적 경사 하강법 SGD(Stochastic Gradient Descent) 기반한 parameter 최적화 알고리즘 있어요.
|5. torch.utils.data|SGD의 반복 연산 시 쓰는 미니 batch용 유틸리티 함수가 있어요.
|6. torch.onnx|ONNX 포맷으로 서로 다른 딥 러닝 프레임워크 간 모델 공유할 때 사용해요.
## 텐서 조작하기
### 벡터, 행렬 그리고 텐서 <br>
👉 감 오시죠? 텐서는 3차원의 데이터입니다. (Data Science 분야에서는 그냥 전부 다 텐서라고 부르기도 한대요)
- 2D Tensor : (Batch size, Dim)
- 3D Tensor : (Batch size, width, height) - 요건 Typical Computer Vision 분야에서의 3차원 텐서
- 3D Tensor : (Batch size, length, dim) - 요건 Typical NLP 분야에서의 3차원 텐서 
<br>👉 컴퓨터는 train data를 보통 덩어리로 처리해요. 만약 3000개 train data 에서 64개씩 꺼내서 처리한다면 batch size = 64 가 되겠죠?
### 2. 파이토치 Tensor Allocation
```python
import torch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
lt = torch.LongTensor([1, 2, 3, 4])
```
- 위의 코드가 1차운 벡터 만든거예요. np.array랑 비슷하죠?
```python
t.dim() #차원
t.shape() #shape
t.size() #size
```
- 휴~ numpy랑 정말 비슷합니다.
```python
print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱
```
- 슬라이싱도 numpy랑 같은 방식으로 동작하네요.
```python
# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)
# 연산 과정
[1, 2]
==> [[1, 2],
     [1, 2]]
[3]
[4]
==> [[3, 3],
     [4, 4]]
```
- 1x2 tensor + 2x1 tensor = 2x2 tensor 가 되네요. 이를 Brodcasting이라고 합니다.
- 자동으로 되니깐 사용자들은 주의해서 코드를 작성하셔야 겠습니다.
### 3. 자주 쓰는 기능
|함수|기능|
|:---|:---|
|matmul|찐 행렬 곱셈|
|mul|같은 위치 원소 끼리 곱셈|
|mean|평균|
|sum|덧셈|
|max|최대값 리턴|
|argmax|최대값 인덱스 리턴|
|ones_like|1로 채워서 텐서 할당|
|zeros_like |0으로 채워서 텐서 할당|

### 4. View, Squeeze, Unsqueeze, Concatenate, Stacking
1. View
- 원소의 수를 유지하면서 Tensor의 크기를 변경하는 reshape과 같은 역할입니다.
```python
ft.view([-1,3]) #ft라는 텐서를 (?,3) 크기로 변경하겠다.
```
2. Squeeze
- 1인 차원 제거
```python
ft.squeeze()
```
3. Unsqueeze
- 반대로 특정 위치에 1인 차원을 추가
```python
ft.unsqueeze(0)
```
4. Concatenate
- 2개 텐서 붙이기
```python
print(torch.cat([x, y], dim=0)) # 첫번째 차원을 늘려!
# 결과
tensor([[1., 2.],
        [3., 4.],
        [5., 6.],
        [7., 8.]])
        
print(torch.cat([x, y], dim=1)) # 두번째 차원을 늘려!
#결과
tensor([[1., 2., 5., 6.],
        [3., 4., 7., 8.]])
```
5. Stacking
- Concatenate 비슷하긴한데 쌓는다는 느낌
```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
print(torch.stack([x, y, z]))
#결과
tensor([[1., 4.],
        [2., 5.],
        [3., 6.]])
```
- 많은 연산을 축약하고 있다고 생각하시면 됩니다.  unsqueeze와 concatenate를 함축하고 있어요 <br>
```python print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)) ```
### 5. 파이썬 클래스
- C++이랑 거의 비슷하네요. 공부할 게 딱히 없습니다.
```python
class Calculator:
    def __init__(self): # 객체 생성 시 호출될 때 실행되는 초기화 함수. 이를 생성자라고 한다.
        self.result = 0

    def add(self, num): # 객체 생성 후 사용할 수 있는 함수.
        self.result += num
        return self.result
cal1 = Calculator()
print(cal1.add(3))
print(cal1.add(4))
```
