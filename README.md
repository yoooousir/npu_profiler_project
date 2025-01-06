# NPU Profiler Project

MNIST 데이터셋과 CNN 모델을 사용해 NPU에서의 성능을 측정하고 분석하는 프로젝트로, 
Fully Connected Layer의 입력 크기를 동적으로 처리하며, 각 레이어의 추론 시간을 측정하고 결과를 시각화했습니다.

----------------------------
- 동적 입력 크기 처리: 입력 데이터와 모델의 크기 불일치를 자동으로 처리하기 위해 Fully Connected Layer (fc1, fc2)가 입력 데이터의 크기에 따라 자동으로 조정됩니다.
- 레이어 프로파일링: 각 레이어의 추론 시간을 측정하고 결과를 시각화합니다.
- MNIST 데이터셋 로드 및 전처리: PyTorch를 사용해 MNIST 데이터를 로드하고 전처리합니다.
----------------------------
### 설치 및 실행
```bash
git clone  https://github.com/yoooousir/npu_profiler_project.git
cd npu_profiler_project
pip install -r requirements.txt
python main.py
```
-----------------------------
### 프로젝트 구조
```
├── main.py                         # 프로젝트 실행
├── model.py                        # SimpleCNN 클래스 정의
├── profiler.py                     # 모델 프로파일링
├── README.md                       # 프로젝트 설명
├── requirements.txt                # 필요한 라이브러리
├── layer_wise_execution_time.png   # 레이어별 실행 시간 시각화 결과
└── .gitignore                      # 불필요한 파일 및 폴더 제외 설정
```
