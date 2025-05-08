# Nova Korean Performance Benchmark

이 저장소는 Nova 모델의 한국어 성능을 벤치마킹하기 위한 코드와 결과를 포함합니다.

## 주요 파일 및 디렉터리

*   `main.py`: 벤치마크 실행을 위한 메인 스크립트입니다.
*   `bedrock.py`: Bedrock 모델 관련 코드를 포함합니다.
*   `evaluation.ipynb`: 평가 관련 노트북입니다.
*   `results/`: 벤치마크 결과가 저장되는 디렉터리입니다.

## 사용 방법

* 요구 사항

    ```bash
    pip install -r requirements.txt
    ```

* 벤치마크 실행

    ```bash
    python main.py
    ```

* 평가 결과 출력

    `evaluation.ipynb`에서 모든 셀 실행
