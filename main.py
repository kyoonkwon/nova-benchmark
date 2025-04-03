from datasets import load_dataset
from bedrock import generate_prompt, invoke_with_retry
from tqdm import tqdm
import concurrent.futures
import pandas as pd
import os
from logger import logger

logger.setLevel("ERROR")

hf_dataset_id = "HAERAE-HUB/KMMLU"
dataset_name = "KMMLU"
kmmlu_category = [
    "Accounting",
    # "Agricultural-Sciences",
    # "Aviation-Engineering-and-Maintenance",
    # "Biology",
    # "Chemical-Engineering",
    # "Chemistry",
    # "Civil-Engineering",
    # "Computer-Science",
    # "Construction",
    # "Criminal-Law",
    # "Ecology",
    # "Economics",
    # "Education",
    # "Electrical-Engineering",
    # "Electronics-Engineering",
    # "Energy-Management",
    # "Environmental-Science",
    # "Fashion",
    # "Food-Processing",
    # "Gas-Technology-and-Engineering",
    # "Geomatics",
    # "Health",
    # "Industrial-Engineer",
    # "Information-Technology",
    # "Interior-Architecture-and-Design",
    # "Korean-History",
    # "Law",
    # "Machine-Design-and-Manufacturing",
    # "Management",
    # "Maritime-Engineering",
    # "Marketing",
    # "Materials-Engineering",
    # "Math",
    # "Mechanical-Engineering",
    # "Nondestructive-Testing",
    # "Patent",
    # "Political-Science-and-Sociology",
    # "Psychology",
    # "Public-Safety",
    # "Railway-and-Automotive-Engineering",
    # "Real-Estate",
    # "Refrigerating-Machinery",
    # "Social-Welfare",
    # "Taxation",
    # "Telecommunications-and-Wireless-Technology",
]

def process_item(item, model_name="Nova Pro", max_retries=3):
    """각 항목을 처리하는 함수"""
    question = item["question"]
    A = item["A"]
    B = item["B"]
    C = item["C"]
    D = item["D"]

    messages = [
        generate_prompt(question, A, B, C, D),
    ]

    retry_count = 0
    while retry_count < max_retries:
        try:
            result = invoke_with_retry(messages, model_name=model_name)
            return result
        except Exception as exc:
            retry_count += 1
            if retry_count >= max_retries:
                raise Exception(f"최대 재시도 횟수({max_retries})를 초과했습니다: {exc}")
            logger.error(f"오류 발생, 재시도 중... ({retry_count}/{max_retries}): {exc}")
            # 재시도 사이에 짧은 지연 시간 추가
            import time
            time.sleep(30)

# 결과 저장을 위한 디렉토리 생성
os.makedirs('results', exist_ok=True)

for idx, category in enumerate(kmmlu_category):
    dataset = load_dataset(hf_dataset_id, category)
    
    # ds_dev = dataset["dev"]
    # ds_dev = ds_dev.map(lambda x: {"answer": "ABCD"[x["answer"]-1]})

    ds_test = dataset["test"]# .select(range(10))
    ds_test = ds_test.map(lambda x: {"answer": "ABCD"[x["answer"]-1]})
    
    # 병렬 처리를 위한 리스트 생성
    items = list(ds_test)
    results = [None] * len(items)  # 결과를 저장할 리스트
    
    # 프로그레스 바 생성
    with tqdm(total=len(items), desc=f"Processing {category}") as pbar:
        # 최대 5개의 작업을 동시에 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 모든 작업을 예약하고 future 객체를 저장
            future_to_idx = {
                executor.submit(process_item, item): i 
                for i, item in enumerate(items)
            }
        
            # 작업이 완료되는 대로 결과 처리
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    pbar.update(1)
                except Exception as exc:
                    # 실패한 작업을 다시 예약
                    logger.warning(f'작업 {idx}에서 에러 발생: {exc}, 재시도합니다.')
                    retry_future = executor.submit(process_item, items[idx])
                    future_to_idx[retry_future] = idx
                    # 프로그레스 바는 업데이트하지 않음 (재시도 작업이 성공할 때만 업데이트)
    
    # 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame({
        'question': [item['question'] for item in items],
        'A': [item['A'] for item in items],
        'B': [item['B'] for item in items],
        'C': [item['C'] for item in items],
        'D': [item['D'] for item in items],
        'correct_answer': [item['answer'] for item in items],
        'model_response': results
    })
    
    # CSV 파일로 저장
    output_path = f'results/{category}_results.csv'
    result_df.to_csv(output_path, index=False)
    logger.info(f'{category} 결과가 {output_path}에 저장되었습니다.')
