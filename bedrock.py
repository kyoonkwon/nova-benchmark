import time
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from botocore.exceptions import ClientError
from logger import logger


models = [
    {
      "model_name": "Claude 3.7 Sonnet",
      "model_id": "anthropic.claude-3-7-sonnet-20250219-v1:0",
      "regions": 
        {
          "us-east-1": "us",
          "eu-central-1": "eu"
        }
    },
    {
      "model_name": "Claude 3.5 Sonnet V2",
      "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
      "regions": 
        {
          "ap-northeast-2": "apac",
          "us-east-1": "us"
        }
    },
    {
      "model_name": "Nova Pro",
      "model_id": "amazon.nova-pro-v1:0",
      "regions": 
        {
          "ap-northeast-2": "apac",
          "us-east-1": "us",
          "eu-central-1": "eu"
        }
    },
]


def invoke_with_retry(messages, model_name="Claude 3.7 Sonnet", max_retries=3):
    """
    쓰로틀링 발생 시 다른 리전으로 전환하며 API 호출을 재시도하는 함수
    """

    # filter model from models using model_name
    model = next((item for item in models if item["model_name"] == model_name), None)

    if(model == None):
      raise Exception("model_name is not valid")
    
    model_id = model["model_id"]
    available_regions = list(model["regions"].keys())
    
    # 출력 파서 설정
    output_parser = StrOutputParser()
    
    retry_count = 0
    
    for region in available_regions:
        try:
            logger.info(f"리전 {region}에서 API 호출 시도 중...")
            
            # 리전에 맞는 모델 ID 생성
            region_prefix = model["regions"][region]
            cris_model_id = f"{region_prefix}.{model_id}"
            
            # ChatConverse 모델 초기화
            bedrock_model = ChatBedrockConverse(
                model=cris_model_id,
                temperature=0.7,
                max_tokens=1000,
                region_name=region
            )
            
            # 체인 구성
            chain = bedrock_model | output_parser
            
            # 모델 호출
            return chain.invoke(messages)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 'ServiceQuotaExceededException'] or 'Throttling' in str(e):
                logger.warning(f"리전 {region}에서 쓰로틀링 발생. 다음 리전으로 시도합니다.")
                retry_count += 1
                # 약간의 대기 시간 추가
                time.sleep(1)
                continue
            else:
                # 쓰로틀링이 아닌 다른 오류는 그대로 발생시킴
                raise e
    
    # 모든 리전에서 쓰로틀링이 발생한 경우
    raise Exception("모든 리전에서 쓰로틀링이 발생했습니다. 잠시 후 다시 시도해주세요.")

def generate_prompt(question, A, B, C, D):
    
    return HumanMessage(content=f"""주어진 질문을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.
    Question: {question}
    A: {A}
    B: {B}
    C: {C}
    D: {D}
    Answer: """)

if __name__ == "__main__":
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?"),
    ]
    
    # 단일 호출 테스트
    result = invoke_with_retry(messages)
    print("단일 호출 결과:", result)