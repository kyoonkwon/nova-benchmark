import time
from botocore.exceptions import ClientError
from logger import logger
import logging
from typing import Literal  # noqa: F401 (kept for future type hints)
import boto3
import json


logging.getLogger("botocore").setLevel(logging.WARNING)


models = [
    {
        "model_name": "Qwen3 235B A22B 2507",
        "model_id": "qwen.qwen3-235b-a22b-2507-v1:0",
        "regions": {
            "ap-northeast-1": "",
            "ap-south-1": "",
            "eu-north-1": "",
            "eu-south-1": "",
            "eu-west-2": "",
            "us-west-2": "",
        },
    },
    {
        "model_name": "DeepSeek-V3.1",
        "model_id": "deepseek.v3-v1:0",
        "regions": {
            "ap-northeast-1": "",
            "ap-south-1": "",
            "eu-north-1": "",
            "eu-west-2": "",
            "us-west-2": "",
        },
    },
    {
        "model_name": "Qwen3 32B (dense)",
        "model_id": "qwen.qwen3-32b-v1:0",
        "regions": {
            "ap-northeast-1": "",
            "ap-south-1": "",
            "eu-north-1": "",
            "eu-south-1": "",
            "eu-west-2": "",
            "us-west-2": "",
        },
    },
    {
        "model_name": "Qwen3-Coder-30B-A3B-Instruct",
        "model_id": "qwen.qwen3-coder-30b-a3b-v1:0",
        "regions": {
            "ap-northeast-1": "",
            "ap-south-1": "",
            "eu-north-1": "",
            "eu-south-1": "",
            "eu-west-2": "",
            "us-west-2": "",
        },
    },
    {
        "model_name": "Qwen3 Coder 480B A35B Instruct",
        "model_id": "qwen.qwen3-coder-480b-a35b-v1:0",
        "regions": {
            "ap-northeast-1": "",
            "ap-south-1": "",
            "eu-north-1": "",
            "eu-west-2": "",
            "us-west-2": "",
        },
    },
    {
        "model_name": "Claude Opus 4.1",
        "model_id": "anthropic.claude-opus-4-1-20250805-v1:0",
        "regions": {"us-east-1": "us"},
    },
    {
        "model_name": "Claude Sonnet 4",
        "model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
        "regions": {"us-east-1": "us", "eu-central-1": "eu", "ap-northeast-1": "apac"},
    },
    {
        "model_name": "Claude 3.7 Sonnet",
        "model_id": "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "regions": {"us-east-1": "us", "eu-central-1": "eu"},
    },
    {
        "model_name": "Claude 3.5 Sonnet V2",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "regions": {"ap-northeast-2": "apac", "us-east-1": "us"},
    },
    {
        "model_name": "Claude 3.5 Haiku",
        "model_id": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "Nova Pro",
        "model_id": "amazon.nova-pro-v1:0",
        "regions": {"us-east-1": "us", "ap-northeast-2": "apac", "eu-central-1": "eu"},
    },
    {
        "model_name": "Nova Premier",
        "model_id": "amazon.nova-premier-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "Nova Micro",
        "model_id": "amazon.nova-micro-v1:0",
        "regions": {"us-east-1": "us", "ap-northeast-2": "apac", "eu-central-1": "eu"},
    },
    {
        "model_name": "Nova Lite",
        "model_id": "amazon.nova-lite-v1:0",
        "regions": {"us-east-1": "us", "ap-northeast-2": "apac", "eu-central-1": "eu"},
    },
    {
        "model_name": "Llama 3.1 70B Instruct",
        "model_id": "meta.llama3-1-70b-instruct-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "Llama 3.3 70B Instruct",
        "model_id": "meta.llama3-3-70b-instruct-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "Llama 4 Maverick 17B Instruct",
        "model_id": "meta.llama4-maverick-17b-instruct-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "Llama 4 Scout 17B Instruct",
        "model_id": "meta.llama4-scout-17b-instruct-v1:0",
        "regions": {
            "us-east-1": "us",
        },
    },
    {
        "model_name": "GPT OSS 120B",
        "model_id": "openai.gpt-oss-120b-1:0",
        "regions": {
            "us-west-2": "",
        },
    },
    {
        "model_name": "GPT OSS 20B",
        "model_id": "gpt-oss-20b-1:0",
        "regions": {
            "us-west-2": "openai",
        },
    },
    {
        "model_name": "Palmyra x5",
        "model_id": "writer.palmyra-x5-v1:0",
        "regions": {"us-west-2": "us"},
    },
]


def invoke_with_retry(messages, model_name="Claude 3.7 Sonnet", max_retries=10):
    """
    쓰로틀링 발생 시 다른 리전으로 전환하며 API 호출을 재시도하는 함수
    """

    # filter model from models using model_name
    model = next((item for item in models if item["model_name"] == model_name), None)

    if model is None:
        raise Exception("model_name is not valid")

    model_id = model["model_id"]
    available_regions = list(model["regions"].keys())

    retry_count = 0

    for region in available_regions:
        try:
            logger.info(f"region: {region} / model_id: {model_id}")

            # 리전에 맞는 모델 ID 생성
            region_prefix = model["regions"][region]
            if region_prefix:
                model_id = f"{region_prefix}.{model_id}"

            # Bedrock Runtime 클라이언트 생성
            client = boto3.client("bedrock-runtime", region_name=region)

            # Converse API 호출
            response = client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={"temperature": 0, "maxTokens": 1000},
            )

            output_text = response["output"]["message"]["content"][-1]["text"]

            # 메트릭/사용량 추출
            metrics = response.get("metrics", {})
            input_token = response.get("usage", {}).get("inputTokens", None)
            output_token = response.get("usage", {}).get("outputTokens", None)
            latency = metrics.get("latencyMs")
            
            return (
                output_text,
                input_token,
                output_token,
                latency,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")

            if error_code in [
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceQuotaExceededException",
            ] or "Throttling" in str(e):
                logger.warning(
                    f"ThrottlingException in {region}. Trying Different Region."
                )
                retry_count += 1
                # 약간의 대기 시간 추가
                time.sleep(1)
                continue
            else:
                # 쓰로틀링이 아닌 다른 오류는 그대로 발생시킴
                raise e

    # 모든 리전에서 쓰로틀링이 발생한 경우
    raise Exception(
        "ThrottlingException in all available regions. Try again after few minutes later"
    )


def generate_prompt(question, A, B, C, D):
    text = f"""{question}
    A: {A}
    B: {B}
    C: {C}
    D: {D}
    정답:"""
    return [{"role": "user", "content": [{"text": text}]}]


def generate_gpt_prompt(question, A, B, C, D):
    return [
        {
            "role": "user",
            "content": [{"text": f"""question: {question} A: {A} B: {B} C: {C} D: {D} 정답:"""}],
        },
    ]


if __name__ == "__main__":
    messages = generate_prompt("다음 유물이 사용된 시대에 대한 설명으로 옳은 것은? 미송리식 토기, 팽이형 토기, 붉은 간 토기", "비파형 동검이 사용되었다.", "오수전 등의 화폐가 사용되었다.", "아슐리안형 주먹도끼가 사용되었다.", "철이 많이 생산되어 낙랑과 왜에 수출되었다.")

    # 단일 호출 테스트
    output, input_token, output_token, latency = invoke_with_retry(messages, model_name="DeepSeek-V3.1")
    print("output: ", output)
    print("input_token: ", input_token)
    print("output_token: ", output_token)
    print("latency: ", latency)
