"""
실행 가능한 Web3 AI Agent Fine-tuning 코드
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
import os
from pathlib import Path

@dataclass
class Web3TrainingConfig:
    """Web3 AI Agent 훈련 설정"""
    
    # 모델 설정
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 2048  # 메모리 절약을 위해 줄임
    
    # 훈련 설정
    num_epochs: int = 3
    batch_size: int = 2  # 메모리 절약을 위해 줄임
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    
    # LoRA 설정
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

class Web3DataProcessor:
    """Web3 대화 데이터 전처리기"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Web3 특화 토큰들 추가
        self.web3_tokens = [
            "<function_call>", "</function_call>",
            "<observation>", "</observation>",
        ]
        
        # 토크나이저에 특수 토큰 추가
        num_added = self.tokenizer.add_tokens(self.web3_tokens)
        print(f"Added {num_added} special tokens")
    
    def process_conversation(self, conversation_data: Dict) -> Dict[str, Any]:
        """대화 데이터를 모델 훈련용으로 변환"""
        
        conversations = conversation_data["conversations"]
        tools = json.loads(conversation_data["tools"]) if isinstance(conversation_data["tools"], str) else conversation_data["tools"]
        
        # 시스템 프롬프트 구성
        system_prompt = self._build_system_prompt(tools)
        
        # 대화 이력 구성
        conversation_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for conv in conversations:
            role = conv["role"]
            content = conv["content"]
            
            if role == "user":
                conversation_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            
            elif role == "function_call":
                # Function call을 특수 형식으로 변환
                formatted_call = self._format_function_call(content)
                conversation_text += f"<|im_start|>assistant\n{formatted_call}<|im_end|>\n"
            
            elif role == "observation":
                # API 결과를 특수 형식으로 변환
                formatted_obs = f"<observation>{content}</observation>"
                conversation_text += f"<|im_start|>system\n{formatted_obs}<|im_end|>\n"
            
            elif role == "assistant":
                conversation_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # 토큰화
        tokenized = self.tokenizer(
            conversation_text,
            truncation=True,
            max_length=2048,
            padding=False,  # 동적 패딩 사용
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0].clone()  # 라벨은 입력과 동일
        }
    
    def _build_system_prompt(self, tools: List[Dict]) -> str:
        """도구 정보를 포함한 시스템 프롬프트 생성"""
        
        tools_desc = []
        for tool in tools:
            tool_info = f"{tool['name']}: {tool['description']}"
            tools_desc.append(tool_info)
        
        system_prompt = f"""You are a Web3 AI Agent specialized in blockchain and DeFi operations.

Available Tools:
{chr(10).join(tools_desc)}

When you need to call a function, use this format:
<function_call>
{{"name": "function_name", "arguments": {{"param": "value"}}}}
</function_call>"""
        
        return system_prompt
    
    def _format_function_call(self, function_call: str) -> str:
        """Function call을 표준 형식으로 변환"""
        try:
            # JSON 파싱하여 검증
            call_data = json.loads(function_call)
            formatted_call = f"<function_call>\n{json.dumps(call_data, indent=2)}\n</function_call>"
            return formatted_call
        except:
            # 파싱 실패 시 원본 반환
            return f"<function_call>\n{function_call}\n</function_call>"

class Web3AgentTrainer:
    """Web3 AI Agent 훈련기"""
    
    def __init__(self, config: Web3TrainingConfig):
        self.config = config
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 데이터 프로세서 초기화
        self.data_processor = Web3DataProcessor(self.tokenizer)
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 토큰 추가로 인한 임베딩 크기 조정
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # LoRA 설정
        if config.use_lora:
            self._setup_lora()
    
    def _setup_lora(self):
        """LoRA 설정"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            print("✅ LoRA configuration applied")
        except ImportError:
            print("❌ PEFT library not found. Please install: pip install peft")
            print("Continuing without LoRA...")
    
    def load_dataset_from_file(self, file_path: str) -> List[Dict]:
        """파일에서 데이터셋 로드"""
        if not os.path.exists(file_path):
            print(f"❌ Dataset file not found: {file_path}")
            return self._create_sample_data()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} conversations from {file_path}")
            return data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict]:
        """샘플 데이터 생성 (실제 데이터가 없을 때)"""
        print("📝 Creating sample data for demonstration...")
        
        sample_data = [
            {
                "conversations": [
                    {"role": "user", "content": "I want to invest 5 ETH across the top 3 AI tokens with the highest social media momentum."},
                    {"role": "function_call", "content": '{"name": "get_social_media_momentum", "arguments": {"category": "AI", "limit": 3}}'},
                    {"role": "observation", "content": '{"tokens": ["AI Token1", "AI Token2", "AI Token3"]}'},
                    {"role": "function_call", "content": '{"name": "create_portfolio", "arguments": {"total_eth": 5, "tokens": ["AI Token1", "AI Token2", "AI Token3"]}}'},
                    {"role": "observation", "content": '{"allocation": {"AI Token1": "1.67 ETH", "AI Token2": "1.67 ETH", "AI Token3": "1.66 ETH"}}'},
                    {"role": "assistant", "content": "Successfully created a balanced portfolio allocation across the top 3 AI tokens based on social media momentum."}
                ],
                "tools": '[{"name": "get_social_media_momentum", "description": "Get social media momentum for tokens", "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["category", "limit"]}}, {"name": "create_portfolio", "description": "Create investment portfolio", "parameters": {"type": "object", "properties": {"total_eth": {"type": "number"}, "tokens": {"type": "array"}}, "required": ["total_eth", "tokens"]}}]'
            },
            {
                "conversations": [
                    {"role": "user", "content": "What's the current price of Bitcoin?"},
                    {"role": "function_call", "content": '{"name": "get_current_prices", "arguments": {"tokens": ["BTC"]}}'},
                    {"role": "observation", "content": '{"BTC": "$45,000"}'},
                    {"role": "assistant", "content": "The current price of Bitcoin (BTC) is $45,000."}
                ],
                "tools": '[{"name": "get_current_prices", "description": "Get current token prices", "parameters": {"type": "object", "properties": {"tokens": {"type": "array"}}, "required": ["tokens"]}}]'
            }
        ]
        
        # 데이터 복제하여 더 많은 훈련 샘플 생성
        extended_data = []
        for _ in range(50):  # 100개 샘플 생성
            extended_data.extend(sample_data)
        
        return extended_data
    
    def prepare_dataset(self, conversation_data: List[Dict]) -> Dataset:
        """데이터셋 준비"""
        
        processed_data = []
        for i, conv_data in enumerate(conversation_data):
            try:
                processed = self.data_processor.process_conversation(conv_data)
                processed_data.append(processed)
            except Exception as e:
                print(f"❌ Error processing conversation {i}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(processed_data)} conversations")
        return Dataset.from_list(processed_data)
    
    def train(self, dataset_path: str = None):
        """모델 훈련"""
        
        # 데이터셋 로드
        if dataset_path:
            conversation_data = self.load_dataset_from_file(dataset_path)
        else:
            conversation_data = self._create_sample_data()
        
        # 데이터셋 준비
        dataset = self.prepare_dataset(conversation_data)
        
        # 훈련/검증 분할
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size)) if train_size > 0 else dataset
        eval_dataset = dataset.select(range(train_size, len(dataset))) if len(dataset) > train_size else None
        
        # 출력 디렉토리 생성
        output_dir = "./web3-agent-model"
        os.makedirs(output_dir, exist_ok=True)
        
        # 훈련 인자 설정
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            logging_dir="./logs",
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM이므로 False
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 훈련 시작
        print("🚀 Starting Web3 AI Agent training...")
        try:
            trainer.train()
            
            # 모델 저장
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            print("✅ Training completed successfully!")
            print(f"📁 Model saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

def main():
    """메인 실행 함수"""
    print("🔧 Initializing Web3 AI Agent Fine-tuning...")
    
    # 설정
    config = Web3TrainingConfig()
    
    # 트레이너 생성
    trainer = Web3AgentTrainer(config)
    
    # 훈련 실행
    # dataset.json 파일이 있다면 해당 파일을 사용, 없다면 샘플 데이터 사용
    trainer.train(dataset_path="dataset.json")

if __name__ == "__main__":
    main()