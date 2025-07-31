"""
그래디언트 오류 수정된 Web3 AI Agent Fine-tuning 코드
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
    max_length: int = 1024
    
    # 훈련 설정
    num_epochs: int = 2
    batch_size: int = 1
    learning_rate: float = 5e-5  # 더 작은 학습률
    warmup_steps: int = 50
    
    # LoRA 설정
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

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
        
        # 대화 이력 구성 - 간단한 형식 사용
        conversation_text = f"System: {system_prompt}\n\n"
        
        for conv in conversations:
            role = conv["role"]
            content = conv["content"]
            
            if role == "user":
                conversation_text += f"User: {content}\n"
            elif role == "function_call":
                formatted_call = self._format_function_call(content)
                conversation_text += f"Assistant: {formatted_call}\n"
            elif role == "observation":
                conversation_text += f"System: <observation>{content}</observation>\n"
            elif role == "assistant":
                conversation_text += f"Assistant: {content}\n"
        
        # 토큰화
        tokenized = self.tokenizer(
            conversation_text,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors="pt"
        )
        
        # 입력과 라벨이 같은지 확인
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # 라벨은 입력과 동일
        }
    
    def _build_system_prompt(self, tools: List[Dict]) -> str:
        """도구 정보를 포함한 시스템 프롬프트 생성"""
        
        tools_desc = []
        for tool in tools:
            tool_info = f"{tool['name']}: {tool['description']}"
            tools_desc.append(tool_info)
        
        system_prompt = f"""You are a Web3 AI Agent. Available tools: {', '.join(tools_desc)}. Use <function_call>{{...}}</function_call> format for function calls."""
        
        return system_prompt
    
    def _format_function_call(self, function_call: str) -> str:
        """Function call을 표준 형식으로 변환"""
        try:
            call_data = json.loads(function_call)
            formatted_call = f"<function_call>{json.dumps(call_data)}</function_call>"
            return formatted_call
        except:
            return f"<function_call>{function_call}</function_call>"

class Web3AgentTrainer:
    """Web3 AI Agent 훈련기"""
    
    def __init__(self, config: Web3TrainingConfig):
        self.config = config
        
        print("🔧 Initializing tokenizer...")
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
            use_fast=True
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 데이터 프로세서 초기화
        self.data_processor = Web3DataProcessor(self.tokenizer)
        
        print("🔧 Loading model...")
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 토큰 추가로 인한 임베딩 크기 조정
        original_size = self.model.get_input_embeddings().weight.size(0)
        self.model.resize_token_embeddings(len(self.tokenizer))
        new_size = self.model.get_input_embeddings().weight.size(0)
        print(f"Resized embeddings: {original_size} -> {new_size}")
        
        # LoRA 설정
        if config.use_lora:
            self._setup_lora()
        else:
            # LoRA 없이 훈련하는 경우, 일부 레이어만 학습 가능하게 설정
            self._setup_partial_training()
    
    def _setup_lora(self):
        """LoRA 설정"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # 모델의 target modules 확인
            target_modules = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if any(target in name for target in ["q_proj", "v_proj", "k_proj", "o_proj"]):
                        target_modules.append(name.split('.')[-1])
            
            # 기본 target modules 사용 (Qwen 모델용)
            if not target_modules:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            print(f"Target modules for LoRA: {target_modules}")
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            print("✅ LoRA configuration applied")
            
        except ImportError:
            print("❌ PEFT library not found. Installing...")
            os.system("pip3 install peft")
            print("Please restart the script after installation.")
            exit(1)
        except Exception as e:
            print(f"❌ LoRA setup failed: {e}")
            print("Continuing without LoRA...")
            self._setup_partial_training()
    
    def _setup_partial_training(self):
        """LoRA 없이 부분 훈련 설정"""
        print("Setting up partial training (last layers only)...")
        
        # 모든 파라미터를 frozen으로 설정
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 마지막 몇 개 레이어만 학습 가능하게 설정
        num_layers = len(self.model.model.layers) if hasattr(self.model.model, 'layers') else 32
        layers_to_train = max(2, num_layers // 8)  # 전체 레이어의 1/8 또는 최소 2개
        
        print(f"Training last {layers_to_train} layers out of {num_layers}")
        
        # 마지막 레이어들 학습 가능하게 설정
        if hasattr(self.model.model, 'layers'):
            for i in range(num_layers - layers_to_train, num_layers):
                for param in self.model.model.layers[i].parameters():
                    param.requires_grad = True
        
        # 임베딩과 헤드도 학습 가능하게 설정
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
        
        # 새로 추가된 토큰 임베딩 학습 가능하게 설정
        embedding_params = self.model.get_input_embeddings().weight
        embedding_params.requires_grad = True
        
        # 학습 가능한 파라미터 수 확인
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _create_sample_data(self) -> List[Dict]:
        """샘플 데이터 생성"""
        print("📝 Creating sample data...")
        
        sample_data = [
            {
                "conversations": [
                    {"role": "user", "content": "What's the price of ETH?"},
                    {"role": "function_call", "content": '{"name": "get_current_prices", "arguments": {"tokens": ["ETH"]}}'},
                    {"role": "observation", "content": '{"ETH": "$2000"}'},
                    {"role": "assistant", "content": "The current price of Ethereum (ETH) is $2,000."}
                ],
                "tools": '[{"name": "get_current_prices", "description": "Get current token prices", "parameters": {"type": "object", "properties": {"tokens": {"type": "array"}}, "required": ["tokens"]}}]'
            },
            {
                "conversations": [
                    {"role": "user", "content": "Show me popular DeFi tokens"},
                    {"role": "function_call", "content": '{"name": "get_trending_tokens", "arguments": {"category": "DeFi", "limit": 5}}'},
                    {"role": "observation", "content": '{"tokens": ["UNI", "AAVE", "COMP", "MKR", "SNX"]}'},
                    {"role": "assistant", "content": "Here are the top 5 trending DeFi tokens: UNI, AAVE, COMP, MKR, and SNX."}
                ],
                "tools": '[{"name": "get_trending_tokens", "description": "Get trending tokens by category", "parameters": {"type": "object", "properties": {"category": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["category", "limit"]}}]'
            }
        ]
        
        # 데이터 복제
        extended_data = sample_data * 20  # 40개 샘플
        return extended_data
    
    def prepare_dataset(self, conversation_data: List[Dict]) -> Dataset:
        """데이터셋 준비"""
        
        processed_data = []
        for i, conv_data in enumerate(conversation_data):
            try:
                processed = self.data_processor.process_conversation(conv_data)
                
                # 텐서 크기 확인
                if len(processed["input_ids"]) > 0:
                    processed_data.append(processed)
                    
            except Exception as e:
                print(f"❌ Error processing conversation {i}: {e}")
                continue
        
        print(f"✅ Successfully processed {len(processed_data)} conversations")
        return Dataset.from_list(processed_data)
    
    def train(self, dataset_path: str = None):
        """모델 훈련"""
        
        # 데이터셋 준비
        conversation_data = self._create_sample_data()
        dataset = self.prepare_dataset(conversation_data)
        
        if len(dataset) == 0:
            print("❌ No valid data found!")
            return
        
        # 출력 디렉토리 생성
        output_dir = "./web3-agent-model"
        os.makedirs(output_dir, exist_ok=True)
        
        # 훈련 인자 설정 - 더 보수적으로
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,  # effective batch size 증가
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            logging_dir="./logs",
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            optim="adamw_torch",
            weight_decay=0.01,
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # 트레이너 생성
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 훈련 전 파라미터 체크
        print("\n🔍 Checking trainable parameters...")
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        if not trainable_params:
            print("❌ No trainable parameters found!")
            return
        
        print(f"✅ Found {len(trainable_params)} trainable parameter groups")
        
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
            import traceback
            traceback.print_exc()

def main():
    """메인 실행 함수"""
    print("🔧 Initializing Web3 AI Agent Fine-tuning...")
    
    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 설정
    config = Web3TrainingConfig()
    
    # 트레이너 생성
    trainer = Web3AgentTrainer(config)
    
    # 훈련 실행
    trainer.train()

if __name__ == "__main__":
    main()