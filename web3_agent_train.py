"""
Web3 AI Agent 훈련 전용 스크립트
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments
)
from datasets import Dataset
import json
import os

def load_training_data(file_path="training_set.jsonl"):
    """training_set.jsonl 파일에서 데이터 로드"""
    if not os.path.exists(file_path):
        print(f"❌ Training file not found: {file_path}")
        print("Creating sample data instead...")
        return create_fallback_data()
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    json_data = json.loads(line.strip())
                    
                    # 대화를 텍스트 형식으로 변환
                    text = convert_conversation_to_text(json_data)
                    if text:
                        data.append({"text": text})
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing error on line {line_num}: {e}")
                except Exception as e:
                    print(f"⚠️ Error processing line {line_num}: {e}")
        
        print(f"✅ Loaded {len(data)} conversations from {file_path}")
        return data
        
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return create_fallback_data()

def convert_conversation_to_text(json_data):
    """JSON 대화 데이터를 텍스트 형식으로 변환"""
    try:
        conversations = json_data.get("conversations", [])
        if not conversations:
            return None
        
        text_parts = []
        
        for conv in conversations:
            role = conv.get("role", "")
            content = conv.get("content", "")
            
            if role == "user":
                text_parts.append(f"User: {content}")
            elif role == "function_call":
                text_parts.append(f"Assistant: <function_call>{content}</function_call>")
            elif role == "observation":
                text_parts.append(f"System: <observation>{content}</observation>")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")
        
        return "\n".join(text_parts)
        
    except Exception as e:
        print(f"⚠️ Error converting conversation: {e}")
        return None

def create_fallback_data():
    """파일이 없을 때 사용할 폴백 데이터"""
    return [
        {
            "text": "User: What's the price of ETH?\nAssistant: <function_call>{\"name\": \"get_current_prices\", \"arguments\": {\"tokens\": [\"ETH\"]}}</function_call>\nSystem: <observation>{\"ETH\": \"$2000\"}</observation>\nAssistant: The current price of Ethereum (ETH) is $2,000."
        },
        {
            "text": "User: Show me popular DeFi tokens\nAssistant: <function_call>{\"name\": \"get_trending_tokens\", \"arguments\": {\"category\": \"DeFi\", \"limit\": 5}}</function_call>\nSystem: <observation>{\"tokens\": [\"UNI\", \"AAVE\", \"COMP\", \"MKR\", \"SNX\"]}</observation>\nAssistant: Here are the top 5 trending DeFi tokens: UNI, AAVE, COMP, MKR, and SNX."
        }
    ] * 20  # 40개 샘플

def tokenize_function(examples, tokenizer):
    """토큰화 함수"""
    # 텍스트를 토큰화
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None  # 리스트로 반환
    )
    
    # 라벨 설정 (입력과 동일)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def main():
    print("🚀 Starting Web3 AI Agent Training...")
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        print("🔧 Using CPU")
    
    # 모델 및 토크나이저 로드
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Web3 특화 토큰 추가
    special_tokens = ["<function_call>", "</function_call>", "<observation>", "</observation>"]
    num_added_tokens = tokenizer.add_tokens(special_tokens)
    print(f"✅ Added {num_added_tokens} special tokens")
    
    print("🤖 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # 토큰 크기 조정
    original_size = model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.get_input_embeddings().weight.size(0)
    print(f"📏 Resized embeddings: {original_size} -> {new_size}")
    
    # PEFT 설정
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        print("🔧 Setting up LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("✅ LoRA configuration applied")
        
    except ImportError:
        print("⚠️ PEFT not available, using partial fine-tuning")
        # 일부 레이어만 학습
        for param in model.parameters():
            param.requires_grad = False
        
        # 마지막 2개 레이어만 학습
        if hasattr(model.model, 'layers'):
            for layer in model.model.layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 헤드 학습
        for param in model.lm_head.parameters():
            param.requires_grad = True
            
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 데이터 준비
    print("📝 Loading training dataset...")
    raw_data = load_training_data("training_set.jsonl")
    
    if not raw_data:
        print("❌ No valid training data found!")
        return
    
    dataset = Dataset.from_list(raw_data)
    
    # 토큰화
    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ Dataset prepared: {len(tokenized_dataset)} samples")
    
    # 출력 디렉토리 설정
    output_dir = "./web3-agent-model"
    os.makedirs(output_dir, exist_ok=True)
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=2,
        save_steps=20,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        report_to=None,
        logging_dir="./logs",
        save_strategy="steps",
        evaluation_strategy="no",
    )
    
    # 커스텀 데이터 콜레이터
    def data_collator(features):
        # 최대 길이 계산
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            
            # 패딩
            padding_length = max_length - len(input_ids)
            
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # 텐서로 변환
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 훈련 시작
    print("🚀 Starting training...")
    try:
        trainer.train()
        
        # 모델 저장
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # config.json 수정 (model_type 추가)
        import json
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # model_type이 없으면 추가
            if "model_type" not in config:
                config["model_type"] = "qwen2"
                
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Config file updated with model_type")
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
        # 훈련 완료 메시지
        print("\n" + "="*50)
        print("🎉 TRAINING COMPLETED!")
        print("="*50)
        print(f"📂 Model location: {output_dir}")
        print("🔄 To test the model, run: python3 inference.py")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()