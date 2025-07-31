"""
Web3 AI Agent 추론 전용 스크립트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

def load_trained_model(model_path="./web3-agent-model"):
    """훈련된 모델 로드"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run the training script first: python3 train.py")
        return None, None
    
    print(f"📚 Loading model from {model_path}...")
    
    try:
        # PEFT 어댑터 파일 확인
        adapter_files = ["adapter_model.bin", "adapter_model.safetensors", "pytorch_model.bin"]
        is_peft_model = any(os.path.exists(os.path.join(model_path, f)) for f in adapter_files[:2])
        
        if is_peft_model:
            # PEFT 모델인 경우
            try:
                from peft import PeftModel
                
                print("🔧 Loading PEFT model...")
                
                # 베이스 모델 로드
                base_model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                # 특수 토큰을 위한 임베딩 크기 조정
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
                special_tokens = ["<function_call>", "</function_call>", "<observation>", "</observation>"]
                tokenizer.add_tokens(special_tokens)
                base_model.resize_token_embeddings(len(tokenizer))
                
                # PEFT 어댑터 로드
                model = PeftModel.from_pretrained(base_model, model_path)
                print("✅ PEFT model loaded")
                
                # 저장된 토크나이저가 있으면 사용
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except:
                    pass
                
            except ImportError:
                print("❌ PEFT library required for this model")
                print("Install with: pip install peft")
                return None, None
        else:
            # 일반 모델인 경우
            print("🔧 Loading standard model...")
            
            # config.json 수정 (필요한 경우)
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if "model_type" not in config:
                    config["model_type"] = "qwen2"
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print("✅ Fixed config.json")
            
            # 토크나이저 먼저 로드
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 모델 로드
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            print("✅ Standard model loaded")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"🔧 Using GPU: {torch.cuda.get_device_name()}")
            print(f"🔧 Model device: {next(model.parameters()).device}")
        else:
            print("🔧 Using CPU")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure training completed successfully")
        print("2. Check if model files exist in the directory")
        print("3. Try running training again: python3 train.py")
        return None, None

def generate_response(model, tokenizer, user_input, max_new_tokens=200):
    """사용자 입력에 대한 응답 생성"""
    
    # 입력 전처리
    prompt = f"User: {user_input}\nAssistant:"
    
    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # GPU로 이동 (모델이 GPU에 있는 경우)
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 생성
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 디코딩
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 응답 부분만 추출
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    return response

def interactive_chat(model, tokenizer):
    """대화형 채팅 인터페이스"""
    
    print("\n🤖 Web3 AI Agent is ready!")
    print("💡 Commands:")
    print("  - Type your question and press Enter")
    print("  - 'quit' or 'exit' to quit")
    print("  - 'clear' to clear screen")
    print("  - 'test' to run test cases")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() == 'test':
                run_test_cases(model, tokenizer)
                continue
            elif not user_input:
                continue
            
            print("🤔 Thinking...", end="", flush=True)
            
            response = generate_response(model, tokenizer, user_input)
            print(f"\r🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def run_test_cases(model, tokenizer):
    """미리 정의된 테스트 케이스 실행"""
    
    test_cases = [
        "What's the price of ETH?",
        "Show me popular DeFi tokens",
        "I want to swap 1 ETH for USDC",
        "Check my wallet balance",
        "What are the gas fees right now?",
        "Find the best yield farming opportunities",
        "Get information about Uniswap protocol"
    ]
    
    print("\n🧪 Running test cases...")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_input}")
        try:
            response = generate_response(model, tokenizer, test_input, max_new_tokens=150)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)

def single_query_mode(model, tokenizer, query):
    """단일 쿼리 모드"""
    
    print(f"📝 Query: {query}")
    try:
        response = generate_response(model, tokenizer, query)
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """메인 함수"""
    
    print("🚀 Web3 AI Agent Inference")
    print("=" * 40)
    
    # 모델 로드
    model, tokenizer = load_trained_model()
    
    if model is None or tokenizer is None:
        return
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test_cases(model, tokenizer)
        else:
            # 단일 쿼리 실행
            query = " ".join(sys.argv[1:])
            single_query_mode(model, tokenizer, query)
    else:
        # 대화형 모드
        interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()