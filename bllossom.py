import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

# NCCL 백엔드를 위한 환경 변수 설정
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def check_memory_usage(stage):
    print(f"{stage} - Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"{stage} - Cached GPU memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

def main(rank, world_size):
    setup(rank, world_size)

    model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

    check_memory_usage("Before loading model and tokenizer")

    try:
        # 모델과 토크나이저 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            load_in_8bit=True,  # 8-bit 양자화
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        check_memory_usage("After loading model and tokenizer")
    except RuntimeError as e:
        print(f"Error during model loading: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    try:
        # 데이터셋 로드
        dataset = load_dataset('squarelike/ko_medical_chat')
        check_memory_usage("After loading dataset")
    except RuntimeError as e:
        print(f"Error during dataset loading: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    try:
        # 데이터셋에서 50개만 선택
        dataset = dataset['train'].select(range(50)).train_test_split(test_size=0.1)
        check_memory_usage("After selecting dataset subset")
    except RuntimeError as e:
        print(f"Error during dataset selection: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    # 데이터 전처리 함수
    def preprocess_function(examples):
        formatted_texts = []
        for conversation in examples['conversations']:
            formatted_conversation = ""
            for message in conversation:
                if message['from'] == 'client':
                    formatted_conversation += f"User: {message['value']}\n"
                elif message['from'] == 'doctor':
                    formatted_conversation += f"Doctor: {message['value']}\n"
            formatted_texts.append(formatted_conversation.strip())  # 마지막에 불필요한 공백 제거
        
        # 토큰화
        tokenized_inputs = tokenizer(formatted_texts, truncation=True, padding='max_length', max_length=512)

        # 텐서 형식으로 변환
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    try:
        # 데이터셋에 전처리 함수 적용
        tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=world_size, remove_columns=["conversations", "id"])
        check_memory_usage("After preprocessing dataset")
    except RuntimeError as e:
        print(f"Error during dataset preprocessing: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    # 전처리된 데이터셋 확인 (샘플 출력)
    print("Sample preprocessed dataset:", tokenized_dataset["train"][0])

    # 각 프로세스에 데이터셋 샤딩 적용
    def shard_dataset(dataset, rank, world_size):
        dataset_length = len(dataset)
        shard_size = dataset_length // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size
        return dataset.select(range(start_idx, end_idx))

    train_dataset = shard_dataset(tokenized_dataset["train"], rank, world_size)
    eval_dataset = shard_dataset(tokenized_dataset["test"], rank, world_size)

    check_memory_usage("After sharding dataset")
    torch.cuda.empty_cache()

    try:
        # LoRA 설정
        lora_config = LoraConfig(
            r=16,  # 랭크
            lora_alpha=32,  # LoRA 알파값
            lora_dropout=0.1,  # 드롭아웃 비율
            bias="none",  # 바이어스 설정
            task_type="CAUSAL_LM"  # 작업 유형
        )
        model = get_peft_model(model, lora_config)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        check_memory_usage("After applying LoRA config")
    except RuntimeError as e:
        print(f"Error during applying LoRA config: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    # 데이터 컬레이터 생성 (동적 패딩 적용)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 파인튜닝 설정
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # 배치 크기를 줄입니다.
        per_device_eval_batch_size=1,   # 배치 크기를 줄입니다.
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=8,  # Gradient Accumulation을 증가시킵니다.
        fp16=True,  # Mixed Precision Training 사용
        dataloader_pin_memory=False,  # 핀 메모리 비활성화 (메모리 최적화)
        report_to="none",  # Disable reporting to save memory
        remove_unused_columns=False  # 사용되지 않는 컬럼 제거 비활성화
    )

    check_memory_usage("After setting training arguments")
    torch.cuda.empty_cache()

    try:
        # 모델을 DDP로 래핑
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        check_memory_usage("After wrapping model with DDP")
    except RuntimeError as e:
        print(f"Error during wrapping model with DDP: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    try:
        # Trainer 생성
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # 모델 파인튜닝
        trainer.train()
        check_memory_usage("After training")
    except RuntimeError as e:
        print(f"Error during training: {e}")
        cleanup()
        return
    
    torch.cuda.empty_cache()

    # 모델 저장
    if rank == 0:  # 주 프로세스에서만 저장
        try:
            trainer.save_model("path_to_save_your_model")
            tokenizer.save_pretrained("path_to_save_your_model")
        except RuntimeError as e:
            print(f"Error during saving model: {e}")

    cleanup()

if __name__ == "__main__":
    world_size = 2  # 사용할 GPU의 개수
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
