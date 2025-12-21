import argparse
import uvicorn
import torch
import json
from threading import Thread
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
from eval_llm import init_model 

app = FastAPI()

# 全局变量
model = None
tokenizer = None
args = None

def parse_args():
    parser = argparse.ArgumentParser(description="gugugaga API Server")
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--lora_weight', default='None', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--port', default=8000, type=int)
    return parser.parse_args()

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, args
    args = parse_args()
    print("正在加载模型...")
    model, tokenizer = init_model(args)
    print("模型加载完成，服务已启动！")

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.85)
    top_p = data.get("top_p", 0.85)
    max_new_tokens = data.get("max_new_tokens", 8192)
    if args.weight == 'pretrain':
        prompt = messages[-1]['content']
        inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(args.device)
    else:
        templates = {"conversation": messages, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': 
            templates["enable_thinking"] = True
        prompt_str = tokenizer.apply_chat_template(**templates)
        inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True).to(args.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        repetition_penalty=1.0
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    def response_generator():
        for new_text in streamer:
            yield new_text

    return StreamingResponse(response_generator(), media_type="text/plain")

if __name__ == "__main__":
    temp_args = parse_args()
    uvicorn.run(app, host=temp_args.host, port=temp_args.port)