from mlx_lm import load, stream_generate

repo = "mlx-community/granite-4.0-h-micro-4bit"
model, tokenizer = load(repo)

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
print()