from email import message
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


gemma2_27b = ("google/gemma-2-27b-it", 8192)
gemma2_9b = ("google/gemma-2-9b-it", 8192)


def infer_gemma2(model_spec, max_output_tokens=512):
    model_name, max_tokens = model_spec
    max_input_tokens = max_tokens - max_output_tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    def infer(messages):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        if input_ids.size(1) > max_input_tokens:
            input_ids = input_ids[:, -max_input_tokens:]
            print("Truncated input for model %s!" % model_name)

        input_ids = input_ids.to(model.device)

        outputs = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_output_tokens,
        )

        response = outputs[0][input_ids.shape[-1] :]
        return tokenizer.decode(response, skip_special_tokens=True)

    return infer


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    model = infer_gemma2(gemma2_27b)
    response = model(messages)
    print(response)
