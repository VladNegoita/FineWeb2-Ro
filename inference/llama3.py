from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

llama3_8b = ("meta-llama/Meta-Llama-3-8B-Instruct", 8000)
llama3_1_8b = ("meta-llama/Meta-Llama-3.1-8B-Instruct", 8000)

llama3_70b = ("meta-llama/Meta-Llama-3-70B-Instruct", 128000)
llama3_1_70b = ("meta-llama/Meta-Llama-3.1-70B-Instruct", 128000)
llama3_3_70b = ("meta-llama/Llama-3.3-70B-Instruct", 128000)


def infer_llama3(model_spec, max_output_tokens=512):
    model_id, max_tokens = model_spec
    max_input_tokens = max_tokens - max_output_tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, max_length=max_tokens - max_output_tokens
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def infer(messages):
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )

        if input_ids.size(1) > max_input_tokens:
            input_ids = input_ids[:, -max_input_tokens:]
            print("Truncated input for model %s!" % model_id)

        input_ids = input_ids.to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = model.generate(
            input_ids,
            eos_token_id=terminators,
            do_sample=False,
            max_new_tokens=max_output_tokens,
        )
        response = outputs[0][input_ids.shape[-1] :]
        return tokenizer.decode(response, skip_special_tokens=True)

    return infer


if __name__ == "__main__":
    model = infer_llama3(llama3_3_70b)

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    print(model(messages))
