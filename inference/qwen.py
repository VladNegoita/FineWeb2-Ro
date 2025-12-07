from transformers import AutoTokenizer, AutoModelForCausalLM

qwen_2_5 = ("Qwen/Qwen2.5-72B-Instruct", 32768)


def infer_qwen(model_spec, max_output_tokens=512):
    model_id, max_tokens = model_spec
    max_input_tokens = max_tokens - max_output_tokens
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, max_length=max_tokens - max_output_tokens
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    def infer(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt")
        if model_inputs.input_ids.size(1) > max_input_tokens:
            model_inputs.input_ids = model_inputs.input_ids[:, -max_input_tokens:]
            print("Truncated input for model %s!" % model_id)

        model_inputs = model_inputs.to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_output_tokens,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return infer


if __name__ == "__main__":
    model = infer_qwen(qwen_2_5)

    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": "Who are you?"},
    ]

    print(model(messages))
