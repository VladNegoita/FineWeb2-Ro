from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

gemma3_4b = ("google/gemma-3-4b-it", 128000)
gemma3_12b = ("google/gemma-3-12b-it", 128000)
gemma3_27b = ("google/gemma-3-27b-it", 128000)


def infer_gemma3(model_spec, max_output_tokens=512):
    model_name, max_tokens = model_spec
    max_input_tokens = max_tokens - max_output_tokens
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"

    def infer(messages):
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        if input_len > max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
            print("Truncated input for model %s!" % model_name)

        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=max_output_tokens, do_sample=False
            )
            generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        return decoded

    return infer


if __name__ == "__main__":
    model = infer_gemma3(gemma3_4b)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {"role": "user", "content": [{"type": "text", "text": "Buna, cum te cheama?"}]},
    ]
    print(model(messages))
