from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

llama4_scout = ("meta-llama/Llama-4-Scout-17B-16E-Instruct", 10000000)


def infer_llama4(model_spec, max_output_tokens=512):
    processor = AutoProcessor.from_pretrained(model_spec[0])
    max_input_tokens = model_spec[1] - max_output_tokens
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_spec[0],
        attn_implementation="flex_attention",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    def infer(messages):
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        token_count = inputs["input_ids"].size(1)
        if token_count > max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
            print("Truncated input to max_input_tokens:", max_input_tokens)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
        )

        response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :])[
            0
        ]
        return response

    return infer


if __name__ == "__main__":
    model = infer_llama4(llama4_scout, max_output_tokens=512)
    prompt = "What is the capital of France?"
    response = model([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    print(response)
