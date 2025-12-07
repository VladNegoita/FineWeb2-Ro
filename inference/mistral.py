from transformers import AutoProcessor, Mistral3ForConditionalGeneration
import torch

mistral_24b = ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", 128000)


def infer_mistral(model_spec, max_output_tokens=512):
    model_id, max_tokens = model_spec
    processor = AutoProcessor.from_pretrained(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.eval()

    def infer(messages):
        SYSTEM = [x["content"] for x in messages if x["role"] == "system"][0]
        USER = [x["content"] for x in messages if x["role"] == "user"][0]
        prompt = f"<s>[INST] {SYSTEM} [/SYS]\n{USER} [/INST]"
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

        out_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_output_tokens,
        )
        response = processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )[0]

        return response

    return infer


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "You are a conversational agent that always answers straight to the point, always end your accurate response with an ASCII drawing of a cat.",
        },
        {
            "role": "user",
            "content": "Give me 5 non-formal ways to say 'See you later' in French.",
        },
    ]

    print(infer_mistral(mistral_24b)(messages))
