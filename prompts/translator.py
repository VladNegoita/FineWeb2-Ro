def translate_prompt(text: str):
    return [
        {
            "role": "system",
            "content": "You are a highly accurate translation assistant. Your mission is to translate the following text from English to Romanian. Please only provide the translation.",
        },
        {"role": "user", "content": "Here is the text:\n" + text},
    ]
