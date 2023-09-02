from transformers import AutoTokenizer

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, from_pt=True)
    return tokenizer

def tokenize_text(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=True)
    return tokenized_inputs