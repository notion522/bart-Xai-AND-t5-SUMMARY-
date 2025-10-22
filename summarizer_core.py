from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from captum.attr import IntegratedGradients
import torch

# Cache models to avoid reloading
_models_cache = {}

def get_summarizer(model_name):
    if model_name not in _models_cache:
        _models_cache[model_name] = pipeline("summarization", model=model_name, device=-1)
    return _models_cache[model_name]

def summarize_text(text, model_name):
    summarizer = get_summarizer(model_name)
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]


def compute_bart_attributions(text):
    """Compute token-level attributions for BART only"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    # Tokenize input
    inputs = tokenizer([text], return_tensors="pt", max_length=512, truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)

    def forward_encoder(embeds):
        encoder_outputs = model.model.encoder(inputs_embeds=embeds, attention_mask=attention_mask)
        pooled = encoder_outputs.last_hidden_state.mean(dim=-1).mean(dim=-1)
        return pooled

    ig = IntegratedGradients(forward_encoder)
    attributions, _ = ig.attribute(embeddings, return_convergence_delta=True)

    attributions = attributions[0].sum(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    clean_pairs = [(t, a) for t, a in zip(tokens, attributions) if t not in tokenizer.all_special_tokens]
    clean_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return clean_pairs[:]
