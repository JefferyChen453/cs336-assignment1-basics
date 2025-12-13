import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.5,
    eos_token="<|endoftext|>",
    device="cuda"
):
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)  # (1, T)

    for _ in range(max_new_tokens):
        logits = model(x)  # (1, T, vocab_size)
        logits_last = logits[:, -1, :]  # (1, vocab)

        # temperature scaling
        if temperature != 1.0:
            logits_last = logits_last / temperature

        # convert to probabilities
        probs = F.softmax(logits_last, dim=-1)  # (1, vocab)

        # top-p sampling
        if top_p < 1.0:
            probs = top_p_filtering(probs, top_p)

        # sample next token
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
        next_token_id = next_token.item()

        # append to sequence
        x = torch.cat([x, next_token], dim=1)

        # stop if hit <|endoftext|>
        if tokenizer.decode([next_token_id]) == eos_token:
            break

    return x[0].tolist()


def top_p_filtering(probs, top_p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumulative > top_p).float().argmax().item()

    mask = torch.ones_like(sorted_probs, dtype=torch.bool, device=probs.device)
    mask[cutoff + 1:] = False
    filtered = sorted_probs * mask
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)

    original = torch.zeros_like(filtered, device=probs.device)
    original.scatter_(1, sorted_idx, filtered)

    return original
