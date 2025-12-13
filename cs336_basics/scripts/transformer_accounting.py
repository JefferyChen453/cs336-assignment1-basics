


def count_parameters(model_conf):
    vocab_size = model_conf["vocab_size"]
    context_length = model_conf["context_length"]
    num_layers = model_conf["num_layers"]
    d_model = model_conf["d_model"]
    num_heads = model_conf["num_heads"]
    d_ff = model_conf["d_ff"]

    # Embedding
    emb_param_count = vocab_size * d_model

    # Attention block: QKVO + RMSNorm
    attn_param_count = 4 * (d_model * d_model) + d_model
    attn_param_count *= num_layers

    # Feedforward block: SwigLU + RMSNorm
    ff_param_count = 3 * (d_model * d_ff) + d_model
    ff_param_count *= num_layers

    # LMHead: RMSNorm + Linear
    lm_head_param_count = d_model + d_model * vocab_size

    # Total
    total_param_count = emb_param_count + num_layers * (attn_param_count + ff_param_count) + lm_head_param_count

    print(f"Total params: {total_param_count}")
    print(f"Embedding Params: {emb_param_count} ({emb_param_count / total_param_count:.2% })")
    print(f"Attention Params: {attn_param_count} ({attn_param_count / total_param_count:.2% })")
    print(f"Attention Params: {attn_param_count} ({attn_param_count / total_param_count:.2% })")

    return total_param_count

def main():
    GPT_2_XL_CONF = {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 6400
    }
    count_parameters(GPT_2_XL_CONF)


if __name__ == "__main__":
    main()