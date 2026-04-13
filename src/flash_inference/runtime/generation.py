"""
Basic decoding loop
"""

import torch
from flash_inference.configs.model_config import ModelConfig
from flash_inference.model.activations import Activation
from flash_inference.model.transformer import Transformer


# single decode step for each seq in batch
def decode_step(model: Transformer, input_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # run the model
    logits = model(input_ids)

    # get the logits from the last position
    # remember logits have shape: batch_size, seq_len, vocab_size
    last_logits = logits[:, -1, :]
    
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # if temperature is very close to 0, fallback to greedy decoding for stability
    if temperature < 1e-5:
        next_tokens = torch.argmax(last_logits, dim=-1, keepdim=True)
    else:
        scaled_logits = last_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

    return next_tokens


# run multiple decode steps
def generate(
    model: Transformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int = None,
    temperature: float = 1.0,
) -> torch.Tensor:

    # switch model from training mode to inference mode; layers like dropout and batch norm turned off
    model.eval()

    # guard against exceeding the max seq len
    model_max_seq_len = model.config.max_seq_len
    batch_size, seq_len = input_ids.shape

    if seq_len > model_max_seq_len:
        raise ValueError(f"seq_len ({seq_len}) exceeds max_seq_len ({model_max_seq_len})")

    remaining = model_max_seq_len - seq_len
    num_steps = min(remaining, max_new_tokens)

    # keep track of different sequences in the batch that may finish at different times
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    # we don't want PyTorch to build a computation graph for us
    # inference mode is same thing as no grad but faster; doesn't keep track of version counts and view tracking
    with torch.inference_mode():
        generated = 0
        while generated < num_steps:
            next_tokens = decode_step(model, input_ids, temperature=temperature)
            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                next_tokens[finished] = eos_token_id

            # append the chosen token to the sequence
            input_ids = torch.cat((input_ids, next_tokens), dim=1)
            generated += 1

            if eos_token_id is not None and finished.all():
                break

    return input_ids


def main():

    # build model and config
    config = ModelConfig(
        num_layers=2,
        vocab_size=128,
        num_heads=4,
        model_dim=16,
        max_seq_len=500,
        bias=True,
        mlp_activation=Activation.GELU,
        device=torch.device("cpu"),
        dtype=torch.float32,
        weight_tying=True,
    )
    model = Transformer(config)

    # dummy data
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(4, 8),
        dtype=torch.long,
        device=config.device,
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Input ids: {input_ids}")

    # generation
    output_ids = generate(model=model, input_ids=input_ids, max_new_tokens=500, temperature=1.0)

    print(f"Output shape: {output_ids.shape}")
    #print(f"Output ids: {output_ids}")


if __name__ == "__main__":
    main()
