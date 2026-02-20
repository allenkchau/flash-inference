"""
This script generates max_new_tokens tokens given a model and a starting prompt.

input_ids is the prompt.

I also measure the time for the prefill phase (process the prompt) and decode phase (each subsequent iteration).

This is an optimization on generate_baseline that uses the KVCache module.
"""

import torch
from src.runtime.inference import prefill, decode_step
from src.runtime.kv_cache import KVCache

def generate_kv(model, input_ids, max_new_tokens):

    B, T = input_ids.shape

    # first we want to create our kv cache
    kv_cache = KVCache(config=model.config, batch_size=B)

    # make sure model and input_ids on same device
    assert input_ids.device == next(model.parameters()).device      # model.parameters() is an iterator so next gets the first parameter tensor

    # create CUDA events with timing enabled
    start_prefill = torch.cuda.Event(enable_timing=True)
    end_prefill = torch.cuda.Event(enable_timing=True)

    start_decode = torch.cuda.Event(enable_timing=True)
    end_decode = torch.cuda.Event(enable_timing=True)

    prefill_time = 0
    decode_time_total = 0
    decode_times = []

    model.eval()
    with torch.no_grad():
        # time prefill
        start_prefill.record()
        
        # run the prefill forward pass
        logits = prefill(model, prompt_ids=input_ids, kv_cache=kv_cache)

        end_prefill.record()
        torch.cuda.synchronize()
        prefill_time = start_prefill.elapsed_time(end_prefill)

        next_id = torch.argmax(logits[:, -1, :], dim=-1).to(torch.long)  # (B,)
        next_id = next_id.unsqueeze(1)                                   # (B, 1)

        if input_ids.shape[1] >= model.max_seq_len:
            return input_ids, prefill_time, decode_time_total, decode_times

        input_ids = torch.cat([input_ids, next_id], dim=1)
        generated = 1
        decode_times.append(0.0)        # first token came from prefill

        assert kv_cache.cur_len == input_ids.shape[1]

        # decode remaining tokens using kv cache
        while generated < max_new_tokens and input_ids.shape[1] < model.max_seq_len:
            start_decode.record()
            # decode_step consumes the last generated token and produces logits for the next token
            last_logits = decode_step(model, next_id, kv_cache=kv_cache)  # (B, vocab)

            end_decode.record()
            torch.cuda.synchronize()
            new_time = start_decode.elapsed_time(end_decode)
            decode_time_total += new_time
            decode_times.append(new_time)

            next_id = torch.argmax(last_logits, dim=-1).to(torch.long)    # (B,)
            next_id = next_id.unsqueeze(1)                                # (B, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated += 1

    # return the generated tokens
    return input_ids, prefill_time, decode_time_total, decode_times
