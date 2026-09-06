from pathlib import Path

path = Path("/kaggle/working/coconut/coconut.py")  # adjust if your clone puts it elsewhere
src = path.read_text()

old_1 = '''            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )'''

new_1 = '''            else:
                # Qwen3.5 patch: at batch_size=1 this truncation was always a
                # no-op (next_compute_range[0] == the cache's current length),
                # so pass kv_cache straight through instead of slicing it.
                # DeltaNet's conv_states/recurrent_states can't be sliced like
                # attention KV anyway -- see project notes.
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                )'''

old_2 = '''            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),'''

new_2 = '''            past_key_values=kv_cache,  # Qwen3.5 patch: same no-op removal as above'''

old_3 = '''    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []'''

new_3 = '''    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        assert input_ids.shape[0] == 1, (
            "Qwen3.5 patch only supports batch_size=1 -- the original truncation "
            "logic assumed multi-instance batches with misaligned latent counts; "
            "at batch_size=1 it was a no-op, so it's removed rather than made "
            "DeltaNet-compatible. Use gradient_accumulation_steps for throughput."
        )

        logits = []'''

for name, old, new in [("mid-loop truncation", old_1, new_1),
                        ("final-pass truncation", old_2, new_2),
                        ("batch_size assert", old_3, new_3)]:
    if old not in src:
        print(f"!!! FAILED to find exact match for: {name}. "
              f"Your coconut.py may differ slightly (whitespace/formatting) -- "
              f"paste back `grep -n 'past_key_values' coconut.py` output and I'll fix the match.")
    else:
        src = src.replace(old, new, 1)
        print(f"Applied: {name}")

path.write_text(src)
print("\nDone. Diff it with: !diff <(git show HEAD:coconut.py) coconut.py")
