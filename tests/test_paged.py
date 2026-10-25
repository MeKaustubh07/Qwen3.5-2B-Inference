import sys
import torch

sys.path.insert(0, "src")
from config import Qwen35Config
from models.qwen3_5 import Qwen35Model
from state import BlockAllocator, OutOfBlocks
from tokenizer import Tokenizer
from weight_loader import SafetensorsFile

D = "models/qwen3.5-0.8b"
cfg = Qwen35Config.from_json(f"{D}/config.json")
model = Qwen35Model(cfg, SafetensorsFile(f"{D}/model.safetensors-00001-of-00001.safetensors"))   # fp32 CPU
tok = Tokenizer(f"{D}/tokenizer.json")
TEXTS = ["The capital of France is", "Hello world", "def fibonacci(n):", "The quick brown fox jumps over the lazy dog.",
         "नमस्ते दुनिया, यह एक परीक्षण है।"]                          # 5, 2, 4, 10 and 14 tokens
PROMPTS = [torch.tensor(tok.encode(t)) for t in TEXTS]

results = []
def check(name, ok):
    results.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}")

# 1. allocator
a = BlockAllocator(4)
got = [a.allocate() for _ in range(4)]
check("allocator hands out 4 distinct blocks", sorted(got) == [0, 1, 2, 3] and a.num_free == 0)
try:
    a.allocate(); check("allocator raises when empty", False)
except OutOfBlocks:
    check("allocator raises OutOfBlocks when empty", True)
a.free(got[:2])
check("freed blocks are reusable", a.num_free == 2 and {a.allocate(), a.allocate()} == set(got[:2]))

def run(ids, state, split):
    """prefill `split` tokens, then decode the rest one at a time; return all logits."""
    parts = [model.forward(ids[:split], state=state)]
    for t in range(split, len(ids)):
        parts.append(model.forward(ids[t:t + 1], state=state))
    return torch.cat(parts)

# 2. paged (block_size 4, so every prompt spans several blocks with ragged ends) == contiguous.
#    Scramble the free list first so block tables come out NON-monotonic, like a pool that has recycled blocks.
pool = model.new_paged_pool(64, 4, max_seqs=4)
taken = [pool.allocator.allocate() for _ in range(10)]
pool.allocator.free([taken[i] for i in (7, 2, 9, 0, 5, 3, 8, 1, 6, 4)])
for i, ids in enumerate(PROMPTS):
    split = max(1, len(ids) // 2)
    ref = run(ids, model.new_state(len(ids)), split)
    seq = model.new_paged_state(pool)
    paged = run(ids, seq, split)
    table = seq.kv.block_table
    d = (paged - ref).abs().max().item()
    blocks_ok = len(table) == -(-len(ids) // 4)
    check(f"prompt {i} ({len(ids)} tokens, table {table}): paged == contiguous, max|diff|={d:.1e}, "
          f"exactly ceil(len/4) blocks", d < 1e-4 and blocks_ok and (len(table) < 2 or table != sorted(table)))
    seq.free()
check("all blocks and state slots returned to the pool after freeing", pool.allocator.num_free == 64 and len(pool.free_seqs) == 4)

# 2b. running out of blocks is all-or-nothing: the failed request holds nothing and has touched no DeltaNet state
#     (its own slot or a neighbour's in the shared pool tensor), and the others keep working
small = model.new_paged_pool(4, 4, max_seqs=2)
sa0 = model.new_paged_state(small); model.forward(PROMPTS[2], state=sa0)                  # 4 tokens -> 1 block
S_a0 = sa0.S.clone()
sb0 = model.new_paged_state(small)
try:
    model.forward(PROMPTS[3].repeat(2), state=sb0); failed = False                         # 20 tokens -> 5 blocks
except OutOfBlocks:
    failed = True
check(f"OutOfBlocks raised before taking anything (failed request holds {len(sb0.kv.block_table)} blocks, "
      f"{small.allocator.num_free} still free); its DeltaNet slot is still zero and its neighbour's unchanged",
      failed and sb0.kv.block_table == [] and small.allocator.num_free == 3 and sb0.length == 0
      and sb0.S.abs().sum() == 0 and sb0.conv_tail.abs().sum() == 0 and torch.equal(sa0.S, S_a0))
model.forward(torch.tensor([13]), state=sa0)
check("the other sequence can still decode after the failure", sa0.length == 5)

# 2c. running out of DeltaNet state slots: a third sequence is refused; freeing one hands its slot over, zeroed
try:
    model.new_paged_state(small); refused = False
except OutOfBlocks:
    refused = True
slot = sa0.seq
sa0.free()                                                   # the slot with history, so "zeroed" means something
sc0 = model.new_paged_state(small)
check(f"a third state on a 2-slot pool raises OutOfBlocks; after a free the new state reuses slot {slot}, zeroed",
      refused and sc0.seq == slot and sc0.S.abs().sum() == 0 and sc0.conv_tail.abs().sum() == 0 and sc0.length == 0)
sb0.free(); sc0.free()

# 3. two sequences interleaved in one pool don't disturb each other (KV blocks or DeltaNet slots)
ids_a, ids_b = PROMPTS[3], PROMPTS[4]
alone_a = run(ids_a, model.new_state(len(ids_a)), 2)
alone_b = run(ids_b, model.new_state(len(ids_b)), 2)
sa, sb = model.new_paged_state(pool), model.new_paged_state(pool)
out_a = [model.forward(ids_a[:2], state=sa)]; out_b = [model.forward(ids_b[:2], state=sb)]
for t in range(2, max(len(ids_a), len(ids_b))):          # alternate one decode step each
    if t < len(ids_a): out_a.append(model.forward(ids_a[t:t + 1], state=sa))
    if t < len(ids_b): out_b.append(model.forward(ids_b[t:t + 1], state=sb))
da = (torch.cat(out_a) - alone_a).abs().max().item(); db = (torch.cat(out_b) - alone_b).abs().max().item()
overlap = set(sa.kv.block_table) & set(sb.kv.block_table)
check(f"interleaved sequences match their solo runs (diff {da:.1e}, {db:.1e}), no shared blocks, slots {sa.seq} and {sb.seq}",
      da < 1e-4 and db < 1e-4 and not overlap and sa.seq != sb.seq)

# 4. memory: paged KV uses ceil(len/bs) blocks; the DeltaNet state is a fixed size per sequence
per_tok = 2 * len(model.attn_layers) * cfg.num_key_value_heads * cfg.head_dim * 4
fixed = len(model.linear_layers) * (cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
                                    + (cfg.linear_conv_kernel_dim - 1) * model.conv_dim) * 4
check(f"paged bytes for {sa.length} tokens = {sa.bytes_used()} (= {len(sa.kv.block_table)} blocks x 4 tokens x {per_tok} B "
      f"+ {fixed} B DeltaNet state)", sa.bytes_used() == len(sa.kv.block_table) * 4 * per_tok + fixed)
sa.free(); sb.free()

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
