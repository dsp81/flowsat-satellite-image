# Captioning a Dataset

FlowSat trains on **(image, caption, metadata)** triples. If your imagery has no
captions, generate them with a vision–language model. `tools/caption_dataset.py`
does this at corpus scale: it walks your directory tree, captions every image,
and writes a parallel tree of `.txt` files that the dataset loader picks up
automatically.

Captions matter more than people expect. Their *style* determines how the model
behaves at inference, and their *content* determines how much control your
metadata retains. Both are discussed below.

---

## 1. Quick start

```bash
python tools/caption_dataset.py \
    --image_root  /path/to/images \
    --output_dir  /path/to/captions \
    --model gemma4-e4b \
    --batch_size 4 --max_new_tokens 400
```

Output mirrors the input tree, so the loader can map an image to its caption by
path:

```
images/airport/airport_0/airport_0_0_rgb.tif
captions/airport/airport_0/airport_0_0_rgb.txt        ← the caption
captions/airport/airport_0/airport_0_0_rgb.meta.json  ← model, prompt, timing
```

**Resume is automatic.** Re-running the same command skips every image that
already has a non-empty caption; every write is atomic (temp file + rename), so
interrupting with Ctrl-C can never leave a partial caption behind.

### Multiple GPUs

Shard by index. Each process handles `i % num_shards == shard_id`, so shards
never touch the same file and can run concurrently, resume independently, and
together cover the corpus exactly once.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/caption_dataset.py ... --shard_id 0 --num_shards 2 &
CUDA_VISIBLE_DEVICES=1 python tools/caption_dataset.py ... --shard_id 1 --num_shards 2 &
```

`--shard_id` indexes the *work partition*, not the GPU. With two shards the ids
are always `0` and `1`, whichever GPUs you place them on.

### Verify coverage when it finishes

Do not trust the "DONE" line — count:

```bash
find $IMAGES  -name '*.tif' | wc -l          # source images
find $CAPTIONS -name '*.txt' | wc -l         # captions written
cat $CAPTIONS/FAILED.shard*.txt | wc -l      # failures
```

Captions + failures should equal the image count. If they do not, re-run the
same command; resume retries only what is missing.

---

## 2. Using a different VLM

`--model` accepts a Gemma alias or any HuggingFace image-text-to-text model id:

```bash
--model gemma4-e4b                       # default
--model gemma4-e2b                       # smaller, faster
--model Qwen/Qwen2.5-VL-7B-Instruct      # any HF image-text-to-text model
```

To add a backend, implement one method:

```python
class MyCaptioner(BaseCaptioner):
    def caption_batch(self, images, prompts, max_new_tokens) -> list[str]:
        """images: list[PIL.Image], prompts: list[str] -> list[str], same order."""
        ...
```

and register it in `build_captioner()`. Everything else — sharding, resume,
atomic writes, OOM retry — is backend-agnostic.

Practical notes: gated models (Gemma among them) need `huggingface-cli login`
and license acceptance; the script falls back through
`Gemma4ForConditionalGeneration` → `AutoModelForImageTextToText` →
`AutoModelForCausalLM`, so a missing class is not fatal; and recent VLM image
processors import `torchvision`, which must match your torch build exactly
(`torch 2.4.1` pairs with `torchvision 0.19.1`).

---

## 3. The prompt determines what your model can control

The captioning prompt is not a formatting detail. It decides which visual
properties end up described in text, and **anything text describes is something
metadata no longer needs to explain.**

If captions state the climate, season and region, then latitude and month are
redundant during training. The model learns to read that information from the
text pathway — which is pretrained, high-dimensional and cross-attended at every
block — and the metadata pathway is starved of gradient. At inference, varying
lat/lon then appears to do very little. That is the model behaving correctly
given what it was taught, not a bug.

So decide deliberately:

| goal | caption should |
|---|---|
| maximum image quality and text fidelity | describe everything, context included |
| maximum **metadata** controllability | describe *content* (structures, materials, layout, colours) and stay silent on *context* (country, season, month, climate) |

The default prompt in the script takes the first path — dense description
including climate cues. If metadata control is your priority, remove the climate
and season clauses from `CAPTION_INSTRUCTION` and do not pass country or date in
the context block.

### Two rules worth keeping whatever you choose

**Disentangle appearance from cause.** Satellite imagery is full of correlated
confounds. If captions never distinguish "white cloud" from "brown haze" from
"low contrast", the model collapses them into one concept and a request for
cloud cover produces a muddy image rather than clouds. Name them separately:

```
- If white or bright grey cloud formations are visible, describe them
  explicitly as clouds and say where ("scattered white clouds across the
  upper right"). Clouds are bright, opaque, soft-edged objects.
- If the ground looks brown, tan, dusty or low-contrast but is NOT covered
  by white cloud, describe it as the terrain or atmosphere it is ("dry brown
  farmland", "muddy brown water", "a faint brown haze") and do NOT call it
  cloudy.
```

**Keep the structure consistent.** Same attribute order in every caption
(palette → terrain → structures → layout → notable objects). Consistent captions
train better than free-form ones because the model learns stable slots.

---

## 4. Length: match it to your token budget

Caption length must fit `--t5_max_length` at training time, or the tail is
silently discarded. A 200-word caption is roughly 260 tokens; at the default
limit of 120, more than half of it never reaches the model — and it is the
*second* half, where spatial composition and object detail usually live.

Before training, print the token-length histogram:

```python
lens = [len(tok(open(p).read()).input_ids) for p in captions[:1000]]
print(f"median {sorted(lens)[500]}, p90 {sorted(lens)[900]}, max {max(lens)}")
```

Set `--t5_max_length` above the 90th percentile, or shorten the captioning
prompt. Also give the VLM enough generation budget: `--max_new_tokens 400` for a
120–200 word target, since a truncated caption ends mid-sentence and reads as
broken text.

---

## 5. Mixing caption styles during training

Training on rich captions alone makes short prompts behave badly at inference,
because they sample an under-represented mode. FlowSat therefore trains on a
mixture, controlled by `--rich_frac` and `--short_frac`:

| bucket | share | purpose |
|---|---|---|
| rich VLM caption | 40% | image quality, text fidelity |
| short template caption | 30% | keeps brief prompts in distribution |
| empty caption | 30% | forces the metadata pathway to carry information alone |

The empty bucket is the one people omit and then wonder why metadata does
nothing. With no caption-free samples, text explains nearly all the variance and
metadata never has to. Thirty percent is a reasonable default; below about ten
percent, metadata conditioning tends to stay weak.

---

## 6. Sanity-check the captions before training

Read ten of them. Specifically check that:

- the caption describes *this* image, not a generic satellite scene;
- cloudy images say "white cloud" and hazy ones do not;
- length matches your token budget;
- captions end with a complete sentence (otherwise raise `--max_new_tokens`);
- the style matches how you intend to prompt at inference.

Ten minutes here is cheaper than discovering a systematic caption flaw after a
multi-day training run.
