# What Makes a Model Metadata-Controllable

Metadata conditioning either works or quietly does nothing, and the difference
is rarely the architecture. This page collects what we learned building
FlowSat, so you can design a dataset, a caption regime and a metadata schema
that leave room for control rather than accidentally engineering it away.

## Controllability is a competition for residual variance

A generative model has to explain the variation in its training images. Every
conditioning signal competes to do that explaining, and the cheapest route
wins. Text is very cheap: it arrives through a large pretrained encoder and is
cross-attended at every block. Metadata is expensive: it arrives as a small
learned embedding through a narrow injection path.

So if a caption already states what a metadata field encodes, that field
becomes redundant during training. Gradient descent routes around it, the
metadata pathway is starved, and at inference the field appears inert. The
model is behaving correctly — there was simply nothing left for it to explain.

The practical formulation: **metadata controllability is inversely proportional
to how much your captions cover that metadata's own domain.** Note this is
about *overlap*, not *richness*. A 300-word caption describing rooftop
geometry, pavement texture and parking layout is dense but leaves latitude's
variance untouched. A 30-word caption saying "an arid Egyptian site in winter"
is short and destroys it. Length is a proxy; overlap is the mechanism.

## The dataset decides what is possible

Before blaming a model, check the data contains what you are asking for.

FMoW is a functional land-use dataset: 62 categories of human infrastructure,
every image centred on a built landmark. It has no desert, forest or tundra
class. Asking a FMoW-trained model to turn "Sahara" coordinates into sand
dunes is asking for an image that does not exist in its training distribution.
What *is* learnable is regional style within a category — roof colour and
material, building density, street and field geometry, ground tone.

Coverage matters as much as content. FMoW spans 207 countries but is heavily
weighted towards Western ones, and locations were deliberately thinned during
collection, so per-region density is low. A field can only be controllable
where the data supports it.

Two consequences for evaluation. Always include a **well-represented control**
in any sweep: if dense-coverage locations respond and sparse ones do not, that
is a coverage result, not a model failure, and you cannot tell the two apart
without both. And state the label space honestly — "the same category rendered
in a different regional style" is a defensible claim; "we generate any biome on
demand" is not.

## Designing the training caption regime

Use a mixture, not a single style. FlowSat trains on roughly:

| bucket | share | why |
|---|---|---|
| rich VLM caption | 40% | image quality and text fidelity |
| short template caption | 30% | keeps brief prompts in distribution |
| **empty caption** | **30%** | **the only regime where metadata must carry the load alone** |

The empty bucket is the one people omit. With no caption-free samples, text
explains nearly all the variance and the metadata pathway never has to do
anything. Below about ten percent, metadata conditioning tends to stay weak.
Our earlier 70/20/10 split gave only 10% caption-free samples and produced
visibly weaker metadata response than 40/30/30.

Two more rules. Make dropout **per sample, not per field**, and drop to a
*learned null embedding* rather than to zeros — after normalisation, zero is
not "unknown", it is a confident wrong assertion (longitude −180°, latitude
−90°, year 1980). Per-field zeroing at p=0.1 over seven fields corrupts about
half your samples and actively teaches the model that metadata lies. And keep
captions **disentangled**: if they never distinguish "white cloud" from "brown
haze" from "low contrast", the model collapses them into one concept and a
request for cloud cover returns a muddy image instead of clouds.

## Designing the inference prompt

The same competition operates at generation time. If you want a metadata field
to visibly steer the output, do not describe that field's domain in the prompt:

- **Short or empty prompts** give metadata the most room. "A stadium" is
  ambiguous about where; coordinates can resolve it.
- **Rich structural prompts** — dense about shapes, materials and layout,
  silent about place, climate and season — retain control while keeping
  quality. This is usually the best operating point.
- **Rich contextual prompts** that state terrain, climate or region will
  dominate the metadata, and should.

When demonstrating controllability, hold the caption and the noise seed fixed
and vary exactly one field. Any other protocol confounds the result.

## Adding a new metadata field: choose orthogonal features

The single best predictor of whether a new field will be controllable is the
oldest idea in feature selection: **is it independent of what you already
have?** A field earns its place only if its variance is not already explained
by another metadata field or by the captions.

**Ground sample distance is the worked example.** GSD is the most reliably
controllable field in FlowSat, and it required no special encoder design, no
tuning, no extra loss term. It works because it is close to perfectly
orthogonal to everything else in the conditioning:

- **No caption ever states it.** No VLM writes "0.5 m per pixel". Its variance
  is invisible to the text pathway, so text cannot explain it away.
- **It is uncorrelated with the other fields.** Resolution is a property of the
  sensor and the acquisition, not of latitude, month or cloud cover. Nothing
  else in the vector predicts it.
- **Its visual effect is monotonic and unambiguous.** Finer GSD means finer
  detail, everywhere, with no confounds.

Because nothing else could account for that variation, the model was forced to
use the GSD channel, and it learned it almost immediately.

Now contrast the fields that struggle. **Latitude and longitude** are heavily
correlated with country names, climate words and terrain descriptions that
captions routinely contain — so text explains them away. **Month** is
correlated with any season word in a caption, and its effect is confounded
with latitude (July is summer at 45°N and winter at 35°S), so month alone is
not even a well-defined visual quantity. **Cloud cover** is correlated with
haze, contrast and colour saturation, which captions describe constantly; if
the captions never separate "white cloud" from "brown haze", cloud cover
collapses into a global muddiness control.

So before adding a field, ask three questions:

1. **Do my captions describe it?** If yes, either remove it from the captions
   or accept that the field will be weak.
2. **Is it predictable from another field?** If yes, the model will use the
   cheaper one, and the new field adds parameters without adding control.
3. **Does it have a consistent visual signature on its own?** If its effect
   depends on another field (month depends on hemisphere), condition on the
   *derived* quantity — solar season rather than raw month — so the field means
   one thing.

A field that passes all three needs no special treatment. A field that fails
any of them needs the *data* fixed, not the architecture.

## Diagnosing weak controllability

In order, because each is cheaper than the next:

1. **Sweep with an empty prompt.** If the field responds with no text but not
   with a caption, you have an overlap problem, not a model problem.
2. **Check the injection gate magnitudes.** Compare the metadata projection
   weights against a trained backbone weight of similar role. Weights one to
   two orders of magnitude smaller mean the pathway is alive but too weak — a
   learning-rate problem, since a freshly initialised adapter cannot catch a
   pretrained backbone at the same LR.
3. **Separate tonal from structural response.** Measure colour difference and
   structural similarity across a sweep separately. Large colour change with no
   structural change means the global modulation path works but nothing is
   reaching spatial content — a real result, and a much narrower claim than
   "controllable".
4. **Only then change the encoder.** Verify any positional encoding on real
   inputs: check that nearby coordinates produce nearby embeddings and distant
   ones produce distant embeddings. A mis-specified frequency ladder can turn a
   geographic encoder into a hash of position, which trains without error and
   generalises to nothing.
