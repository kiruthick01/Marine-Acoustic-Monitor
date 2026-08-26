# Datasets

What data this project needs, where each kind comes from, and which pipeline
stage / research claim it serves. Consistent with [DECISIONS.md](../DECISIONS.md)
and [ml-pipeline.md](ml-pipeline.md).

Two hard facts shape everything below:

1. **The core research claim is multimodal** (joint acoustic + environmental
   anomaly detection on-device). No public dataset provides co-located
   hydrophone audio + temperature/pH/turbidity/salinity *with anomaly ground
   truth*. That gap is what makes the project publishable — and it means the
   headline evaluation can only come from our own deployment data.
2. **Stage 2 needs no labels** (unsupervised, calibrated per site), but any
   *evaluation* of Stage 2, and all of Stage 3, needs labeled anomalies from
   somewhere: manual review, controlled playback, or public datasets.

## 1. Data we must generate ourselves (the paper's core dataset)

**1a. Site calibration baseline (unlabeled).** Feature vectors from an
initial post-deployment period assumed normal. Minimum one full diel cycle
(≥144 windows at the default 10-minute interval, per
[pi-implementation.md](pi-implementation.md)); for a paper, multiple weeks
spanning at least one weather cycle, ideally a seasonal boundary to
demonstrate recalibration (`edge/recalibrate.py`).

**1b. Reviewed anomaly labels.** Every Stage 2 flag gets manually reviewed
offline against the retrieved Tier 1 audio and labeled: true anomaly
(+ category: vessel / biological / weather / equipment / unknown) or false
alarm. This is simultaneously the paper's evaluation set and the seed for
Stage 3. Requires a written labeling protocol before deployment (who
reviews, what tool, category definitions, and — if more than one reviewer —
inter-annotator agreement reporting).

**1c. Controlled playback / injection experiments (recommended).** The
strongest evaluation design available to us: play known sounds (vessel
recording from a speaker, artificial tones) and/or induce environmental
steps (e.g. localized freshwater/turbidity plume near the sensors) at logged
times, then measure detection. Sethi et al. (PNAS 2020) used exactly this
playback-experiment design to validate unsupervised eco-acoustic anomaly
detection — precedent worth citing for the methodology.

**1d. Negative/confounder data.** Rain, wind chop, mooring clank, boat wake,
biofouling drift — captured naturally during deployment and labeled during
review. Needed to report false-alarm behavior honestly.

## 2. Public vessel-noise datasets (synthetic realism now; Stage 3 later)

| Dataset | What it is | Use here |
|---|---|---|
| ShipsEar (Santos-Domínguez et al., Applied Acoustics 2016) | ~90 recordings, 11 vessel classes + ambient, Spanish Atlantic coast; available on request | Validate/replace `synthetic_audio.py`'s vessel model; Stage 3 vessel classes |
| DeepShip (Irfan et al., Expert Systems with Applications 2021) | ~47 h, 4 vessel classes, Strait of Georgia | Larger-scale Stage 3 training/transfer |
| QiandaoEar22 (EURASIP J. Adv. Signal Processing 2024) | ~9.5 h ship-radiated noise + ~22 h background, lake environment | Additional vessel + background diversity |

Nearest-term use: check the synthetic vessel generator's spectral shape
(shaft/blade-rate tonals at 60–120 Hz, per [ml-pipeline.md](ml-pipeline.md))
against real recordings — cheap credibility for the simulation section.

## 3. Marine bioacoustic call libraries (synthetic realism; Stage 3 transfer)

- **Watkins Marine Mammal Sound Database (WHOI)** — decades of recordings
  across dozens of marine mammal species, free for research use. The
  standard source for injecting *real* biological calls into evaluation
  audio and for Stage 3 biological-call classes.
- **NOAA NCEI Passive Acoustic Data archive** — large federated PAM archive,
  open access.
- **DCLDE workshop datasets** — labeled marine mammal detection/classification
  challenge data; useful as an external benchmark for any Stage 3 claims.
- **Orcasound open data (AWS Open Data)** — Pi-recorded hydrophone streams +
  community-labeled orca calls. Doubly useful: same sensing platform class as
  ours (already cited in [related-work.md](related-work.md)), so it's both a
  data source and a realism check for Pi-grade audio quality.

## 4. Long-duration ambient soundscapes (baseline realism; conditioning tuning)

- **MBARI MARS cabled observatory (AWS Open Data, "pacific-sound")** —
  continuous deep-sea hydrophone recording since 2015. Gold standard for
  what a real marine ambient baseline + diel/seasonal variation looks like;
  use to sanity-check the synthetic ambient generator and to tune Stage 0
  conditioning on real noise before our hardware exists.
- **NOAA SanctSound** — multi-year soundscape monitoring across US national
  marine sanctuaries with standardized metrics; useful for what "normal
  variation" looks like across sites.

## 5. Co-located acoustic + environmental data (the multimodal gap)

The closest public approximations to our modality combination — none has
anomaly ground truth, but both provide real joint acoustic/environmental
time series to (a) validate the synthetic environmental generator's
correlation structure and (b) dry-run the joint feature vector + Stage 2 on
real data pre-deployment:

- **Ocean Observatories Initiative (OOI)** — cabled/moored arrays with
  broadband hydrophones co-located with CTD (temperature/salinity), pH, and
  turbidity-class sensors; open access via OOI data portal.
- **Ocean Networks Canada (ONC)** — cabled observatories (e.g. coastal
  community observatories) with hydrophones + water-quality instruments;
  open access via Oceans 3.0 portal.

A useful pre-deployment paper experiment: run our exact pipeline (Stage 0–2)
over an OOI/ONC segment containing a known event (documented storm, vessel
passage from AIS logs) and report whether the joint detector flags it. This
substitutes partially for field data if deployment slips.

## 6. Environmental-only time series (synthetic generator calibration)

- **NOAA NDBC buoy network** — free historical met/ocean time series;
  calibrate `synthetic_environmental.py`'s baselines, diel cycles, and storm
  transients against real buoy data.
- **IOOS ERDDAP servers** — programmatic access to US coastal water-quality
  time series (temperature, salinity, turbidity, pH where instrumented).

## 7. Pretrained audio models (Stage 3 transfer learning path)

Per [ml-pipeline.md](ml-pipeline.md) Stage 3's transfer-learning
precondition:

- **VGGish / YAMNet (AudioSet)** — generic audio embeddings, small enough
  that embedding extraction offline is trivial.
- **Google humpback-whale detector (TF Hub)** — marine-specific pretrained
  model; even off-target species, its embeddings are marine-audio-tuned.
- **Perch / SurfPerch (Google)** — bioacoustic embedding models; SurfPerch
  is specifically tuned for marine (coral reef) soundscapes and validated
  for transfer with unsupervised downstream methods.

## Priority order

1. **Now (no hardware):** MARS/Orcasound ambient + ShipsEar vessel spectra to
   validate the synthetic generators; NDBC for the environmental generator;
   OOI/ONC segment for a real-data dry run of the joint pipeline.
2. **At deployment:** calibration baseline (1a) + labeling protocol (1b)
   written *before* the device goes in the water.
3. **During deployment:** playback experiments (1c), review cycle (1b/1d).
4. **Stage 3 (later):** Watkins/DCLDE/DeepShip + pretrained embeddings, once
   reviewed labels exist.

## Verification status

Dataset names, sources, and access models above were checked against public
listings (August 2026); approximate sizes/hours are from memory of their
publications and should be re-verified against the source papers before
appearing in a manuscript. The "Ratilal et al. 2022" entry previously cited
in [related-work.md](related-work.md) could not be verified and is flagged
there.
