# Related Work

Status: This document positions the project against prior art, per [DECISIONS.md](../DECISIONS.md). Not claiming novelty on the base pattern (Pi-based passive acoustic monitoring) — the differentiation is in the specific combination of edge-computed multimodal correlation and hybrid tiered telemetry, detailed below.

## Prior art

**CORMA project (OGS Italy)**
Marine acoustic monitoring project (2018–2021) from the Istituto Nazionale di Oceanografia e di Geofisica Sperimentale — see "Underwater Noise Monitoring with Real-Time and Low-Cost Systems: The CORMA Experience" (J. Mar. Sci. Eng. 9(4):390, 2021). Low-cost, autonomous, solar-powered PAM stations transmitting over mobile networks — the closest prior art to this project's deployment shape. Key difference: CORMA transmits recorded data to a central server where processing happens; this project computes features and anomaly detection on-device and transmits only compact summaries, targeting sites without the bandwidth CORMA's cellular-streaming model assumes.

**Orcasound (open-source PAM on Raspberry Pi)**
Open-source hydrophone network using Raspberry Pi for real-time whale-call monitoring, with live audio streaming for community/researcher listening. Contributes the core precedent that a Raspberry Pi is viable as the edge compute unit for underwater PAM, and an open-source software model. Differs from this project in telemetry approach: Orcasound streams audio live over available bandwidth (e.g. shore-based internet), whereas this project assumes bandwidth-constrained deployment and never transmits raw audio, only computed summaries.

**University of Sao Paulo Pi-based recorder (PLOS ONE)**
Caldas-Morgan et al., "An Autonomous Underwater Recorder Based on a Single Board Computer," PLOS ONE 10(6), 2015 (doi:10.1371/journal.pone.0130297). Peer-reviewed low-cost Raspberry Pi-based underwater audio recorder, field-deployed off the Brazilian coast for cetacean and vessel-traffic monitoring; notably used a USB audio codec for hydrophone digitization. Contributes validation of Pi + hydrophone as a viable, published, low-cost recording platform. Primarily a recording device — contributes to hardware feasibility, not to the on-device processing or telemetry model used here.

**Low-cost DIY hydrophone designs (CoPiDi, etc.)**
Open hardware hydrophone designs oriented at accessibility and low cost. Contributes hydrophone-level hardware precedent relevant to [hardware-spec.md](hardware-spec.md), independent of the compute/software architecture.

**Sethi et al., unsupervised eco-acoustic anomaly detection**
Sethi et al., "Characterizing soundscapes across diverse ecosystems using a universal acoustic feature set," PNAS 117(29), 2020 (preprint: bioRxiv 865980, 2019). Embeds soundscapes into a learned acoustic feature space and detects anomalous sounds via unsupervised density estimation (GMM) against the site's typical soundscape — no labels required. Directly supports the Stage 2 approach in [ml-pipeline.md](ml-pipeline.md): unsupervised detection against a learned site baseline is a validated technique for eco-acoustic anomaly detection, reinforcing why this project can start there rather than with supervised classification.

**Multimodal underwater benchmark dataset (Ratilal et al. 2022)**
⚠️ CITATION UNVERIFIED — a 2022 multimodal underwater benchmark dataset attributable to Ratilal et al. could not be located (Purnima Ratilal's group at Northeastern works on large-aperture coherent hydrophone arrays / POAWRS, but no matching 2022 dataset publication was found). Before citing in any paper, either locate the intended publication or replace this entry with a verifiable co-located acoustic+environmental data source (e.g. Ocean Observatories Initiative hydrophone + CTD/pH nodes, Ocean Networks Canada, or NOAA SanctSound).

## How this project differs

None of the cited prior art is being displaced — the base pattern (Pi + hydrophone + on-device processing for underwater PAM) is well established, as the list above shows. This project's specific contribution is the combination of two things, neither of which is centrally addressed by the works above:

1. **Edge-computed multimodal correlation.** Acoustic features (MFCCs, spectral centroid, ZCR, RMS energy, spectral flatness) and environmental sensor features (temperature, pH, turbidity, salinity, and their rates-of-change) are extracted and concatenated into one joint feature vector on-device, then run through anomaly detection together, rather than treating acoustic and environmental data as separate streams analyzed independently or only correlated later in post-processing. See [ml-pipeline.md](ml-pipeline.md).

2. **Hybrid tiered telemetry.** Full-resolution data (raw audio, sensor logs) is deliberately kept local rather than streamed (contrast with Orcasound's live-streaming model), while a low-bandwidth link (LoRa or low-bandwidth cellular) continuously carries only compact computed summaries — feature summaries, environmental readings, anomaly alerts — on a duty cycle. This targets deployments where continuous high-bandwidth connectivity (shore-based internet) isn't assumed. See [architecture.md](architecture.md) and [data-pipeline.md](data-pipeline.md).

The project is, in effect, an application and integration of established PAM and unsupervised eco-acoustic anomaly detection techniques to a specific edge/telemetry-constrained, multimodal deployment shape — not a claim of new base sensing or algorithmic technique.
