# Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription  
*Anna Hamberger, Sebastian Murgul, Jochen Schmidt, Michael Heizmann*  
*arXiv:2506.14223v1 — 17 Jun 2025*  
Source: :contentReference[oaicite:0]{index=0}

---

## Abstract
Music transcription is essential in Music Information Retrieval (MIR), especially for stringed instruments where MIDI lacks playability details. This work presents **Fretting-Transformer**, an encoder–decoder model using a T5 transformer to convert MIDI into guitar tablature by framing the task as symbolic translation. The system handles string-fret ambiguity and physical playability, using datasets DadaGP, GuitarToday, and Leduc with novel preprocessing and tokenization. Custom metrics evaluate tab accuracy and playability. Experiments show the model surpasses A* and commercial tools like Guitar Pro, with further improvements from context-sensitive processing and tuning/capo conditioning.

---

## 1. Introduction
Notation-level transcription converts musical audio or symbolic data into written form. Automatic Music Transcription (AMT) aims to algorithmically convert musical input into symbolic representations.

Standard notation lacks string and fret information crucial for guitarists, making tablature preferred. Translating symbolic notation (e.g., MIDI) into tablature is challenging due to multiple playable positions for the same pitch and physical constraints.

This work addresses transcription of MIDI to acoustic guitar tablature using a T5 transformer, treating the task as symbolic translation and resolving ambiguity while ensuring realistic playability.

---

## 2. Related Work

### Rule-based & Probabilistic Methods
Early systems used manually defined rules. Some minimized hand movement but were restrictive for advanced players.  
Genetic algorithms improved playability; later work integrated audio features.

HMM-based approaches mapped audio to likely fingerings.

### Graph-based Approaches
Sayegh introduced the optimum path paradigm. Later A*-based systems (e.g., A-star-Guitar) applied biomechanical weighting to graph traversal.

### Neural Network Approaches
CNN-based models (TabCNN) mapped spectrograms to tablature.  
Transformers improved symbolic sequence modeling (Music Transformer, Pop Music Transformer, etc).

DadaGP provided a large tokenized dataset.  
Transformer-XL variants generated tablature but struggled with low-pitch placement.

BERT-based MIDI-to-Tab approaches showed promise but lacked tuning/capo flexibility.

---

## 3. Methodology

### 3.1 Datasets
**GuitarToday:** 363 beginner-friendly fingerstyle tabs in standard tuning.

**DadaGP:** 26k+ tracks; 2,301 acoustic guitar tracks selected; high stylistic diversity.

**Leduc:** 232 jazz guitar tablatures with complex voicings.

### 3.2 Data Pre-Processing
Pipeline includes:

- Filter guitar tracks using MIDI channel IDs + keyword matching  
- Remove duplicates  
- Convert GuitarPro → MIDI  
- Extract (pitch, string, fret, timing)  
- Tokenize sequences  
- Split into training/validation/test  

Five encoding schemes explored.

### 3.3 Data Augmentation
Performed for tuning and capo imbalance:

- Capo augmentation: transpose standard-tuned pieces to capo 0–7  
- Tuning augmentation: apply standard, half-down, whole-down, drop-D

### 3.4 Model
Task framed as translation: MIDI tokens → TAB tokens.

Model features:

- Custom reduced T5 (d_model=128, d_ff=1024, 3 layers, 4 heads)  
- Adafactor optimizer  
- Event-based NOTE ON/OFF + TIME SHIFT tokens  
- Outputs combined TAB<string,fret> tokens  
- Conditioned model includes CAPO and TUNING tokens  

Training datasets:  
Standard: ~16k sequences; Conditioned: ~130k sequences.

### 3.5 Post-Processing
Matches predicted notes to input within ±5-note windows.  
If unmatched, assigns first valid string-fret combination.  
Ensures pitch correctness.

### 3.6 Evaluation Metrics
Three metrics:

1. **Pitch Accuracy** — correctness of pitch (0–100%).  
2. **Tab Accuracy** — match to ground-truth fingerings (0–100%).  
3. **Difficulty Score** — based on fret stretch, locality, and vertical movement.  
   Ranges from 0 (easy) to 18.5 (max difficulty).

---

## 4. Experiments and Results

### 4.1 Data Encodings
Five encoding variants tested:

- **v1:** pitch-only + STRING + FRET  
- **v2:** pitch + combined TAB token  
- **v3:** event-based NOTE ON/OFF + TIME SHIFT + TAB (**best performing**)  
- **v4:** event-based but separate string/fret tokens  
- **v5:** simplified event-based without NOTE OFF  

v3 generalizes best across datasets.

### 4.2 Effects of Post-Processing
| Method | Pitch Acc | Tab Acc |
|--------|-----------|---------|
| No post-processing | 97.23% | 68.56% |
| Overlap | 99.92% | 72.15% |
| Overlap + neighbor search | **100%** | **72.19%** |

Post-processing crucial for musical fidelity.

### 4.3 Domain Adaptation
Pre-trained **t5-small** converges faster but underperforms the custom reduced T5 when both use Adafactor.

### 4.4 Alternative NLP Task Formulations
Architectures tested:

- **T5** (translation)  
- **BERT** (fill-mask for TAB tokens)  
- **GPT-2** (text completion: "MIDI: ... TABS: ...")  

T5 performs best overall.

### 4.5 Conditioning on Tuning and Capo
Conditional models perform well:

- T5 best for GuitarToday & Leduc  
- GPT-2 slightly better for DadaGP  

Model can also *suggest* tunings by omitting condition tokens.

### 4.6 Comparison with Baselines
Methods compared: Baseline (lowest fret), A*, TuxGuitar, Guitar Pro, Fretting-Transformer.

Example tab accuracies:

| Dataset | Baseline | A* | Guitar Pro | Ours |
|---------|----------|----|------------|------|
| GuitarToday | 98.30% | 89.39% | 97.58% | **98.41%** |
| Leduc | 58.11% | 62.60% | 56.03% | **72.19%** |
| DadaGP | 79.21% | 78.92% | 76.58% | **81.58%** |

Fretting-Transformer is most accurate overall.

---

## 5. Conclusions
Fretting-Transformer effectively transcribes MIDI to tablature using a T5 encoder–decoder model. Contributions include:

- Novel tokenization and preprocessing  
- Conditioning on tuning and capo  
- Playability-aware evaluation metrics  
- Superior performance over A*, commercial tools, and prior transformer approaches  

Limitations include dataset quality and lack of expressive features.  
Future work: expand datasets, incorporate dynamics and voicing features, integrate with MIDI-to-score pipelines.

---

## 6. References
*(Reference list reproduced directly from the paper.)*

