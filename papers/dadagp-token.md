# TR;DL

1. **`artist:<name>`** — Identifies the song’s artist; **name = arbitrary string from metadata**.
2. **`downtune:<n>`** — Indicates global semitone downtuning applied to all strings; **n ∈ {0, −1, −2, …}**, limited to supported tunings (standard, Drop D, Drop AD, etc.).
3. **`tempo:<bpm>`** — Sets the initial tempo of the piece; **bpm typically ∈ [40–300]**.
4. **`start`** — Marks the beginning of the tokenized song; **no range**.
5. **`end`** — Marks the end of the tokenized song; **no range**.
6. **`instrument:note:string:fret`** — Encodes a pitched note with instrument type, string number, and fret; **string ∈ {1–7}, fret ∈ {0–24}** depending on instrument.
7. **`instrument:note:rest`** — Encodes a rest event for a specific instrument; **no numeric range**.
8. **`drums:note:<id>`** — Drum hit encoded via GP5 percussion map; **id = MIDI percussion number (e.g., 36, 38, 40)**.
9. **`wait:<ticks>`** — Represents time between events; **ticks = multiples of 960 ticks/quarter (e.g., 240, 480, 960...)**.
10. **`new_measure`** — Marks the start of a new measure; **no range**.
11. **`measure:repeat`** — Indicates a measure-level repeat symbol; **no range**.
12. **`nfx:<effect>`** — Encodes note-level effects such as palm-mute, tie, slide, hammer-on, pull-off, bend, vibrato, accents; **effect ∈ defined effect set** (e.g., `palm_mute`, `tie`, `hammer`, `slide`).
13. **`bfx:tempo_change`** — Signals a tempo change mid-performance; **no numeric range**.
14. **`instrument:<type>`** — Implicit via note tokens; instruments include up to 9 supported classes (distorted guitars, clean guitars, bass, drums, lead, pad); **type ∈ predefined instrument list**.

---

# **DadaGP Tokenization Summary**

(Extracted from *DadaGP: A Dataset of Tokenized GuitarPro Songs for Sequence Models*)


---

## **1. Purpose of the Tokenization**

DadaGP provides a text-based, event-style tokenization of GuitarPro tablature files so they can be used in sequence models (e.g., Transformers).
It adapts ideas from event-based MIDI encodings but preserves tablature details such as **string, fret, and guitar playing techniques**.

---

## **2. Global Song Structure**

Each tokenized song begins with four header tokens:

* `artist`
* `downtune`
* `tempo`
* `start`

Every song ends with an `end` token, which instructs the decoder to stop reconstructing the GuitarPro file.

The system is intentionally **syntax-tolerant**, meaning even randomly shuffled tokens will still decode into valid music.

---

## **3. Note Representation**

### **3.1 Pitched instruments (guitars, bass, etc.)**

Notes are encoded with instrument, note event, string, and fret:

**instrument:note:string:fret**

Example:
guitar1:note:5:7  → string 5, fret 7 on instrument “guitar1”.

### **3.2 Rests**

Encoded as:

**instrument:note:rest**

### **3.3 Drums**

Use GuitarPro 5 percussion MIDI numbers:

**drums:note:<MIDI_value>**

Example:
drums:note:36 → kick
drums:note:40 → snare

---

## **4. Timing Representation (wait tokens)**

DadaGP uses a single type of timing event:

**wait:<tick_amount>**

* Tick resolution is **960 ticks per quarter note**
* Example mappings:

  * Eighth note → wait:480
  * Sixteenth note → wait:240

Durations do **not** require note-off tokens; a new note replaces the previous one unless effects such as “let ring” are used.

---

## **5. Structural Tokens**

### **5.1 Measures**

* new_measure
* measure:repeat (used when GuitarPro includes repeated measures; may be removed in future versions due to difficulty for models)

### **5.2 Tempo**

* tempo (initial tempo)
* bfx:tempo_change (explicit mid-song changes)

The encoder can process multiple tempo changes throughout a piece.

---

## **6. Effects (Guitar Techniques)**

Effect tokens begin with **nfx** and apply to the **preceding note**.

Examples of supported techniques include:

* nfx:palm_mute
* nfx:bend
* nfx:vibrato
* nfx:slide
* nfx:hammer (covers hammer-ons and pull-offs)
* nfx:tie (ties two notes)
* nfx:accentuated_note
* nfx:heavy_accentuated_note

These allow the encoding to capture expressive guitar performance features normally absent in MIDI.

---

## **7. Instrument Handling**

The dataset standardizes instruments into nine possible tracks:

* Three distorted/overdriven guitars
* Two clean/acoustic guitars
* One bass
* One drumset
* One lead instrument (e.g., piano, synth pluck)
* One pad instrument (ambient textures)

Rare instruments are folded into lead/pad tracks.
Drum tracks are merged into a single unified drumset track.

Supported tunings include:

* Standard tunings
* 7-string guitars
* 5- and 6-string basses
* Common downtunings (Drop D, Drop A/D)
* Rare tunings are excluded

---

## **8. Meter and Time Signatures**

Time is inferred by summing wait tokens between new_measure markers.

Important notes:

* Tick resolution: 960 ticks per quarter note
* Time signatures are *not* emitted as tokens
* Distinguishing between similar meters (e.g., 3/4 vs 6/8) requires reading the original GuitarPro via PyGuitarPro rather than tokens alone

---

## **9. Missing or Unsupported Features**

The tokenization currently **does not encode**:

* Key signatures (usually absent in GuitarPro files; nearly all default to C major when inferred automatically)
* Note velocity (unlike MIDI)
* Traditional dynamics (p, mf, ff, etc.)
* Lyrics
* Clear handling of vocal melodies (often written as “lead” instruments)

---

## **10. Resilience and Design Philosophy**

* Random token sequences still decode into playable music → useful for creative generation experiments
* Syntax is minimal: each event is independent and self-contained
* Effects apply directly to the previous note, avoiding complex hierarchical structures
* Conversion is fully reversible using the provided encoder/decoder

---

If you'd like, I can also prepare:

* a **concise cheat-sheet version**,
* a **diagram or flowchart of the tokenization process**, or
* a **side-by-side example of GuitarPro → tokens → GuitarPro**.
