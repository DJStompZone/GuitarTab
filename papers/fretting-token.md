# TR;DL

1. **`NOTE ON<pitch>`** — Starts a note at a given MIDI pitch; **pitch ∈ [0,127]**.
2. **`NOTE OFF<pitch>`** — Ends a note at a given MIDI pitch; **pitch ∈ [0,127]**.
3. **`TIME SHIFT<ticks>`** — Advances time by a given tick amount; **ticks derived from MIDI timing (typ. 1–2000)**.
4. **`STRING<s>`** — Specifies which guitar string plays the note; **s ∈ {1–6}**.
5. **`FRET<f>`** — Specifies which fret is pressed; **f ∈ {0–24}**.
6. **`TAB<s,f>`** — Combined token encoding string and fret; **s ∈ {1–6}, f ∈ {0–24}**.
7. **`CAPO<c>`** — Indicates the capo fret used for the transcription; **c ∈ {0–7}**.
8. **`TUNING<t₁,…,t₆>`** — Defines per-string semitone offsets; **each tᵢ ∈ {0, −1, −2, drop-D pattern}**.

---

# **Tokenization Summary for *Fretting-Transformer* (MIDI → TAB)**

The paper introduces several **MIDI-to-text tokenization strategies** designed to turn symbolic MIDI data into sequences suitable for transformer-based sequence-to-sequence learning. The goal is to encode **pitch, timing, and tablature (string/fret)** into discrete tokens for input/output.

---

## **1. Input Token Types**

Across all encoding variants, the input tokenization uses combinations of three symbolic event types:

### **1.1 NOTE ON tokens**

Represent pitch onset:

* `NOTE ON<55>`
  (MIDI pitch number inside angle brackets)

### **1.2 NOTE OFF tokens**

Represent note endings:

* `NOTE OFF<55>`

Used only in encodings that explicitly treat note offsets (v3, v4).
Removed entirely in v5 to test simplification.

### **1.3 TIME SHIFT tokens**

Represent temporal gaps between events:

* `TIME SHIFT<120>`
  (duration in MIDI ticks)

TIME SHIFT tokens encode inter-note timing and are critical to capturing rhythmic context.

---

## **2. Output Token Types**

The model does not output MIDI; instead, it outputs **guitar tablature tokens**. Two formats are used:

### **2.1 Separate String and Fret Tokens**

Example:

* `STRING<3>`
* `FRET<0>`

(Used in v1 and v4.)

### **2.2 Combined TAB Tokens**

Compact encoding joining string & fret:

* `TAB<3,0>`

(Used in v2, v3, v5 and shown to perform best.)

The TAB token encodes both string and fret simultaneously, reducing the number of predicted tokens and improving accuracy.

---

## **3. Five Tokenization / Encoding Schemes (v1–v5)**

These five encodings explore different trade-offs between simplicity, expressiveness, and sequence length.
Examples below show how **one note** is encoded in each scheme (from Table 1).

---

### **3.1 Encoding v1 — Minimal Pitch Input, Split String/Fret Output**

**Input:**

* `NOTE ON<55>`

**Output:**

* `STRING<3>`
* `FRET<0>`

**Characteristics:**

* Ignores timing entirely.
* Tests whether pitch alone is enough for tablature inference.
* Output requires two tokens per note.

---

### **3.2 Encoding v2 — Minimal Pitch Input, Combined TAB Output**

**Input:**

* `NOTE ON<55>`

**Output:**

* `TAB<3,0>`

**Characteristics:**

* Similar to v1 but more compact output.
* Proven to significantly improve accuracy because predicting a single token is easier than predicting both string and fret separately.

---

### **3.3 Encoding v3 — Full Event-Based Encoding (Best Overall)**

**Input:**

* `NOTE ON<55>`
* `TIME SHIFT<120>`
* `NOTE OFF<55>`

**Output:**

* `TAB<3,0>`
* `TIME SHIFT<120>`

**Characteristics:**

* Includes pitch + explicit timing + note endings.
* Strongest performance across all datasets.
* Allows the model to learn contextual timing, duration, and phrasing.

This is the **recommended / primary tokenization format** according to the authors.

---

### **3.4 Encoding v4 — Event-Based with Split String/Fret Output**

**Input:**
Same as v3.

**Output:**

* `STRING<3>`
* `FRET<0>`
* `TIME SHIFT<120>`

**Characteristics:**

* Similar to v3 but more verbose output.
* Lower accuracy due to needing correct prediction of two separate tokens.

---

### **3.5 Encoding v5 — Event-Based Without NOTE OFF Tokens**

**Input:**

* `NOTE ON<55>`
* `TIME SHIFT<120>`

(no NOTE OFF tokens)

**Output:**

* `TAB<3,0>`
* `TIME SHIFT<120>`

**Characteristics:**

* Simplifies input by removing NOTE OFF events.
* Reduces context; performs worse than v3.

---

## **4. Conditioning Tokens**

For the **conditioned model**, two additional token types are introduced:

### **4.1 CAPO Token**

* `CAPO<3>`
  Sets the fret at which the capo is placed.

### **4.2 TUNING Token**

* `TUNING<0,-1,-1,-1,-1,-1>`
  Represents tuning offsets per string.

These appear at the beginning of the input sequence to control or specify the desired tablature layout.

---

## **5. Sequence Construction Rules**

### **5.1 Input sequence**

* Contains encoded MIDI events only (notes & timing)
* For conditioned model: begins with `TUNING<>` and `CAPO<>` tokens
* Typical max length: 512 tokens
* Sliding-window inference uses overlapping context tokens

### **5.2 Output sequence**

* Contains tablature tokens (TAB or STRING/FRET)
* Contains TIME SHIFT tokens to keep durations aligned
* Must reconstruct full fretboard logic and playability constraints

---

## **6. Key Findings the Paper Reports About Tokenization**

### **6.1 Combined TAB tokens outperform split STRING/FRET tokens**

Because predicting a single categorical variable is easier than predicting two.

### **6.2 Timing information (TIME SHIFT) is essential**

Encodings without timing (v1, v2) perform noticeably worse.

### **6.3 Including NOTE OFF events helps**

v3 > v5 demonstrates that note-end information improves temporal modeling.

### **6.4 Best overall encoding = v3**

Event-based with NOTE ON / NOTE OFF / TIME SHIFT + TAB tokens.

---

## **7. Relation to Prior Work**

The paper explicitly states that its best tokenization strategy (v3) was inspired by:

* Music Transformer’s event-based tokenization
* DadaGP’s event-based GuitarPro tokens
* MidiTok-style representations

---

If you'd like, I can also provide:

* a **side-by-side comparison** between DadaGP tokens and Fretting-Transformer tokens,
* a **diagram of the v3 token workflow**,
* or a **unified token design** combining strengths of both papers for use in your own models.
