# Transformers for Music — Exercise Plan (≈20 h block)

*Deep Learning for Music Technology · 1-week course · Transformer block*
*Draft 3c · 22 Nov 2026 · E2–E3 switched from folk (IrishMAN) to jazz (Weimar Jazz Database); E2 notebook built and tested, then improved for musical quality (12-key augmentation, model-friendly ABC, temperature 0.9, backing band at playback, solos over real chord changes; data and playback checked end to end)*

---

## 0. Brief (from our Q&A)

| Decision | Choice |
|---|---|
| Students | Music technology, **basic Python**: they read and tweak code but don't write models from scratch |
| Existing material | DUDL notebooks (PyTorch, Mike X Cohen): ANN, AE, RNN (incl. `DUDL_RNN_loremipsum`), transfer learning, GANs… |
| Data | **Mostly MIDI/symbolic**, plus one audio exercise |
| Musical focus | **Jazz** for the from-scratch GPT and conditioning (E2–E3): solos from the Weimar Jazz Database, converted to ABC with their chord changes |
| Time | **~20 h for Transformers**: ~10 h in class + ~10 h at home |
| Outcome | **Mini-project in pairs + listening demo** |
| Platform | Free Google Colab (T4 GPU, 16 GB, sessions can disconnect) |

**Main idea of the block:** *"Music as a language"*. Students first train a tiny GPT from scratch on text-like music (jazz solos in ABC notation), then learn that **conditioning is just what you put in the context**, then use a **pretrained** MIDI transformer and fine-tune it, and finally see that **audio is tokens too**. Each exercise reuses the previous one, so they change a few cells at a time and don't start over.

---

## 1. The block at a glance

| # | Session (class) | Class | Home | Transformer concepts | Music task | Data | GPU time |
|---|---|---|---|---|---|---|---|
| **E1** | Music as tokens | 1.5 h | 1 h | tokens, vocabulary, embeddings, sequence length, next-token probabilities | Tokenize MIDI and ABC, decode back, hear what is lost | your own MIDI files, a few MAESTRO/POP909 files | none |
| **E2** | Your first music GPT | 2.5 h | 1.5 h | self-attention, causal mask, positional embeddings, loss/perplexity, sampling (temperature, top-k), context length, LSTM vs Transformer | Train a mini-GPT that writes jazz choruses over chord changes | Weimar Jazz Database → ABC (1,257 choruses × 12 keys, built inside the notebook) | ≈14 min on an M1 Pro; T4 to check |
| **E3** | Conditioning | 2 h | 1.5 h | prompts and control tokens, prefix conditioning, first fine-tune, out-of-distribution controls, (cross-attention: demo) | Choruses in a chosen key/style/soloist; a solo over **your** chord changes; reverse it into a harmonizer | E2 model + the same jazz ABC (the chords are already in it) | ≈10 min |
| **E4** | Pretrained models & fine-tuning | 2 h | (project) | pretraining vs fine-tuning, transfer learning, (LoRA), infilling | Co-compose with the Anticipatory Music Transformer: accompaniment, infilling, then fine-tune on a small style corpus | Lakh-pretrained AMT + small corpus | ≈15–20 min |
| **E5** | Audio is tokens too | 1 h | (project) | neural audio codec tokens, multi-codebook sequences, text conditioning via cross-attention | EnCodec round-trip; MusicGen text + melody conditioning using the tune from E2/E3 | own audio clips | inference only |
| **P** | Project demos + listening test | 1 h | **6 h** | evaluation | Pairs present their conditioned generator | their choice | ≤ 20 min per run |
| | **Total** | **10 h** | **10 h** | | | | |

Suggested placement in the week: ANN/AE/RNN first (days 1–2). E1–E2 on day 3, E3–E4 on day 4, E5 + demos on day 5. The project starts at home after E3.

---

## 2. Design principles

1. **Listen in every notebook.** Every notebook ends with something to listen to: a piano-roll player, an ABC sheet with a play button (abcjs, with swing and a backing band), or audio. Learning curves are also *heard* (checkpoints at several training steps).
2. **Basic Python is enough.** Students run code, change named parameters (`temperature = 0.9`), type prompts (`C:John Coltrane (ts)`, `K:Eb`, a first bar with a chord), and answer questions. The model code is given and commented. Optional *"Code challenges"* (same name as in DUDL) are for faster students.
3. **Free Colab is the limit.** No training run longer than ~20 min. Checkpoints go to Google Drive. For each exercise, the teacher keeps a **pre-trained fallback checkpoint** so a student whose run fails can still do the exercise.
4. **Measure the control.** Each conditioning exercise has an automatic check (Is it in the requested key? Do the solo notes fit the chord?) besides listening. This becomes the core of the project evaluation.
5. **Build on DUDL.** E2 is the same as `DUDL_RNN_loremipsum` (char lookup tables, `CrossEntropyLoss`, Adam), but the `nn.LSTM` is replaced by attention. E4 extends the `transferlearning` notebooks to music.
6. **Honesty about generation.** A memorization check (is it copying the training set?) and "3 cherry-picked + 3 random samples" in every report.

---

## 3. What already exists online (research summary)

### 3.1 Courses with Transformer-for-music exercises

| Resource | What it has | What to borrow | Fit for your students |
|---|---|---|---|
| **UPF – Generative Algorithms for Sound and Music** (SMC master, 2025–26, V. Velardo & L. Wyse) | Symbolic part: "Melody generation with Transformers" code walkthrough; assignment **"Polyphonic Transformer for Bach Chorales"** (free representation choice); extra Hugging Face tutorial on fine-tuning a pretrained MIDI GPT-2 with **PEFT/LoRA**. Audio part: EnCodec-latent assignments, transformer on EnCodec tokens as paper topic. | The structure (symbolic → audio), the Bach chorale task as a project option, LoRA fine-tuning idea | Close to us and the right topics, but made for master's students who code well. Assignments are open-ended. |
| **The Sound of AI – Generative Music AI course** (Velardo, GitHub + YouTube) | Lectures "Transformers I/II" (slides) and **"19. Melody generation with transformers"** (TensorFlow code, ~1 h video) | Video as **home preparation** before E2 (optional) | Good to watch; the code is TF/Keras, not the PyTorch your students know |
| **UCSD CSE 190 – ML for Music** (S. Dubnov) | **Assignment 10: "Transformer (GPT) for Music Generation"**, in both PyTorch and Keras notebooks; earlier RNN MIDI assignment | Compare with your E2; their RNN → GPT order matches DUDL | Assignment-style notebooks; a useful template for teachers |
| **UCSD CSE 253** (J. McAuley) | Module on symbolic generation: representations (MIDI, REMI), n-gram → RNN → Transformer language models | Slides for the theory part of E1–E2 | Theory, no notebooks found |
| **NTU – Deep Learning for Music Analysis & Generation** (Yi-Hsuan Yang, 2025) | Weeks on Transformer music generation, codecs and audio LMs, symbolic MIDI generation; slides on GitHub (`affige/DeepMIR`) | Slides by the REMI / Pop Music Transformer group | Theory and research projects |
| **Stanford CCRMA – Music 356 "Music and AI"** | A survey course including transformers and LLMs | Discussion and ethics material | Not hands-on Transformer training |
| **Berklee Online – "AI for Music and Audio" / "AI in Music"** | Lessons on Transformers for music generation; a **text-to-MIDI assignment**; Colab/Jupyter notebooks | Shows that text-to-MIDI works as an assignment for musicians | Paid; material not public |
| **Music Hackspace – Practical Guide to Generative Music AI** | Hands-on session running a music transformer to generate MIDI | Workshop format | Short workshop |

### 3.2 Notebooks and tutorials (ready to adapt)

| Resource | Content | Notes for free Colab |
|---|---|---|
| **Keras – "Music Generation with Transformer Models"** (keras.io, 2024) | Decoder-only Music Transformer (relative attention) on MAESTRO, 6 blocks, 256 dim; plays audio with fluidsynth; Colab button | Small and fast (epochs of ~30 s on their hardware). Best "works out of the box" example, but Keras, and the code is dense for beginners. |
| **MidiTok – HF Mistral notebook** (`colab-notebooks/` in the MidiTok repo) | REMI + BPE tokenizer trained on MAESTRO, Mistral model (8 layers, 512 dim), HF `Trainer`, generation to MIDI | **Good reference, too heavy as-is** (20 000 steps, High-RAM runtime, no Drive saving). Reduce it a lot. |
| **HF blog: "Using 🤗 to train a GPT-2 for music generation"** (juancopi81) | Tokenized Lakh dataset (MMM format), word-level tokenizer, GPT-2 (6 layers), W&B, Gradio Space; 4 Colab notebooks | Clear pipeline; suggests **JS-Fakes** for quick tests |
| **TristanBehrens/js-fakes-4bars** (HF) | Small GPT-2 (6 layers, 8 heads, vocab 119) generating 4-voice Bach-style bars + generation notebook + dataset | Ready-made small model for a chorale project |
| **Anticipatory Music Transformer** (Stanford CRFM; `jthickstun/anticipation`) | `stanford-crfm/music-small-800k` (**128 M params, GPT-2 architecture, Apache-2.0**), loads with `AutoModelForCausalLM`; Colab for generation, **accompaniment of a melody, infilling** | Inference works well on Colab. **The repo has no training code.** Fine-tuning needs a custom cell (see E4). |
| **HF Audio Course – Unit 4** | Fine-tune an encoder transformer (DistilHuBERT/AST) on **GTZAN** genre classification, ~83–87 % accuracy, Gradio demo | Free Colab friendly; optional audio home exercise |
| **musicgen-dreamboothing** (ylacombe) | LoRA fine-tuning of MusicGen on an audio + text dataset; ~10 GB GPU for musicgen-melody with LoRA + fp16 | Tested on A100 only; **T4 untested**; weights CC-BY-NC. Advanced option only. |
| **text2midi** (AMAAI Lab) + **MidiCaps** | Text-to-MIDI: FLAN-T5 text encoder + transformer decoder, trained on 168 k captioned MIDI | Training is far beyond free Colab; **inference demo** is possible |
| **Folk ABC + GPT experiments** (Gwern's GPT-2 on The Session; WhistleGen with char-level minGPT; **TunesFormer** control codes; **IrishMAN** dataset, 216 k ABC tunes) | Proof that small char-level GPTs on ABC give good results fast | The folk version of our E2; IrishMAN stays a possible alternative dataset |
| **Weimar Jazz Database** (Jazzomat, HfM Weimar; SQLite, **ODbL**) | 456 monophonic jazz solos by 78 players, note by note, with beat grid, **chord changes**, form, style, feel, key, tempo, instrument | **No ABC version exists**: the E2 notebook converts it (one ABC tune per chorus) in ~10 s. Basis for E2–E3 |
| **WikiMT** (`sander-wood/wikimusictext`) | 1,010 Wikifonia lead sheets with genre labels (213 "Jazz") | The public file no longer contains the ABC (it must be rebuilt from Wikifonia MusicXML, whose licensing is unclear). Not used |
| **Transformer Explainer** (Georgia Tech, poloclub) | Interactive GPT-2 in the browser: embeddings, attention, temperature | No install; perfect for E1 |

### 3.3 Gaps this plan fills

- Most existing notebooks are **made for people who code** (dense Keras/PyTorch, or long runs). Ours are "change parameters and listen" notebooks with a fallback checkpoint.
- Few tutorials **measure** whether conditioning works. Here every control has an automatic check.
- No public course **connects symbolic and audio** in one line: E5 takes a tune generated in E2/E3 and arranges it with MusicGen.
- Nothing builds on DUDL's RNN/transfer-learning notebooks, and here E2 and E4 do.
- There is **no jazz dataset in ABC notation**; E2 builds one from the Weimar Jazz Database (solos + chord changes), so the char-level GPT idea works for jazz, not only folk.

---

## 4. Exercises in detail

### E1 · Music as tokens (class 1.5 h · home 1 h)

**Goal.** Understand that a transformer only sees a sequence of integers, and that choosing the representation is a musical decision.

**In class**

- **Part A (30 min) – Transformer Explainer** (browser). Type a sentence, then an ABC line (`|:G2B d2B|c2A F2A|`). Guided questions:
  - What are the 5 most likely next tokens? How does temperature change them?
  - Which earlier tokens does an attention head look at?
  - GPT-2 was trained on text. Why does it still guess something reasonable after `|:G2B d2B|`?
- **Part B (60 min) – notebook `E1_tokens.ipynb`**
  1. Load 3 MIDI files (one exported from their own DAW). Play them in the notebook (piano roll).
  2. Tokenize with **MidiTok REMI** and **TSD**. Print the first 40 tokens (`Bar`, `Position_8`, `Pitch_64`, `Velocity_80`, `Duration_1.0`…).
  3. **Decode back to MIDI and listen**: what changed? (quantized timing, fewer velocity levels, lost rubato/swing).
  4. The same tune in ABC: count characters vs REMI tokens.
  5. Table: tokens per bar for a jazz solo chorus, a piano étude and a drum loop.

**Code challenges (optional)**
- Change `beat_res` / `num_velocities` and plot *sequence length vs fidelity*.
- Train MidiTok **BPE** with a small vocabulary and see the sequence get shorter.

**Questions**
- Attention cost grows with the square of the sequence length. Which representation is cheapest for a 3-min piano piece?
- Which tokenization would you choose for drums? For a jazz performance with expressive timing?

**Home H1 (1 h).** Tokenize one of your own pieces and report: number of tokens, vocabulary size, what was lost in the round-trip (with audio examples). Answer 5 short questions about attention from the Explainer.

---

### E2 · Your first music GPT (class 2.5 h · home 1.5 h)

**Status: built and tested.** Notebook `Transformers_music/E2_music_GPT_jazz.ipynb`; teacher files in `Transformers_music/E2_teacher/` (fallback checkpoint, prepared dataset, executed copy with all outputs). Runs top to bottom on an Apple M1 Pro; **dry run on a free Colab T4 still to do**.

**Goal.** Train a small decoder-only transformer from scratch on jazz solos and *hear* it learn.

**Why ABC.** ABC is plain text: the tokenizer is a character table (exactly like `DUDL_RNN_loremipsum`), students can **read** the output and see errors, and with the chord symbols inside the text the model learns melody *and* harmony together.

**Data: Weimar Jazz Database → ABC.** No jazz dataset in ABC exists, so a (hidden) helper cell downloads `wjazzd.db` (42 MB, Jazzomat server) and converts it in ~30 s:
- the 431 solos in 4/4 → **1,257 ABC tunes, one per chorus** (12-bar blues, 32-bar AABA…), ~950,000 characters, 79 characters in the vocabulary;
- **data augmentation: every solo in all 12 keys** (`N_KEYS = 12`, a form field) → 15,084 choruses, 11.7 M characters. With one key the model overfits after ~2,000 steps; with 12 keys it does not overfit within 3,000 steps;
- header lines carry the metadata: `T:` title, `C:` soloist (instrument), `R:` style and feel (`bebop, swing`), `Q:` tempo, `K:` key; `M:4/4` and `L:1/8` everywhere;
- rhythm rounded per beat to 16ths or 8th-note triplets; **swing 8ths written straight**, as on a lead sheet (swing is added back at playback); short gaps closed (rests only when they last a quarter or more); a space marks each beat, a new line every 4 bars, `|]` ends the chorus;
- Jazzomat chord names rewritten as lead-sheet chords (`C-7` → `Cm7`, `Fj7` → `Fmaj7`, `Bb79b` → `Bb7b9`), placed at the beat where they change **and repeated at the start of every bar**;
- **every flat or sharp is written on its note** (`_B` even in `K:Bb`; a plain letter is always a natural). `show_abc` hides the redundant accidentals in the sheet music, and `make_prompt` adds them to prompts typed in normal ABC;
- split **by solo** (90 / 10 %) so that validation choruses come from solos the model has never seen. All 30,130 bars pass the bar-duration check.

Example (first lines of a chorus):
```
T:Anthropology
C:Art Pepper (cl)
R:cool, swing
M:4/4
L:1/8
Q:1/4=219
K:Bb
"Bb6"F2 _E3/_B,/ _D2 _E2|"Bb6"_B,2 _B,2- "G7"_B,3/z/ z2|"Cm7"z4 "F7"z4|"Bb"z4 "Gm7"D,2 A,C-|...
```
Licence: **ODbL** (credit the database; share changed versions, such as our `jazz_abc_12keys.txt`, under the same licence). Data-ethics points in the notebook: real musicians' improvisations, transcription as interpretation, and the soloist's name as a control ("a solo by Charlie Parker").

**Model.** A 3.3 M-parameter mini-GPT (4 blocks, 4 heads, 256-dim embeddings, context 256 characters ≈ 8–10 bars, dropout 0.1), nanoGPT style. The **10-line self-attention function** is shown first, commented line by line (Q, K, V, causal mask, softmax), and tried on a toy jazz snippet. The training loop is the DUDL one (`CrossEntropyLoss`, Adam, lr 1e-3, batch 64 random chunks), in a `train_model()` function that the code challenges reuse.

**In class**
1. (15 min) Download and convert the data, read one chorus as text and as sheet music (abcjs player: the solo over a band added at playback, MIDI download), ABC cheat sheet (including *why* the data is written for the model: straight 8ths, accidentals on every note, a chord in every bar), build `letter2number` exactly as in `loremipsum`.
2. (15 min) Self-attention in 10 lines, the model classes, count parameters (link to `DUDL_ANN_numParameters`; the loremipsum LSTM has 5.4 M).
3. (15–20 min, GPU) Train 3,000 steps with a live loss plot (train, validation, uniform-guessing line); checkpoints at steps 200, 1,000 and 3,000 saved to Drive. **Measured: 14 min on an M1 Pro**; ≈10 min expected on a T4 (to check).
4. (30 min) **The learning curve you can hear**: the same prompt at each checkpoint, with sheet music, playback and an automatic report (bars, % bars with the right length, invalid characters, finished?). The report also gives the % of notes on chord tones (real solos ≈ 50 %, random notes ≈ 33 %). Measured (temperature 0.9, `Blues for E2` prompt): step 200 ≈ 5 % of the bars right, ≈ 30 % chord tones; step 1,000 ≈ 50–65 %, ≈ 40 %; step 3,000 ≈ 94 %, ≈ 48 %. At step 3,000 the chords are locally idiomatic (ii–V chains, turnarounds) but have no form: only 1 of 30 `Blues for E2` choruses had blues changes, and many drift to other keys.
5. (40 min) **Sampling**: temperature 0.7 / 0.9 / 1.2, top-k. Measured trade-off: a low temperature gives more chord tones but non-stop lines with no rests (a rest is never the single most likely character); a high one gives realistic phrasing but more wrong notes. Default 0.9. Then **a solo over real chord changes**: `generate_over(model, prompt, lead_sheet('blues', 'Bb'))` writes the chord of every bar at each `|` (two chords in a bar: beats 1 and 3) and does not let the model write `"`, so the model only plays the solo. The free solo and the one over a Bb blues are compared side by side. Presets: blues, minor blues, rhythm changes, ii–V–I, modal, in any key, or a lead sheet typed by the student. Finally a Colab form for your own prompt (soloist, style, key, tempo, title, changes, first bar typed in normal ABC).
6. (20 min) **Attention maps**: on an unseen chorus, score every head for "bar line → previous bar line" (counting beats) and "note → its chord symbol" (reading the harmony), then plot where the best head looks.

**Code challenges (optional)**
1. **Context length:** `block_size=32` (about one bar) vs 256. Are its bars still the correct length? Does it keep the chord progression going?
2. **LSTM vs Transformer:** the `loremipsum` LSTM with `batch_first=True` and a 256-dim embedding, trained with the same `train_model()`. Compare validation loss, time per step and samples.
3. **No positional embedding:** `MusicGPT(..., use_pos=False)`. What happens to bar structure? (Open-ended.)
5. **No data augmentation:** `N_KEYS = 1`. Compare the curves (overfitting after ≈1,800 steps) and the chord-tone %.
4. **Memorization check:** `longest_common_substring(sample, train_text)`, compared with the same measure for real unseen choruses (jazz players reuse licks; many tunes share changes). Measured: ≈17–23 characters for generated choruses vs ≈19–26 for real unseen ones, i.e. no real copying.

**Fallback.** `E2_teacher/e2_jazz_gpt.pt` (40 MB: checkpoints 200 / 1,000 / 3,000 + vocabulary + loss curves; 12-key data, final validation loss 0.95). Students set `TRAIN = False` and load it from their Drive folder or from `FALLBACK_URL`. Tested.

**Home H2 – Sampling lab (1.5 h).** 2 checkpoints × 3 temperatures × 5 samples, one function call (≈ 5 min on an M1 Pro). For each setting the table gives:
- % valid (no invalid characters, at least 4 bars),
- % of bars with the correct duration for 4/4,
- % finished choruses (`|]`),
- % of notes on chord tones (real solos ≈ 50 %, random notes ≈ 33 %),
- longest copied fragment from the training set.

All samples are saved as one `.abc` file. Fill a listening log (musical? follows the changes?). Hand in the table, the best and worst sample (ABC + MIDI), and a 5-line explanation.

**Changes from Draft 1.** Jazz instead of IrishMAN; playback with abcjs in the browser instead of abc2midi + fluidsynth (no `apt install`); validity measured with a small built-in ABC checker instead of music21/abc2midi; helpers live in hidden cells of the notebook (not yet in `course_utils.py`).

**Quality pass (Draft 3).** The first version "sounded like jazz but not convincing". Comparing 40 generated choruses with real unseen ones showed four problems, each fixed and measured (same model size and training time):

| Problem (first version) | Cause | Fix |
|---|---|---|
| Non-stop lines: 40 notes between rests (real: 20), rests 13 % of the time (real: 21 %) | temperature 0.8 suppresses rare characters (rests, `|]`) | default temperature 0.9 (and the trade-off becomes question 17) |
| Notes fit the chords less and less: 43 % chord tones in the first 256 characters, 30 % later | the `K:` line leaves the 256-character context, so a plain `B` is ambiguous | every flat/sharp written on its note; the sheet music hides them again |
| A new chord at 94 % of the bar lines (real: 50 %), aimless harmony | chords written only when they change, so the current chord is often out of context | the chord repeated in every bar |
| Overfitting after ≈1,800 steps | only 860 k characters | every solo in 12 keys (octave chosen to stay in the instrument's register) |

Result (temperature 0.9 vs the first version at 0.8): chord tones 41 → 44 % (real 50 %), notes between rests 40 → 26, rests 13 → 21 % of the time, chord changes per bar 1.3 → 0.9 (real 0.7); bars with the right length unchanged (≈ 86 %). Playback now adds a **band** (rule-based walking bass, ride + hi-hat, piano comping), plays the soloist's instrument and swings less at fast tempos (the first version played the solo on piano over abcjs's oom-pah chords). A 512-character context was tried and did not help at this training budget. The remaining gap to real solos is mostly compute: the validation loss is still going down at step 3,000.

**Chords vs melody (Draft 3b).** The next test: "the chords don't match the melody". Playback was checked first (abcjs plays exactly the written notes), so the cause is the model:
- **It invents its own chords, with no form** (see step 4).
- **Its line follows the key more than the chords.** Move every chord a fourth up or down and the notes fit almost as well: 48 % vs 42 % duration-weighted chord tones on the class prompt. Real solos: 53 % vs 41 %. First version: 43 % vs 42 %. The walking bass makes this easy to hear.

Two training changes, each measured on 40 choruses, did not fix it:
- *Original keys only:* validation loss 1.01 at best (overfitting after step 2,500) vs 0.97. The free solos fit the chords slightly better (4 points above the shifted chords vs 2), but over given changes the fit is worse (5 points vs 9).
- *Cosine learning-rate decay:* 0.977 vs 0.970.

What works is **giving the model the changes** (step 5, with the same model). Over a Bb blues: 50 % chord tones in `report()` (real 50 %, free solos 47 %), 9 points above the shifted chords (real 12), 92 % of the bars right, every chorus finished. Over rhythm changes: 49 %.

**Data and playback, checked end to end (Draft 3c).** The next question was whether the ABC conversion is to blame (`E3/B,/ _D2 E2` in the first version became `_E3/_B,/ _D2 _E2`). It is not. In K:Bb both lines are the same notes, E♭ B♭ D♭ E♭: an accidental in ABC is absolute, so `_E` is E♭ whatever the key signature says. Checked on everything:
- both notations give the same pitch, onset and chord for all 174,692 notes of the 1,257 choruses;
- rebuilt from the database, every bar parses back to the database pitches (182,639 notes, both notations);
- the 12-key file is exactly each solo moved by its shift, notes and chord roots (13,827 transposed choruses).

Running abcjs itself (JavaScriptCore) on the data found two small playback bugs, now worked around in `arrange()` (audio only):
- after a note tied in over the bar line, abcjs forgets that note's accidental for the rest of the bar: 0.13 % of the notes were played wrong (0.38 % with the first version's notation);
- abcjs voices a few chord names wrongly or as bare triads (`7#9`, `m11`, `7alt`, `aug`, …): 6 % of the piano chords.

After the fix: 0 wrong notes out of 390,000, and 0.01 % of the chords. In generated solos the tie bug changed only 7 of about 13,000 notes, so it does not explain the chord/melody mismatch above.

---

### E3 · Telling the model what you want: conditioning (class 2 h · home 1.5 h)

**Goal.** In a causal transformer, **conditioning = putting information before what you want generated**. This is how control tokens work in CTRL, TunesFormer and the EMOPIA emotion model. The other route, an encoder plus cross-attention, is what MusicGen and text2midi use; here it is only shown in a demo.

**Data.** The same jazz ABC as E2 (Weimar Jazz Database), so no new dataset: the controls are already in the text. The meter is always 4/4, so **meter control (Draft 1) is replaced by key, style/soloist and chords.**

**Part A (40 min) – "You already have a conditional model."** The E2 model always saw the header lines (`C:` soloist, `R:` style and feel, `Q:` tempo, `K:` key, even `T:` title) before the notes. So no new training: prompt with `K:Bb` vs `K:Eb` vs `K:Cm`, `R:bebop, swing` vs `R:cool, swing` vs `R:postbop, latin`, Parker vs Coltrane vs Chet Baker.
- **Measure it:**
  - *key*: 20 choruses per requested key; estimate the key from the pitch-class histogram (music21 or Krumhansl profiles) and plot a **confusion matrix** (requested vs obtained). Also: do the chord symbols start on the tonic or on a ii–V into it?
  - *style / soloist*: simple features per condition (notes per bar, % 16ths, % triplets, range, % chromatic notes) compared with the same features on the real solos of that style or player. Can a small classifier tell the generated "Parker" from the generated "Getz"?
  - *title*: `T:Blues for …` vs `T:Rhythm …` vs `T:Giant Steps`. Do the generated changes follow a 12-bar blues, rhythm changes, Coltrane changes? (A first glimpse of memorization of chord changes.)
- **Out-of-distribution:** ask for something that never appears in the data: `M:3/4` (all 3/4 solos were left out), `K:F#m`, a singer as soloist (`C:Billie Holiday (voc)`), `R:bossa nova`. What happens?

**Part B (80 min) – A solo over *your* changes.**
- **B1, no training (20 min): forced chords, measured.** E2 already gave students `generate_over`: every time the model writes `|`, the notebook writes the next chord of the lead sheet, and `"` is masked so the model cannot write chords. Conditioning is still "what is in the context", now written by us during generation. B1 opens the box and measures it with a *shifted-chords control*: score the notes against the given chords and against the same chords moved by 1–11 semitones. A line that follows the changes fits the real chords clearly best; a line that only follows the key fits a chord a fourth away almost as well. Measured in E2: free solos 48 % vs 42 %, over a forced blues 52 % vs 42 %, real solos 53 % vs 41 %. Then: fast changes (two chords per bar, Coltrane changes), chords the model has rarely seen, and a random chord line.
- **B2, first fine-tuning (60 min): the chords up front.** Preprocessing (given): pull the chords out into a **chord line** placed *before* the solo, then the solo without chords:
  ```
  R:cool, swing
  K:Bb
  %CHORDS: Bb6 | Bb6 G7 | Cm7 F7 | Bb Gm7 | ...
  F2 E3/B,/ _D2 E2|B,2 B,2- B,3/z/ z2|z8|z4 D,2 A,C-|...
  ```
  **Fine-tune the E2 model** on this format for ~10 min (a first taste of transfer learning, like DUDL's `transferlearning` notebooks). Now the model can see what is *coming* (the next chord, the turnaround), not only the current chord as in B1.
- Students type a progression (a blues, rhythm changes, their own tune) and get a solo, played back by abcjs with the chords underneath (they appear again as chord symbols at playback).
- **Measure it:** *chord-tone ratio* = % of strong-beat notes (beats 1 and 3) that belong to the current chord. Because jazz solos use many tensions and chromatic approach notes, also report a softer *chord-scale ratio* (chord tones + 9/11/13). Compare (a) real WJazzD solos, (b) B1, (c) B2, (d) B2 given a random chord line. The real solos are the reference, not 100 %.

**Code challenges (optional)**
- **Harmonizer / reharmonizer:** swap the order (solo first, chord line after) and fine-tune again. Now the model writes chords for a melody. Same code, the condition is just in a different place.
- **Condition dropout:** in 20 % of training examples, leave the chord line empty, so one model works both with and without chords.
- **Folk contrast:** fine-tune on Nottingham (folk tunes with chords) and compare chord-tone ratios: why are folk melodies much more "inside" than bebop lines?
- **Demo (teacher, 10 min): text2midi** inference. A caption goes through a text encoder and the decoder reads it via cross-attention. Discuss: what would you need to train this yourself? (168 k captioned MIDI files and a lot of GPU.)

**Home H3 (1.5 h).** Report on one control (key, style/soloist, title or chords): the metric, the confusion matrix / feature comparison / chord-tone ratio, and a 3-sample listening comparison. Then propose the control token(s) for your project.

---

### E4 · Standing on giants: pretrained models and fine-tuning (class 2 h)

**Goal.** Use a transformer pretrained on a large MIDI corpus and adapt it to a small target style. Compare that with training from scratch.

**Model.** **Anticipatory Music Transformer – `stanford-crfm/music-small-800k`**: 128 M params, GPT-2 architecture, trained on Lakh MIDI, Apache-2.0. It loads with `AutoModelForCausalLM`; the `anticipation` library handles MIDI ↔ tokens.

**Part A – Co-composition (50 min, inference only)**
1. Unconditional generation (15–20 s of multi-instrument MIDI).
2. **Accompaniment:** give a melody (a POP909 melody, or the student's own MIDI, or their E3 tune converted to MIDI) and generate an accompaniment around it.
3. **Infilling:** keep bars 1–4 and 9–12 and regenerate bars 5–8. Repeat until you like it, as in the CRFM "co-composition" workflow.
4. Discuss how the model can take a melody as a control. (Anticipation: control notes are placed in the sequence a little *before* their time.)

**Part B – Fine-tune on a small corpus (70 min)**
- Corpus: 100–200 MIDI files from one style, tokenized by the teacher in advance (e.g. one composer from MAESTRO, VGMIDI video-game piano, or the class's own pieces).
- HF `Trainer`, fp16, gradient checkpointing, ~15–20 min on a T4 (**estimate, check in a dry run**).
- **Compare before and after** with simple features: pitch-class histogram, note density, mean duration, distance to the target corpus. Optional: **Fréchet Music Distance** (`frechet-music-distance` package).
- **Transfer-learning experiment** (links to DUDL `transferlearning`): the same 15 minutes and the same 150 files, *from scratch* vs *fine-tuned*. Listen to both.
- Optional: **LoRA** with PEFT. Compare the number of trainable parameters and the result.

**⚠ Teacher note.** The `anticipation` repo has **no training code**. You'll need to write a fine-tuning cell: convert MIDI to events with the library, cut fixed-length sequences in the model's autoregressive format, then use the HF `Trainer`. **Do a dry run on a free T4 before the course.**
**Fallback B:** fine-tune the E2/E3 jazz ABC model on a different ABC collection (Nottingham, or a regional folk collection from abcnotation.com): a jazz → folk style transfer, with "from scratch vs fine-tuned" on the same small corpus. Same concept, guaranteed to run.

---

### E5 · Audio is tokens too (class 1 h, inference only)

**Goal.** See that audio models (MusicGen) are transformers on **codec tokens**, and why they are heavier than MIDI models.

1. **EnCodec round-trip (20 min):** encode a 10-s clip with `facebook/encodec_24khz` (HF transformers). Look at the shape (*codebooks × frames*). Decode at 1.5 / 3 / 6 / 12 / 24 kbps and listen.
   - Compute **tokens per second** for audio (75 frames/s × number of codebooks) vs REMI tokens per second from E1. Why are audio LMs so much bigger? Why does MusicGen interleave codebooks with a delay pattern?
2. **MusicGen (40 min):**
   - `musicgen-small`, text-to-music: write prompts and change one word at a time.
   - `musicgen-melody`, **melody conditioning**: render your E2/E3 tune to audio and ask MusicGen to arrange it ("string quartet", "8-bit chiptune", "lo-fi hip hop"). This gives a **symbolic → audio** pipeline in one notebook.
   - Discuss: the text is encoded by T5 and enters the decoder through **cross-attention**, the other conditioning route from E3.

**Optional home extensions**
- HF Audio Course Unit 4: fine-tune DistilHuBERT/AST on GTZAN (an *encoder-only* transformer for classification). Use it to **classify the genre of your MusicGen outputs**, an automatic "did the style prompt work?" metric.
- Advanced: LoRA fine-tuning of MusicGen with `musicgen-dreamboothing` on a few minutes of your own instrumental music. **Untested on a T4.** Treat it as an experiment, not a requirement.

---

## 5. Mini-project (pairs · ~6 h at home · 1 h demo)

### 5.1 Menu (★ = difficulty)

| # | Project | Starts from | Control to measure |
|---|---|---|---|
| 1 | ★ **Jazz chorus generator** with key / style / soloist control (or the folk version on IrishMAN with meter / key / tune type) | E2 + E3A | key confusion matrix, feature stats per style or soloist |
| 2 | ★★ **Solo over your changes**: a chorus over a chord progression you type (blues, rhythm changes, your own tune) | E3B | chord-tone / chord-scale ratio vs real solos |
| 3 | ★★ **(Re)harmonizer**: chords for your melody | E3B (reversed) | chord-tone ratio, chord vocabulary |
| 4 | ★★ **Drum groove generator** with style control (Groove MIDI Dataset; drums written as a 16-step text grid, so the E2 code works as-is) | E2 + E3 | style classifier or feature stats per style |
| 5 | ★★ **Co-composer**: AMT fine-tuned on your own corpus, with accompaniment and infilling | E4 | feature distance to the corpus |
| 6 | ★★ **Symbolic → audio**: your generator + MusicGen-melody arrangement, judged by the GTZAN classifier | E3 + E5 | genre classifier accuracy |
| 7 | ★★★ **Bach chorale continuation** (JS-Fakes / JSB; `js-fakes-4bars` model, MidiTok) | E4 | voice-leading checks (range, parallel fifths) |
| 8 | ★★★ **Emotion-controlled piano** (EMOPIA, a Q1–Q4 token in front, fine-tune) | E3 + E4 | listener agreement with the requested quadrant |
| 9 | ★★★ **"Text-lite" to MIDI**: MidiCaps attributes (genre, mood, tempo, key) turned into control tokens | E3 + E4 | attribute accuracy |
| 10 | Your own idea (approved by the teacher) | — | — |

### 5.2 Deliverables
1. A notebook that runs from top to bottom on free Colab (checkpoints on Drive or HF Hub).
2. **3 cherry-picked + 3 random** samples (MIDI + audio).
3. **One objective metric** showing that the control works, compared with a baseline.
4. **Mini listening test**: at least 5 listeners, blind, with 1 real piece as an anchor. Rate 1–5 for "musical" and "matches the requested condition" (a Google Form is enough).
5. **1-page reflection**: failure cases, memorization check, data licence, how you would improve it.
6. **5-min demo** in the final session (live generation if possible).

### 5.3 Rubric
| Criterion | Weight |
|---|---|
| Working, reproducible pipeline | 20 % |
| Conditioning works **and is measured** | 25 % |
| Listening test: design and results | 20 % |
| Critical reflection (failures, memorization, data, ethics) | 20 % |
| Demo / presentation | 15 % |

*Timing:* with ~10 pairs, 5 min each fills the 1 h demo session. Run the listening test online before the session and show the results live.

---

## 6. Practical Colab kit (teacher prep)

**E2 is self-contained:** its helpers (WJazzD → ABC converter, `show_abc` with the backing band, `check_abc`, `chord_tone_ratio`, `longest_common_substring`, `make_prompt`, `tidy_accidentals` / `explicit_accidentals`) are folded Colab cells inside the notebook, and it needs no installs (abcjs loads from a CDN in the output cell). When E3 is written, move them into `course_utils.py` so E2 and E3 share one copy.

**Shared helper module `course_utils.py`** (in a GitHub repo, `!wget` in the first cell). It hides boilerplate so notebooks stay short:
`play_midi(path)` (piano-roll player via `midi-player` / html-midi-player) · `show_abc(text)` (abcjs sheet + playback) · `abc_to_midi()` · `render_audio()` (fluidsynth + soundfont) · `check_abc(text)` (valid? bar durations? estimated key) · `chord_tone_ratio()` · `longest_common_substring()` · `save_ckpt()/load_ckpt()` to Drive.

**Install cell (pinned versions!):** `torch` (preinstalled), `transformers`, `peft`, `miditok`, `symusic`, `music21`, `midi-player`, `anticipation` (git), `apt install abcmidi fluidsynth` + a General MIDI soundfont. Pin the versions you tested: MidiTok and transformers APIs change between releases.

**Free-tier rules for students**
- Runtime → T4 GPU. Mount Drive in the first cell. Save a checkpoint every N steps.
- Training runs ≤ 20 min. If the GPU quota runs out, use the **fallback checkpoint** and do inference on CPU.
- Work in pairs on one GPU account to halve quota usage.

**Timings to check in a dry run** (the numbers here are estimates): E2 training at 3 k steps (measured 14 min on an M1 Pro, H2 lab 5 min, whole notebook 21 min; T4 still to check), E3 fine-tune, E4 fine-tune with batch/sequence length that fits in 16 GB, MusicGen-melody generation time.

---

## 7. Datasets cheat sheet

| Dataset | Type | Size | Labels usable for conditioning | Use in |
|---|---|---|---|---|
| **Weimar Jazz Database** (WJazzD v2.1, Jazzomat) → our ABC | solo transcriptions (SQLite) → ABC | 456 solos → **1,257 choruses** in 4/4 (950 k characters); × 12 keys = 15,084 choruses (11.7 M characters) | **chords**, key, style, feel, soloist, instrument, tempo, title | E2, E3, P1–P3 |
| **IrishMAN** (HF `sander-wood/irishman`) | ABC | 216 k tunes | meter, key (headers) | folk alternative for E2 / P1 |
| **Nottingham Music Database** | ABC | ~1 k tunes | **chords**, meter, key, tune type | E3 folk contrast, E4 Fallback B |
| **POP909** | MIDI | 909 pop songs | melody/accompaniment tracks, chords, beats | E4, P2, P5 |
| **Groove MIDI Dataset** (Magenta) | drum MIDI | ~13.6 h | **style/genre**, tempo, drummer | P4 |
| **JSB Chorales / JS-Fakes** | MIDI / tokens | hundreds / thousands | 4 voices | P7 |
| **EMOPIA** | piano MIDI + audio | 1 087 clips | **emotion quadrant** Q1–Q4 | P8 |
| **MidiCaps** | MIDI + captions | 168 k | **text**, genre, mood, key, tempo | E3 demo, P9 |
| **MAESTRO** | piano MIDI + audio | ~200 h | composer | E1, E4 |
| **Lakh MIDI** | multi-track MIDI | 176 k | (genre via matching) | pretraining of AMT |
| **GTZAN** (HF `marsyas/gtzan`) | audio | 1 000 × 30 s | genre | E5 optional |

Licence reminder: WJazzD (ODbL: credit it, share derived data such as `jazz_abc_12keys.txt` under the same licence), IrishMAN (research use requested), MusicGen weights (CC-BY-NC), Lakh (mixed sources). All fine for teaching. Students should not publish commercial outputs.

---

## 8. Teacher checklist and risks

**Before the course**
- [ ] Write `course_utils.py` and test it on Colab.
- [x] E2 notebook (jazz) written and run end to end locally; fallback checkpoint `e2_jazz_gpt.pt` and `jazz_abc_12keys.txt` in `Transformers_music/E2_teacher/`.
- [ ] E2 dry run on a free Colab T4: training time, H2 lab time, abcjs playback and *Download MIDI* link inside Colab. Put `e2_jazz_gpt.pt` on Drive or set `FALLBACK_URL`.
- [ ] Prepare the other datasets (tokenized E4 corpus; Nottingham for the E3 contrast) as a Drive zip or an HF dataset. Keep a copy of `jazz_abc_12keys.txt` in case the Jazzomat server is down.
- [ ] Pre-train fallback checkpoints: E3 chord model, E4 fine-tuned AMT.
- [ ] Write and dry-run the **AMT fine-tuning cell** (main technical risk), or decide to use Fallback B.
- [ ] Dry-run every notebook on a *free* account and note real timings in each notebook.
- [ ] Prepare a "seed melodies" folder (MIDI + ABC) for students without their own material.
- [ ] Listening-test Google Form template.

**Risks**
| Risk | Mitigation |
|---|---|
| Colab GPU quota / disconnections | ≤ 20-min runs, Drive checkpoints, fallback checkpoints, pairs share a GPU |
| Library version changes | Pinned install cell, tested the week before |
| ABC parse errors | abcjs renders broken ABC with warnings instead of failing; `check_abc` counts invalid characters and wrong bar lengths as metrics |
| Jazzomat download server down | Cached `jazz_abc_12keys.txt` on Drive (the notebook uses it if present) |
| abcjs or its sound-font server (paulrosen.github.io) unreachable | No audio in the notebook; the ABC text and the `.abc` file of H2 still work in any ABC tool |
| AMT fine-tuning effort | Fallback B (ABC fine-tune on another collection) |
| Students only "press run" | Every exercise has questions to answer, a metric, and a listening log |

---

## 9. Sources

**Courses**
- UPF – Generative Algorithms for Sound and Music: https://upf-smc-genai-music.github.io/lectures · https://upf-smc-genai-music.github.io/assignments/ · https://www.upf.edu/web/smc/generative-algorithms-for-sound-and-music
- The Sound of AI – Generative Music AI course: https://github.com/musikalkemist/generativemusicaicourse
- UCSD CSE 190 (ML4Music) assignments: https://sites.google.com/eng.ucsd.edu/ucsd-cse-190/assignments · notebooks: https://github.com/sdubnov/ML4Music-HW
- UCSD CSE 253 symbolic generation module: https://cseweb.ucsd.edu/classes/sp25/cse253-a/slides/module3.pdf
- NTU Deep Learning for Music Analysis and Generation: https://affige.github.io/teaching_deepmir25.html
- Stanford CCRMA Music and AI syllabus: https://ccrma.stanford.edu/courses/356-winter-2024/music-and-ai-syllabus.pdf
- Berklee Online – AI for Music and Audio: https://online.berklee.edu/courses/ai-for-music-and-audio
- Music Hackspace – Practical Guide to Generative Music AI: https://musichackspace.org/a-practical-guide-to-generative-music-ai-for-developers-october-2025

**Notebooks, models and tools**
- Keras – Music Generation with Transformer Models: https://keras.io/examples/generative/midi_generation_with_transformer/
- MidiTok (incl. HF Mistral Colab): https://github.com/Natooz/MidiTok · docs: https://miditok.readthedocs.io
- HF blog – GPT-2 for music generation: https://huggingface.co/blog/juancopi81/using-hugging-face-to-train-a-gpt-2-model-for-musi
- js-fakes-4bars model: https://huggingface.co/TristanBehrens/js-fakes-4bars
- Anticipatory Music Transformer: https://github.com/jthickstun/anticipation · https://huggingface.co/stanford-crfm/music-small-800k · https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html · https://crfm.stanford.edu/2024/08/06/co-composition.html
- HF Audio Course Unit 4 (music genre classifier): https://huggingface.co/learn/audio-course/chapter4
- MusicGen docs: https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md · LoRA fine-tuning: https://github.com/ylacombe/musicgen-dreamboothing
- text2midi: https://huggingface.co/amaai-lab/text2midi · MidiCaps paper: https://arxiv.org/abs/2406.02255
- Transformer Explainer: https://poloclub.github.io/transformer-explainer · https://github.com/poloclub/transformer-explainer
- html-midi-player (works in Colab via `midi-player`): https://github.com/cifkao/html-midi-player
- Fréchet Music Distance: https://pypi.org/project/frechet-music-distance/ · MusPy: https://arxiv.org/abs/2008.01951

**Datasets and papers**
- Weimar Jazz Database: https://jazzomat.hfm-weimar.de/dbformat/dboverview.html · download: https://jazzomat.hfm-weimar.de/download/download.html · licence (ODbL): https://opendatacommons.org/licenses/odbl/1.0/
- WikiMT (jazz-labelled lead sheets, ABC no longer included): https://huggingface.co/datasets/sander-wood/wikimusictext
- abcjs (sheet music + synth in the browser, swing option): https://docs.abcjs.net/audio/synthesized-sound
- IrishMAN: https://huggingface.co/datasets/sander-wood/irishman · TunesFormer (control codes): https://arxiv.org/abs/2301.02884
- Gwern – GPT-2 folk music: https://gwern.net/gpt-2-music · WhistleGen (char-level minGPT on ABC): https://wandb.ai/johnowhitaker/whistlegen_v2/reports/WhistleGen-V2-Making-Music-With-Transformers--VmlldzoyMTAwNjAz
- EMOPIA: https://arxiv.org/abs/2108.01374
- Moonbeam MIDI foundation model (309 M+, too large for this course; for reference): https://arxiv.org/abs/2505.15559
- NotaGen (ABC classical generation): https://arxiv.org/abs/2502.18008
