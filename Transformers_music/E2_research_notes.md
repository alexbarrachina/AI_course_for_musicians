# E2 "Your first music GPT" (jazz): research notes

Handoff for an LLM that continues this work. State on 24 Nov 2026.
All numbers are the recorded results of 21–24 Nov. They were not re-measured for this document.

Each claim is marked when it matters: **[measured]**, **[hypothesis]** or **[listening]** (the teacher's verdict by ear). "The teacher" is the user. They decide by listening.

---

## 1. Summary

1. **Goal.** A class exercise: music technology students with basic Python train a small character-level GPT from scratch on jazz solos written as ABC text, with chord symbols. It must train in about 20 min on a free Colab T4.
2. **Versions.** Four exist. **E2** (first version) is the reference, and the only one committed. **v2** was reverted. **E2b** measured worse than E2. **E2c** is built, but has not been trained on the Debian machine, measured or heard.
3. **Main problem.** The music sounds like jazz but is not convincing, and the melody does not follow the chords.
4. **Main finding [measured].** Every 3,000-step model uses the chord symbols mostly to know the key, and hardly at all to choose chord tones. Left alone, the models also write chords with no form.
5. **Best explanation [hypothesis].** A character model learns chord use last, after the melody stops improving. Data in 12 keys delays that moment, so at a fixed budget it halves chord use.
6. **Now.** The teacher trains E2c on their Debian machine. Next: measure it, listen to it, compare it with E2 and E2b (section 9).

---

## 2. Constraints and the teacher's preferences

- **ABC text stays.** The teacher likes it because it is text, as for ChatGPT.
- **Jazz with chords and melody stays**, because E3 (conditioning) needs both. The teacher finds folk data too simple. Classical in ABC was considered and not recommended: it is polyphonic, has no chord symbols and needs much bigger models.
- **Listening decides.** v2 was better on paper, and the teacher rejected it by ear. Never present metrics alone as proof of quality.
- **Change one thing at a time.** v2 changed five things at once, so nobody knows which one hurt.
- **Ask before making E2 more complex**, and before any long run on the Mac. The teacher said: "Stop training. I will train in Debian machine" and, about using the Mac run of E2c, "No, Debian only".
- **Do not re-run or reproduce recorded results** when asked to write something up.
- **Commit or push only when asked.**
- Explain in plain words. The teacher doubts that 12-key augmentation is the cause of the problem, so explain it as a budget effect, not as "augmentation is harmful".

---

## 3. Data

**Weimar Jazz Database (WJazzD) → ABC.** No ready-made jazz dataset in ABC was found. The notebook (Helper 1) downloads `wjazzd.db` (42 MB, ODbL licence) and converts it.

- 431 solos in 4/4 → **1,257 tunes, one per chorus**, 30,130 bars, about 864,000 characters, 79-character vocabulary.
- Header: `X T C R M L Q K` (title, soloist and instrument, style and feel, metre, unit, tempo, key). `M:4/4` and `L:1/8` always.
- Rhythm rounded per beat to 16ths or 8th triplets. Swing 8ths written straight; swing is added at playback (66).
- Chord symbols only where the chord changes (E2, E2b, E2c). v2 repeated the chord in every bar.
- **Split by solo:** solos (title + soloist) sorted, shuffled with `random.Random(42)`, first 10 % = validation. That gives 189 validation choruses and about 30,069 notes. Validation is always in the original keys.

**Notation variants**

| Name | Accidentals | Chords | Keys | Where |
|---|---|---|---|---|
| std | key signature | on changes | 1 | `E2_teacher/jazz_abc.txt` |
| std 12 | key signature | on changes | 12 | `E2b_teacher/jazz_abc_12keys.txt` |
| x | every flat/sharp written | on changes | 1 and 12 | E2c writes `jazz_abc_explicit.txt` and `jazz_abc_explicit_12keys.txt` |
| xc | every flat/sharp written | start of every bar too | 1 and 12 | `dataset_tester/jazz_abc_1keys.txt`, `saved_not_working_E2_teacher/jazz_abc_12keys.txt` (v2) |

Example, the same notes in K:Bb: std `F2 E3/B,/ _D2 E2` = explicit `F2 _E3/_B,/ _D2 _E2`.

**The data is right [measured].** The teacher suspected the ABC conversion and the transposition. Both were checked:
- Both notations give the same pitch, onset and chord for all 174,692 notes of the 1,257 choruses, and match the database.
- v2's 12-key file: all 13,827 transposed choruses are exact.
- E2b's 12-key file: 13,819 of 13,827 are exact. The other 8 are *So What* in `Dbdor`/`Gbdor` and *Adam's Apple* in `Gbmix`. They carry the wrong key label. This is 0.05 % of the file and does not explain what is heard.
- E2c's text has exactly the same notes, rhythms and chords as E2b's, in all 15,084 choruses (2,385,288 notes). Removing the extra accidentals gives E2b's text back byte for byte.
- In the 12-key files, the choruses of one solo are stored together, key by key. Neighbouring tunes are different choruses, not transpositions of each other. This caused one false alarm.

---

## 4. Model and training (common to all versions)

- Mini-GPT, nanoGPT style: 4 blocks, 4 heads, 256-dim embeddings, dropout 0.1, learned positions, about 3.3 M parameters.
- Adam, lr 1e-3, 3,000 steps, seed 0, checkpoints at 200 / 1,000 / 3,000. E2b and E2c also keep `best` (lowest validation loss, checked every 100 steps).
- Checkpoint file: `config`, `vocab`, `checkpoints` (dict), `losses`.
- E2: context 256, batch 64. E2b and E2c: context 1,024, batch 12.
- On an M1 Pro: E2 about 13–14 min, E2b about 21 min. The T4 time is still unmeasured.

**Losses are per character.** Explicit notation has 5–7 % more characters for the same music, and those characters are easier to predict. **Never compare the loss across notations.** Use bits per note, bars, chord tones, chord gain and listening.

| Run | Best validation loss (step) | Final train / validation |
|---|---|---|
| E2 | 1.137 (2,000) | 0.865 / 1.179, overfits after step 2,000 |
| E2b, 3,000 steps | 1.153 (2,800) | 1.047 / 1.156 |
| E2b, 6,000 steps | 1.107 (6,000) | 0.957 / 1.107 |
| v2 (other notation) | 0.944 (2,900) | 0.892 / 0.951 |
| E2c on the Mac (other notation; discarded, see section 8) | 1.071 (2,900) | 0.950 / 1.089 |

---

## 5. History: versions and verdicts

### E2, first version (21 Nov). Reference, committed
1 key, standard ABC, chords on changes, context 256, temperature 0.8, abcjs piano with its chord track.
- [measured] Bars of the right length: about 0–5 % at step 200, 20–60 % at 1,000, about 90 % at 3,000. Idiomatic ii–V's. No copying (longest shared fragment 15–20 characters; real unseen choruses 14–16).
- [listening] "It works (…) It sounds jazz but not very convincing."

### v2 (21–22 Nov). Reverted
Five changes at once: 12 keys, explicit accidentals, chord in every bar, temperature 0.9, a backing band at playback (walking bass, drums, piano). Later also `generate_over`, which writes a given lead sheet's chords at each bar line so that the model plays only the solo.
- [measured] See section 6. Slightly closer to real solos than E2. `generate_over` on a Bb blues, 30 solos: 50 % chord tones (real solos 50 %), 92 % of bars right, every chorus finished. On rhythm changes: 49 %.
- [listening] "This new version works worse than the first one. It seems the chords don't match the melody." After the fixes: "it's not working".
- [hypothesis] The walking bass plays every root clearly, so mismatches that were always there became easy to hear.
- Archived in `saved_not_working_E2_teacher/` (notebook, checkpoint, data, plan Draft 3c).

### E2b (22 Nov). Built, not committed
E2 with four changes: real choruses to listen to first (plus a blind "real or model?" cell), best checkpoint kept, context 1,024 (85 % of the choruses fit), 12 keys in standard notation.
- [measured] Worse than E2 within the class budget (section 6).
- [listening] 23 Nov, on the 12-key model: "the melody matches the chords worse than the original 1key version". No verdict is recorded for the blind page `E2b_teacher/E2_vs_E2b_listening.html`.
- Recommendation given: keep the examples and the best checkpoint; drop the long context and the 12 keys. Not decided.

### E2c (23 Nov). Built, not committed, waiting for the Debian run
The teacher's definition: "E2c is E2b but writing an accidental on every note", plus the playback fix.
- Rule: `EXPLICIT_ACCIDENTALS = True`. Every flat and sharp is written, so a plain letter is always natural. `=` appears only where normal ABC needs it. **Not confirmed by the teacher:** whether every natural should also get `=`. It would change the training text.
- The sheet music is shown in normal notation (`sheet_abc`); the player gets its own text (`playback_abc`); students can type normal ABC in prompts (`explicit_abc`).
- A short test ran end to end without errors on the Mac. Not tested on Debian/CUDA or Colab.

---

## 6. Measurements

### 6.1 Generated choruses, E2 vs E2b and its ablations
20 samples per prompt, seeds 100–119, temperature 0.8. Cells: Blue Train / Blues for E2 / Ornithology.

| Run | Bars right % | Finished % | Median bars (real 13 / – / 33) | Chord tones % (real 52) |
|---|---|---|---|---|
| E2: 1 key, ctx 256, 3,000 steps | 85 / 90 / 91 | 65 / 80 / 85 | 17 / 20 / 19 | 44 / 42 / 40 |
| A: 12 keys, ctx 256, 3,000 | 60 / 63 / 56 | 65 / 65 / 55 | 26 / 27 / 32 | 40 / 37 / 38 |
| C: 1 key, ctx 1,024, 3,000 | 42 / 42 / 52 | 35 / 50 / 65 | 33 / 35 / 30 | 39 / 37 / 26 |
| E2b: 12 keys, ctx 1,024, 3,000 | 61 / 69 / 68 | 65 / 80 / 65 | 19 / 25 / 28 | 35 / 36 / 38 |
| E2b, 6,000 steps (44 min) | 90 / 91 / 87 | 70 / 80 / 65 | 21 / 25 / 23 | 37 / 34 / 35 |

- Each change alone is worse than E2 at 3,000 steps. Twice the training brings the bars back, but not the chord tones.
- Most wrong bars are one 16th short. Choruses are no closer to the form.

### 6.2 Generated choruses, v2 experiments (context 256)
Different scripts and a slightly different chord-tone measure (real solos 50 %). **Do not mix with 6.1.** 40 samples unless "class" (30 samples, the "Blues for E2" prompt).

| Run, temperature | Bars right % | Finished % | Chord tones % | Notes between rests (real 20) | Chord changes per bar (real 0.71) |
|---|---|---|---|---|---|
| E2, T 0.8 | 87 | – | 41 | 40 | 1.3 |
| E2, class, T 0.8 | 78 | 83 | 42 | 28 | 0.93 |
| std 12 keys, ctx 512, cosine, T 1.0 | 83 | 93 | 35 | 22 | 1.26 |
| xc 12 keys, T 0.8 / 0.9 / 1.0 | 86 / 86 / 85 | 53 / 70 / 90 | 46 / 44 / 42 | 42 / 26 / 22 | 1.05 / 0.92 / 0.98 |
| xc 12 keys, cosine lr, T 0.9 | 92 | 68 | 44 | 23 | 0.95 |
| v2 final, class, T 0.9 | 87 | 83 | 47 | 24 | 0.81 |
| **xc 1 key, step 2,500, T 0.9** | 88 | 85 | 46 | 26 | 0.88 |
| **xc 1 key, step 2,500, class, T 0.9** | 92 | 97 | 46 | 19 | 0.78 |

- **Temperature trade-off:** a lower temperature gives more chord tones, but lines with no rests and fewer finished choruses.
- **xc with 1 key is the best on paper among the 256-context runs**, and it was never offered to the teacher for listening by itself.
- Context 512 did not help at this budget.

### 6.3 Chords without form
- [measured] v2, 30 free "Blues for E2" choruses: 1 had blues changes; many drifted to A♭, D♭ or G♭.
- [measured] E2, free: moving every chord by a fourth hardly changes the chord-tone fit (43 % real chords, 42 % moved; real solos 53 % vs 41 %). **The line follows the key, not the chords.**

### 6.4 Chord gain (teacher forcing). The most reliable measure
**Definition.** Give the model the real validation choruses. Compare how likely it finds the real notes with the real chords, and with every chord symbol moved by k semitones (same qualities, melody unchanged). Only the pitch characters of the notes count. Unit: bits per note, average over k = 1…11. A model that ignores chords has gain 0. "Fourth" = average of k = 5 and 7; it isolates the choice of chord tones, because a fourth away is nearly the same key. A note's pitch costs these models about 3.5–3.9 bits. 189 choruses, 30,069 notes, context 256.

| Step 3,000 | Gain (95 % interval) | Fourth | Bits per note |
|---|---|---|---|
| std, 1 key (E2 recipe) | 0.060 (0.049–0.072) | 0.028 | 3.85 |
| std, 12 keys | 0.029 (0.024–0.034) | 0.016 | 3.65 |
| xc, 1 key | 0.164 (0.135–0.189) | 0.045 | 3.78 |
| xc, 12 keys | 0.088 (0.068–0.105) | 0.021 | 3.55 |
| xc, 12 keys, cosine lr | 0.064 (0.048–0.076) | 0.018 | 3.62 |
| Reference: small log-linear model with pitch numbers, last 2 notes + chord | 0.22 | 0.11 | – |

Growth of the gain during training:

| Step | 200 | 1,000 | 1,500 | 2,000 | 3,000 |
|---|---|---|---|---|---|
| std, 1 key | 0.006 | 0.016 | – | – | 0.060 |
| std, 12 keys | 0.001 | 0.006 | 0.019 | – | 0.029 |
| xc, 1 key | 0.056 | 0.060 | 0.072 | 0.114 | 0.164 |
| xc, 12 keys | 0.031 | 0.034 | 0.055 | – | 0.088 |

At step 3,000 the gain is about 2–3 times larger on the first note after a chord change than on other notes.

### 6.5 Playback (abcjs 6.4.4) [measured]
Two bugs in the player, not in the data:
1. After a note tied over the bar line, it forgets that note's accidental for the rest of the bar.
2. It reads `#` in a chord name wrongly: `7#9` and `9#11` become bare triads, `m11` gets a major 7th, `7alt`/`aug` a natural 5th.

Effect without the fix: E2's data 0.38 % wrong notes (666 of 174,692); explicit 12-key data 0.21 % (of 2,096,304 played). About 1.2 % of the piano chords have a wrong note and 4.6 % miss their 6th or 7th. With `playback_abc`: 0 wrong notes, about 0.01 % of the chords wrong. The model's own solos are hardly affected, so **the bugs do not explain the chord mismatch.**

---

## 7. Why the line ignores the chords

**Ruled out [measured]**
- Wrong ABC or wrong transposition (section 3).
- The player (6.5).
- "Each chord root is seen less often": the 12-key models lose as much on common roots as on rare ones.
- "Chords were learned only in the keys seen": 1-key models keep their gain on the same solos in other keys (0.060 → 0.051, 0.164 → 0.154).
- "It memorized tunes": the gain is not higher on tunes another soloist plays in the training data.

**Leading explanation [hypothesis]: learning order and budget**
- The gain is still rising at step 3,000 in every run. In the 1-key runs it jumps near the validation-loss minimum (about step 2,000), when there is no new melody to learn (each chorus seen about 50–60 times). The 12-key runs are still improving the melody at step 3,000.
- Chord use comes last because the last few notes already tell the model most of it (key, scale, direction), and because `"Eb7"` is three characters with no link to E♭–G–B♭–D♭. Each link must be learned separately.
- In standard notation with 12 keys, a letter's pitch also depends on a `K:` line far back. With context 256 it leaves the view after a few bars.
- **Untested prediction:** a 12-key run of about 9,000 steps catches up. It was started once and stopped at step 100.

**Other contributing facts**
- The dataset is the size of tiny Shakespeare. Improvised lines rarely repeat, and repetition and form make generated music convincing.
- E2 never sees a whole chorus: 256 characters are about 9 bars, a typical chorus is 570 characters.
- A longer context costs about 2× per character on MPS, and 12 long windows per batch are less varied than 64 short ones.
- The loss hides musical errors: a missing 16th or a wrong flat costs only a character or two.

**Cautions about the evidence**
- **Confound:** "explicit accidentals double chord use" comes from the xc runs, which *also* repeat the chord in every bar. Explicit accidentals alone (x) were never trained and measured, except the discarded Mac run of E2c. E2c is the first real test. Some older notes state the effect without this caution.
- Every run has one seed. Generated samples are 20–40 per cell. Trust the chord gain over small differences in generated samples: an early "v2 is worse on chord tones" was within the noise.
- The gains in 6.4 come from experiment runs with the same recipes, not from the E2b or E2c checkpoints themselves. Those have no measured chord gain.
- In xc 1 key, part of the late gain appears while the validation loss is flat or rising. Check that it helps the music.

---

## 8. Known problems and open points

1. **E2 (committed): key parser.** `abc_key_signature` in Helper 2 is wrong for typed keys F#, C#, Cb, D#m, A#m, Abm, C#mix, D#dor, G#dor, so playback has wrong notes there. The data and the dropdown keys are fine. E2c has a correct, general version to copy. Reported to the teacher, not fixed.
2. **E2b: player without the abcjs fix.**
3. **E2b: stale 12-key file.** `parse_key` in the notebook was fixed on 23 Nov (C#dor, F#dor, F#mix), but `E2b_teacher/jazz_abc_12keys.txt` and the checkpoints predate the fix (36 lines differ).
4. **E2c: `=` on every natural?** Not confirmed (section 5).
5. **E2c: no `E2c_teacher/` folder**, no section in the course plan.
6. **E2c Mac run.** It finished 3,000 steps before it was stopped, but the teacher wants Debian only. The file is in a temporary folder, unused. Do not copy it into the project.
7. **Colab T4 dry run** not done for any version (training time, MIDI download link).
8. **Git.** Committed: E2 (a46a154). Untracked: the E2b notebook, `E2b_teacher/`, the E2c notebook, `E2_research_scripts/`, this file. `E2b_teacher/e2b_jazz_gpt.pt` is 72 MB (GitHub warns above 50 MB and refuses above 100 MB).
9. On 24 Nov the teacher also asked another session to run E2 on the Debian machine. No result was recorded.

---

## 9. Next steps, in order

1. **Get the Debian checkpoint** `E2c_files/e2c_jazz_gpt.pt` (run the E2c notebook top to bottom; needs PyTorch, numpy, pandas, matplotlib). Confirm point 4 of section 8 before training if possible.
2. **Measure E2c** with the table of 6.1 and the chord gain of 6.4, next to E2 and E2b (both at context 256 and 1,024). Use the explicit 1-key file for E2c's validation text and the standard one for E2 and E2b. Do not compare losses.
3. **Listening check**: a blind page with E2, E2b, E2c and real choruses, same prompts and seeds, all played with the fixed player.
4. **Separate the factors.** Every cell is one setting in the E2c notebook:

   | Question | Settings |
   |---|---|
   | Explicit accidentals alone | `TRANSPOSE=False`, `block_size=256`, `batch_size=64` (E2 + accidentals). **Most informative single run.** |
   | 12 keys with explicit accidentals, short context | `TRANSPOSE=True`, `block_size=256`, `batch_size=64` |
   | E2c as built | `TRANSPOSE=True`, `block_size=1024`, `batch_size=12` |

5. **Test the learning-order prediction:** a 12-key run of about 9,000 steps or more on the Debian machine, measuring the chord gain every 1,000 steps. If it works, students can load a teacher checkpoint instead of training that long.
6. **Let the teacher hear xc 1 key** (6.2), if chords in every bar are acceptable again.
7. **If the model must follow given chords,** forcing the chords during generation (`generate_over`, archived) already gave 50 % chord tones. The course plan puts this in E3 Part B1; the teacher has not judged it apart from the rest of v2.

**Larger ideas discussed, not tried**
- **Pretrain, then fine-tune:** pretrain the same mini-GPT on a large ABC corpus with melody and chords, fine-tune on jazz in about 10 min, compare with training from scratch. Candidates: PDMX lead sheets (public domain, `no_license_conflict` subset, unknown how many have chord symbols), OpenEWLD (clean, small), iReal Pro charts in `mikeoliphant/JazzStandards` (chords only, teaches form, no licence stated). Folk (IrishMAN) would teach syntax but may make lines diatonic.
- **Bigger tokens:** one token per note, chord or bar line, or BPE. 256 tokens would cover about 25–30 bars, and one token per chord removes the spelling problem of section 7.
- **E4:** fine-tune MuPT (ABC, Apache-2.0) with LoRA on the jazz ABC. Not tested on a T4.
- Little more jazz data exists: DTL1000 (automatic transcriptions, chords not confirmed), FiloSax (research licence only).

**E3–E5:** build on E2's format and checkpoint (first version) until the teacher chooses another. Reuse `playback_abc`.

---

## 10. Files

In `AI_course_for_musicians/Transformers_music/`:

| Path | Content |
|---|---|
| `E2_music_GPT_jazz.ipynb`, `E2_teacher/` | Reference version; checkpoint md5 `ebfe654a…`; executed copy; `jazz_abc.txt` |
| `E2b_music_GPT_jazz.ipynb`, `E2b_teacher/` | E2b; 3,000- and 6,000-step checkpoints; both datasets; blind listening page |
| `E2c_music_GPT_jazz.ipynb` | E2c, ready to train; teacher notes in the last cell |
| `saved_not_working_E2_teacher/` | Archive of v2, including `generate_over` and plan Draft 3c |
| `dataset_tester/` | std and xc 1-key texts |

Course plan: `Transformers_Music_Exercises_Plan.md`, in the `IA_course` folder above the repository (Draft 2b; has E2, "Tried and reverted" and E2b notes, nothing on E2c).

**Scripts: `E2_research_scripts/`.** Copied on 24 Nov from temporary folders on the Mac, unchanged, with the small result files. Not committed.

| Path in `E2_research_scripts/` | Use |
|---|---|
| `e2_eval.py` | **Start here.** One portable script (CUDA, MPS or CPU) for the chord gain and table 6.1; all paths are arguments. Chord gain checked on E2 (gives 0.060); its generation part is not checked at full size. To run it: `python E2_research_scripts/e2_eval.py E2c_music_GPT_jazz.ipynb CKPT.pt best DATA.txt both 1024`, with DATA = the 1-key file in the model's own notation. The last number is the context; it cannot exceed the model's (256 for E2) |
| `chord_gain/` | `chord_swap.py` (the original chord gain), `chord_swap_last.json` (the numbers of 6.4), `why_test.py` and `by_root.py` (the explanations ruled out in 7), `count_ref.py` and `loglin_ref.py` (the reference model) |
| `e2b_eval/` | `eval_ckpt.py` (table 6.1), `make_ab3.py` + `ab_template.html` (blind page), `train_exp.py` (ablations A and C), `run_nb.py` (run a notebook) |
| `e2c_eval/` | `common.py`, `eval_e2c.py`, `gain_e2c.py` (E2 / E2b / E2c comparison, never run; `e2_eval.py` replaces them), `check_c.py` (E2c text = E2b music), `run12.js`, `cmp12.py`, `cmp_arr.py` (what abcjs plays) |
| `v2_experiments/` | `train_v2.py`, `evalq.py`, `abc_analyze.py`, `fit_compare.py`, `gen_forced.py`, `score.py`, and the `eval_*.txt` and `log_*.txt` of 6.2 |
| `data_checks/` | `check_transpose.py` (12-key file), `check_conv.py` and `check_db.py` (notations, database), `jazz_abc.py` (v2's converter, used by `check_db.py`) |

Except `e2_eval.py`, the scripts keep fixed Mac paths (`/private/tmp/…`, `/Volumes/…`), some use the `mps` device, and some load each other across folders. Before running one, change those lines. The 1-key texts they ask for are in the project: `E2_teacher/jazz_abc.txt` (std) and `dataset_tester/jazz_abc_1keys.txt` (xc). The E2b notebook they read (`E2b_executed.ipynb`) is `E2b_teacher/E2b_music_GPT_jazz_executed.ipynb`. Not copied: the experiment checkpoints (`run_*.pt`, `exp_*.pt`), the experiments' 12-key texts, the generated samples and abcjs itself.

**Tools on the Mac.** Conda env `harmony_detector` (torch 2.8 with MPS). Node is broken; to test what abcjs plays, use `/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc` with `load('abcjs-basic.js')` and `ABCJS.parseOnly(abc)[0].setUpAudio({})`. On MPS, pad the generation input to multiples of 128 (E2b and E2c do), or it recompiles at every step.

**What the LLM cannot do:** listen. Every musical judgement needs the teacher.
