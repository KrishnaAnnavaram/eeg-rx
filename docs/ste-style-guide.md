# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **subject** | One patient with one recording and one label | patient (in code and results), case |
| **recording** | The EEG of one subject: 19 channels at one sampling rate | file (for the signal), trace |
| **epoch** | One non-overlapping 15 s part of a recording | segment, window (for 15 s) |
| **window** | One 25-sample part of a channel that gives one pattern code | epoch |
| **pattern code** | A number from 0 to 31 from the 5x5 matrix rule | LBP value, bit string |
| **pattern histogram** | The normalised count of the 32 pattern codes of one channel | LBP feature vector |
| **band power** | The power of a channel in one frequency band | energy, amplitude |
| **responder** | A subject with label 1 | positive patient |
| **non-responder** | A subject with label 0 | negative patient |
| **subject score** | The mean epoch probability of one subject | prediction, risk |
| **outer fold** | One test group of subjects in the nested evaluation | split, partition |
| **inner fold** | One validation group inside the training subjects of an outer fold | validation split |
| **nested evaluation** | Outer subject-wise folds with an inner selection loop | cross-validation (alone) |
| **leaky protocol** | Feature selection on all data and epoch-wise folds. Used only for comparison | standard CV, baseline |
| **permutation test** | The nested evaluation repeated with labels shuffled between subjects | randomisation test |
| **model card** | The file `MODEL_CARD.md` that states the use, the data and the limits | documentation |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read recordings and labels from files |
| **filter** | Apply the band-pass filter (0.5 Hz to 45 Hz) |
| **epoch** | Cut a recording into 15 s epochs |
| **reject** | Remove an epoch with a peak-to-peak value above the limit |
| **extract** | Calculate the features of each epoch |
| **select** | Keep the top features inside a training fold |
| **evaluate** | Run the nested evaluation and calculate the subject-level metrics |
| **aggregate** | Calculate the subject score from the epoch probabilities |
| **predict** | Calculate the subject score of a new recording with a trained model |
