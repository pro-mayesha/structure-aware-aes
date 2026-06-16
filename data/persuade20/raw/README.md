# PERSUADE 2.0 raw data (not included in this repository)

Place the official PERSUADE 2.0 corpus files here **after obtaining them under the dataset’s own terms** (see [REALISE / PERSUADE 2.0](https://github.com/scenario-nlp/REALISE)).

Required filenames (must match exactly):

- `persuade_corpus_2.0_train.csv`
- `persuade_corpus_2.0_test.csv`

Example layout:

```text
data/persuade20/raw/persuade_corpus_2.0_train.csv
data/persuade20/raw/persuade_corpus_2.0_test.csv
```

Alternatively, you may place the CSVs in `data/` and create symlinks:

```bash
ln -sf ../../persuade_corpus_2.0_train.csv data/persuade20/raw/persuade_corpus_2.0_train.csv
ln -sf ../../persuade_corpus_2.0_test.csv data/persuade20/raw/persuade_corpus_2.0_test.csv
```

**Do not commit these files to git.** They are listed in `.gitignore`.
