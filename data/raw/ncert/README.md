# NCERT Raw PDF Drop Zone

Place Amit's NCERT PDFs here before running the admin ingestion pipeline.
Raw PDFs are treated as source material and are intentionally not committed to Git.

Expected folder pattern:

```text
backend/data/raw/ncert/
  class_10/
    science/
      chapter_01_chemical_reactions.pdf
  class_11/
    chemistry/
      chapter_01_some_basic_concepts_of_chemistry.pdf
  class_12/
    physics/
      chapter_02_electrostatic_potential.pdf
```

Naming rules:

- Use `class_<number>` for class folders.
- Use one subject folder per subject, for example `science`, `maths`, `physics`, `chemistry`, `biology`, `history`.
- Start chapter files with `chapter_<number>_`.
- Keep one complete chapter per PDF.

The automated Class 11 Chemistry download has a checked-in nine-chapter source
map (`kech1` Units 1-6, then `kech2` Units 7-9). New files use canonical names
such as `chapter_03_classification_of_elements_and_periodicity_in_properties.pdf`.
On resume, any valid existing `chapter_NN*.pdf` is reused, so manually supplied
descriptive Chapter 1/2 files are not downloaded again under generic names.

Pipeline rule:

- Ingest PDFs first.
- Generate or import concept JSON second.
- Review validation report.
- Approve or publish only after coverage is acceptable.
- Study Lab retrieval uses only `approved` or `published` chapters from the pipeline.
