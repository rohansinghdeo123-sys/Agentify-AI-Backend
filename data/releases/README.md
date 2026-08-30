# Production content releases

`ncert_class_11_chemistry_v1.json.gz` is the deployment snapshot produced from
the quality-gated Class XI Chemistry ingestion run. It contains chapters 1–9,
their extracted pages, grouped concepts, retrieval chunks, and normalized
semantic embeddings.

The application verifies the bundle SHA-256, every per-chapter digest, source
mapping, coverage gate, unique IDs, and the embedding contract before a remote
database is changed. Restores preserve an existing chapter row ID, replace only
that chapter's derived content when required, and create one auditable
`content_release_restore` ingestion job per release digest.

Regenerate only from a reviewed database:

```bash
python scripts/content_release.py export \
  --embedding-model gemini-embedding-001 \
  --embedding-dimensions 3072 \
  --embedding-provider google-generative-language \
  --embedding-endpoint-host generativelanguage.googleapis.com
python scripts/content_release.py verify
```

Remote deployments restore the bundle during application startup after schema
migrations. Local SQLite runs skip it unless `BUNDLED_CONTENT_BOOTSTRAP=true`.
