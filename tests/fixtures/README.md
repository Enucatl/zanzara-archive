# Synthetic fixtures

Fixtures in this directory must be deterministic, small and synthetic. They
may represent contract records, timestamps, vectors or short generated text,
but must not contain private archive audio, credentials, identities, model
weights or local database state. Tests that require real model output or
external services belong behind the appropriate pytest marker and private
evidence storage.
