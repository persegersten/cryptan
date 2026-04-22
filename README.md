# cryptan
Crypto trading model

## Quickstart

### 1. Lokal development (CLI)

```bash
# Konfigurera secrets
cp .env.example .env
# Redigera .env och ersätt "changeme" med riktiga värden

# (Valfritt) Lokal config override — kortare datumspann, annan artifacts-sökväg etc.
cp config/local.yaml.example config/local.yaml
# Justera config/local.yaml efter behov

# Kör pipeline
python -m src.pipeline.train_pipeline --config config/training.yaml
```

### 2. Remote / CI-host (miljövariabler)

```bash
# Sätt krävda env-vars i shell eller CI-systemets secrets store — inga lokala filer behövs
export CRYPTAN_DATA_API_KEY=xxx
export CRYPTAN_DATA_API_SECRET=yyy

# Kör pipeline exakt likadant
python -m src.pipeline.train_pipeline --config config/training.yaml
```

## Konfigurationslager

| Lager | Fil | I git? | Syfte |
|---|---|---|---|
| Bas-config | `config/training.yaml` | ✅ Ja | ML-parametrar, symboler, split, modell |
| Lokal override | `config/local.yaml` | ❌ Nej | Lokala sökvägar, dev-justeringar |
| Secrets | `.env` / OS env vars | ❌ Nej | API-nycklar och känsliga värden |
| Template | `.env.example` | ✅ Ja | Mall med `changeme` som platshållare |

Känsliga värden (`CRYPTAN_DATA_API_KEY`, `CRYPTAN_DATA_API_SECRET`) måste anges som
miljövariabler. Om de saknas eller fortfarande är `changeme` kastas ett tydligt
`EnvironmentError` vid uppstart.
