# cryptan
Crypto trading model

## Quickstart

### 1. Lokal development (CLI)

```bash
# Konfigurera secrets om datakällan kräver autentisering
cp .env.example .env
# Redigera .env och ersätt "changeme" med riktiga värden vid behov

# (Valfritt) Lokal config override — kortare datumspann, annan artifacts-sökväg etc.
cp config/local.yaml.example config/local.yaml
# Justera config/local.yaml efter behov
# Datumspann anges som dag-offsets från idag UTC: -1 = igår, 0 = idag.

# Kör pipeline
python -m src.pipeline.train_pipeline --config config/training.yaml
```

### 2. Remote / CI-host (miljövariabler)

```bash
# Sätt env-vars i shell eller CI-systemets secrets store vid behov — inga lokala filer behövs
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

Känsliga värden (`CRYPTAN_DATA_API_KEY`, `CRYPTAN_DATA_API_SECRET`) ska anges som
miljövariabler när datakällan kräver dem. Om de saknas, är tomma eller fortfarande är
`changeme` loggas en varning och pipeline fortsätter med `changeme`.
