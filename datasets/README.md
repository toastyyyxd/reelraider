# ReelRaider Culturally-Aware Movie Search System

**Complete movie recommendation system with semantic search and cultural localization**

## 🚀 Quick Start

See [`QUICK_REFERENCE.md`](./QUICK_REFERENCE.md) for complete documentation.

```bash
# 1. Build pipeline (one-time setup)
python -m datasets.ids
python -m datasets.ratings  
python -m datasets.omdb
python -m datasets.scalar_norm
python -m datasets.embedding
python -m datasets.culturally_aware_embedding

# 2. Search movies
python -m datasets.query_engine "Hong Kong crime thriller"
python -m datasets.query_engine --similar-to "The Godfather" --preset cultural
```

## 📁 Key Components

### Production Scripts
- **`query_engine.py`** - Main search interface with cultural presets
- **`culturally_aware_embedding.py`** - 128-dimension localization embedding system  
- **`runtime_weighted_search.py`** - Core search engine with dynamic weights
- **Data pipeline**: `ids.py`, `ratings.py`, `omdb.py`, `scalar_norm.py`, `embedding.py`

### Generated Data (in `dist/`)
- **`movies_processed_sn.parquet`** - 25,096 movies with metadata
- **`embedding_components.npz`** - Plot/genre/localization/popularity embeddings
- **`culturally_aware_model.json`** - Model configuration
- **`movies_culturally_aware.index`** - FAISS search index

## 🌍 Key Features

- **Natural language search**: "Space opera about rebellion against empire"
- **Cultural awareness**: Localized recommendations for Hong Kong, Japanese, French cinema etc.  
- **Personalized recommendations**: Based on movies users have watched
- **Runtime weight tuning**: No index rebuilding required for adjustments
- **Proven results**: 8/10 Hong Kong movies for HK-specific queries

## 📜 Legal & Data Sources

- **OMDb API**: Movie metadata under Creative Commons BY-NC 4.0 license  
- **IMDb data**: Used only for initial filtering and ID mapping
- **Outputs**: All processed data generated in `dist/` directory
- **License**: See [LICENSE](../LICENSE) at repository root

## 🛠️ Development

For API integration, use the `MovieQueryEngine` class from `query_engine.py`. The system supports runtime weight adjustment without requiring index rebuilds.

---

**📖 Complete documentation**: [`QUICK_REFERENCE.md`](./QUICK_REFERENCE.md)
