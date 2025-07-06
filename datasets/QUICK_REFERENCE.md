# ReelRaider Culturally-Aware Movie Search

> **Complete culturally-aware movie recommendation system with 128-dimension localization embeddings and runtime weight tuning**

## 🚀 Quick Start

### 1. Build Pipeline (One Time Setup)
```bash
# From project root - run these in order
python -m datasets.ids           # Process IMDb IDs
python -m datasets.omdb          # Fetch plot summaries via OMDb API
python -m datasets.ratings       # Fine-tune ratings data  
python -m datasets.scalar_norm   # Normalize popularity scores
python -m datasets.embedding     # Generate plot/genre embeddings
python -m datasets.culturally_aware_embedding  # Build 128-dim cultural embeddings ⭐
```

### 2. Search Movies
```bash
# Natural language search
python -m datasets.query_engine "Hong Kong crime thriller about undercover cops"

# Personalized recommendations (NEW!)
python -m datasets.query_engine --similar-to "Infernal Affairs"

# Cultural preferences
python -m datasets.query_engine "samurai movies" --countries Japan --languages Japanese
```

## 🎯 Core Features

### Natural Language Search
- **Plot-based**: "Space opera about rebellion against empire"
- **Genre-aware**: "French new wave romantic drama"  
- **Cultural context**: "Japanese samurai honor revenge"
- **Hybrid queries**: "Hong Kong triad crime undercover police infiltration"

### Culturally-Aware Recommendations  
- **Auto-adaptation**: Recommending "Zatoichi" → Japanese samurai films
- **Context extraction**: "The Godfather" → American-Italian crime family films
- **128-dimension localization**: Effective cultural filtering without semantic loss

### Tuned Preset Configurations
- **Balanced** (plot=0.5, genre=0.25, localization=0.1, popularity=0.15): General purpose
- **Popular** (plot=0.4, genre=0.2, localization=0.2, popularity=0.2): Mainstream focus  
- **Cultural** (plot=0.44, genre=0.13, localization=0.1, popularity=0.33): Regional cinema ⭐
- **Niche** (plot=0.6, genre=0.3, localization=0.05, popularity=0.05): Art house/indie films

## 🌍 Cultural Search Examples

### Hong Kong Cinema (Proven Results)
```bash
python -m datasets.query_engine "Hong Kong triad crime undercover police infiltration" \
  --countries "Hong Kong" --languages "Cantonese,Chinese" --preset cultural

# Top Results: Infernal Affairs II, As Tears Go By, A Better Tomorrow, 
#              Police Story, Hard Boiled, Election, Infernal Affairs, City on Fire
```

### Japanese Cinema
```bash
python -m datasets.query_engine "Japanese samurai" \
  --countries Japan --languages Japanese --preset cultural

# Results: The Blind Swordsman: Zatoichi, Shogun Assassin, Ninja Scroll, Kagemusha
```

### Personalized Recommendations
```bash
# Based on Hong Kong crime film
python -m datasets.query_engine --similar-to "Infernal Affairs II" --preset cultural
# → New Police Story, Infernal Affairs III, Infernal Affairs, A Better Tomorrow

# Based on Japanese film  
python -m datasets.query_engine --similar-to "Zatoichi" --preset cultural
# → Hara-Kiri, Kagemusha, Azumi, Rurouni Kenshin

# Based on American crime film
python -m datasets.query_engine --similar-to "The Godfather" --preset cultural  
# → The Many Saints of Newark, A Bronx Tale, The Godfather Part III
```

## 📁 System Architecture

### Production Scripts
- **`query_engine.py`** - Main CLI interface with cultural presets ⭐
- **`runtime_weighted_search.py`** - Core search engine with dynamic weights
- **`culturally_aware_embedding.py`** - 128-dim embedding system
- **`utils.py`** - Common utilities

### Generated Data  
- **`dist/movies_processed_sn.parquet`** - 25,096 movies with metadata
- **`dist/embedding_components.npz`** - Separate plot/genre/localization/popularity embeddings
- **`dist/culturally_aware_model.json`** - Model configuration
- **`dist/movies_culturally_aware.index`** - FAISS search index

### Key Innovations
- **128-dimension localization vectors** (vs original 32-dim) for effective cultural awareness
- **Runtime weight adjustment** - no index rebuilding required for tuning
- **Semantic + cultural balance** - maintains ML accuracy while respecting cultural preferences
- **Auto-cultural extraction** - recommendations inherit cultural context from source movies

## 🔧 Advanced Usage

### Custom Weight Tuning
```bash
# Manual weight configuration
python -m datasets.query_engine "your query" \
  --plot-weight 0.5 --genre-weight 0.2 \
  --localization-weight 0.2 --popularity-weight 0.1

# Disable cultural awareness (not recommended)
python -m datasets.query_engine "your query" --no-runtime-weights
```

### Filters and Search Options
```bash
# Advanced filtering
python -m datasets.query_engine "romantic comedy" \
  --min-rating 7.0 --min-votes 10000 \
  --year-from 2000 --year-to 2020 \
  --preset balanced -n 15

# Show database statistics
python -m datasets.query_engine --stats
```

### Rebuilding System
```bash
# Rebuild embeddings (after data changes)
python -m datasets.culturally_aware_embedding --force

# Check embedding dimensions
python -c "
import numpy as np
data = np.load('datasets/dist/embedding_components.npz')
print('Plot:', data['plot_embeddings'].shape)
print('Localization:', data['localization_embeddings'].shape)
"
```

## 🎬 Presentation Demo Commands

### Natural Language Search
```bash
# Hong Kong crime cinema
python -m datasets.query_engine "Hong Kong triad crime undercover police infiltration" \
  --countries "Hong Kong" --languages "Cantonese,Chinese" --preset cultural -n 5

# Japanese samurai films  
python -m datasets.query_engine "Japanese samurai honor revenge" \
  --countries Japan --languages Japanese --preset cultural -n 5

# French art house cinema
python -m datasets.query_engine "French new wave romantic drama" \
  --countries France --languages French --preset niche -n 5
```

### Personalized Recommendations
```bash  
# Crime thriller fan
python -m datasets.query_engine --similar-to "Infernal Affairs" --preset cultural -n 5

# Samurai film fan
python -m datasets.query_engine --similar-to "Zatoichi" --preset cultural -n 5

# Classic Hollywood fan
python -m datasets.query_engine --similar-to "The Godfather" --preset cultural -n 5
```

## 🛠️ Integration Notes

### For Web API Development
- Use `MovieQueryEngine` class from `query_engine.py`
- Leverage `SearchRequest` and `SearchResult` dataclasses
- Runtime weight adjustment requires no index rebuilding
- Cultural presets are production-ready

### Performance
- **Search time**: ~1-2 seconds per query (includes embedding generation)
- **Memory usage**: ~2GB for full 25k movie dataset  
- **Storage**: ~500MB for all embeddings and indices
- **Scalability**: FAISS index supports millions of movies

### Cultural Localization Coverage
- **403 unique country/language combinations** in 128-dimensional space
- **Effective for major film industries**: Hollywood, Bollywood, Hong Kong, Japan, France, etc.
- **Semantic + cultural balance**: Avoids pure country filtering while maintaining relevance
