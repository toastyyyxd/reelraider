# OMDB API Response & Casting Schema

This document describes the OMDB API response fields used in `OmdbAggregator` and how each field is cast or transformed in the resulting dataset.

## OMDB API Fields and Casting

| Output Field | OMDB API Field | Type / Cast | Notes |
|--------------|---------------|-------------|-------|
| tid          | (input)       | int         | IMDb title ID (from input) |
| title        | Title         | str         | Movie title |
| year         | Year          | int         | Parsed with `_parse_int` |
| rated        | Rated         | str         | Rating (e.g., "PG-13") |
| runtime      | Runtime       | int         | Minutes, parsed with `_parse_duration` (handles "2h 13min", "173 min", etc.) |
| genre        | Genre         | List[str]   | Split on "," |
| director     | Director      | List[str]   | Split on "," |
| actors       | Actors        | List[str]   | Split on "," |
| plot         | Plot          | str         | Movie plot summary |
| language     | Language      | List[str]   | Split on "," |
| country      | Country       | List[str]   | Split on "," |
| awards       | Awards        | List[str]   | Split on "&" |
| poster       | Poster        | str         | Poster URL |
| rating       | imdbRating    | float       | Parsed with `_parse_float` |
| votes        | imdbVotes     | int         | Parsed with `_parse_int` |
| metascore    | Metascore     | int         | Parsed with `_parse_int` |

Missing values originally denoted with "N/A" are set to `None` in python.

## Example OMDB API Response

```
{
  "Title": "The Shawshank Redemption",
  "Year": "1994",
  "Rated": "R",
  "Released": "14 Oct 1994",
  "Runtime": "2h 22min",
  "Genre": "Drama",
  "Director": "Frank Darabont",
  "Writer": "Stephen King, Frank Darabont",
  "Actors": "Tim Robbins, Morgan Freeman, Bob Gunton",
  "Plot": "Two imprisoned men bond over a number of years...",
  "Language": "English",
  "Country": "USA",
  "Awards": "Nominated for 7 Oscars. Another 21 wins & 43 nominations.",
  "Poster": "https://...jpg",
  "imdbRating": "9.3",
  "imdbVotes": "2,500,000",
  "Metascore": "80",
  ...
}
```

## Notes
- Fields not present in the OMDB response or with value "N/A" are cast to default values (empty list, 0, -1, or empty string as appropriate).
- The `awards` field is split on `&` to separate multiple awards.
- The `runtime` field is robustly parsed to handle hours and minutes.
