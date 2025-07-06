# Ratings Processing & Output Schema

This document describes how we process ratings and statistics for movies and TV movies, and details the resulting columns in the ratings dataset. All data is sourced solely from OMDB.

## Ratings Processing Overview


The ratings dataset is constructed by normalizing and combining user and critic statistics from OMDB. The process includes:

1. **Normalization**: User votes, user ratings, and Metascores are normalized to a 0-1 scale.
2. **Mean Calculation**: Global means for votes, ratings, and metascores are computed for Bayesian adjustment.
3. **Bayesian Adjustment**: User ratings are adjusted using a Bayesian formula to account for vote count reliability.
4. **Metascore Fallback**: If user ratings or votes are missing, Metascore is used as a fallback.
5. **Final Rating**: If both Bayesian rating and Metascore are available, their average is used. If only one is available, it is used as the final rating.
6. **Controversy Score**: Measures the disagreement between user and critic ratings, weighted by vote count.
7. **Filtering**: Rows without any rating information are dropped.

## Relevant Output Columns & Schema

| Column            | Type      | Description                                                                                 |
|-------------------|-----------|---------------------------------------------------------------------------------------------|
| tid               | int       | OMDB title ID                                                                               |
| votes             | float     | Number of OMDB user votes (0 replaced with None, normalized for calculations)               |
| rating            | float     | OMDB user rating (original 1-10 scale, normalized to 0-1 as `n_rating`)                     |
| metascore         | float     | Metascore (original 0-100 scale, normalized to 0-1 as `n_metascore`)                        |
| n_rating          | float     | User rating normalized to 0-1                                                               |
| n_metascore       | float     | Metascore normalized to 0-1                                                                 |
| bayes_rating      | float     | Bayesian-adjusted user rating (0-1), or None if insufficient data                            |
| final_rating      | float     | Final rating (0-1): average of bayesian and metascore if both, else whichever is available   |
| controversy_score | float     | Disagreement between user and critic ratings, weighted by vote count (higher = more divisive)|

## Example Row

| tid      | votes   | rating | metascore | n_rating | n_metascore | bayes_rating | final_rating | controversy_score |
|----------|---------|--------|-----------|----------|-------------|--------------|--------------|------------------|
| 1234567  | 25000   | 8.2    | 75        | 0.82     | 0.75        | 0.81         | 0.78         | 0.03             |

## Notes
- All columns are float except `tid` (int).
- Missing values are set to `None`.
- `votes` is set to `None` if 0, to avoid division by zero.
- `final_rating` is always present in the output; rows without any rating are dropped.
- `controversy_score` is only present if both user and critic ratings are available.
- All ratings are normalized to 0-1 for consistency.

