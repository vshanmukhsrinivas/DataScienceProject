# Amazon Fine Food Reviews — Data Dictionary

## Dataset overview

- **Source:** Stanford SNAP Amazon Fine Food Reviews
- **Parsed observations:** 568,454 reviews
- **Parsed columns:** 14
- **Unit of observation:** One review record in the downloaded dataset
- **Raw data:** `data/raw/finefoods.txt`
- **Parsed data:** `data/processed/finefoods_parsed.csv.gz`

## Variables

| Variable | Type | Description | Role / Notes |
|---|---|---|---|
| `ProductId` | String | Identifier of the reviewed product | Categorical identifier |
| `UserId` | String | Identifier of the reviewer | Categorical identifier; may repeat |
| `ProfileName` | String | Display name of the reviewer | Optional metadata; contains missing values |
| `HelpfulnessRaw` | String | Original helpfulness count, such as `3/5` | Preserved source representation |
| `HelpfulnessNumerator` | Integer | Number of helpful votes | Count variable |
| `HelpfulnessDenominator` | Integer | Total helpfulness votes represented in the source | Count variable; zero means no votes |
| `ScoreRaw` | Numeric | Original rating value as parsed | Retained for traceability |
| `Score` | Numeric | Rating on the 1–5 scale | Main structured outcome candidate |
| `TimeRaw` | Integer | Original Unix timestamp | Retained for traceability |
| `ReviewDateUTC` | String / date | Human-readable review date derived from the timestamp | Temporal feature |
| `Summary` | String | Short review headline | Text feature; contains missing values |
| `Text` | String | Full review text | Main text feature |
| `ReviewLengthChars` | Integer | Number of characters in the review text | Derived feature |
| `SummaryLengthChars` | Integer | Number of characters in the summary | Derived feature |

## Data-quality notes

1. The parsed dataset contains 568,454 records.
2. `ProfileName` has 26 missing values and `Summary` has 27.
3. There are 281 additional occurrences of exact duplicate rows.
4. Two records have helpfulness numerators greater than their denominators.
5. There are 270,052 records with a zero helpfulness denominator. A helpfulness rate is undefined for these records.
6. Repeated product and user identifiers are expected in review data and do not independently establish duplicate observations.
7. Duplicate handling, invalid helpfulness handling, and any feature transformations must be documented in the cleaning report.

## Derived-variable conventions

For valid vote counts with a positive denominator:

`HelpfulnessRate = HelpfulnessNumerator / HelpfulnessDenominator`

The rate is undefined when the denominator is zero or when the counts are invalid. The original vote counts should be preserved.

Review and summary lengths are character counts, not word counts.
