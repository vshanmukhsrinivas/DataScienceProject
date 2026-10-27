
import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FIELD_MAP = {
    "product/productId": "ProductId",
    "review/userId": "UserId",
    "review/profileName": "ProfileName",
    "review/helpfulness": "HelpfulnessRaw",
    "review/score": "ScoreRaw",
    "review/time": "TimeRaw",
    "review/summary": "Summary",
    "review/text": "Text",
}

COLUMNS = [
    "ProductId", "UserId", "ProfileName",
    "HelpfulnessRaw", "HelpfulnessNumerator",
    "HelpfulnessDenominator", "ScoreRaw", "Score",
    "TimeRaw", "ReviewDateUTC", "Summary", "Text",
    "ReviewLengthChars", "SummaryLengthChars",
]


def open_source(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_record(lines, report):
    record = {col: "" for col in FIELD_MAP.values()}
    current_key = None
    seen = set()

    for line in lines:
        matched_key = None
        value = None

        for key in FIELD_MAP:
            prefix = key + ": "
            if line.startswith(prefix):
                matched_key = key
                value = line[len(prefix):]
                break

        if matched_key is not None:
            column = FIELD_MAP[matched_key]

            if matched_key in seen:
                report["repeated_fields"][matched_key] += 1
                record[column] += "\n" + value
            else:
                record[column] = value
                seen.add(matched_key)

            current_key = matched_key

        elif current_key is not None:
            # Preserve continuation lines instead of discarding them.
            column = FIELD_MAP[current_key]
            record[column] += "\n" + line
            report["continuation_lines_attached"] += 1

        else:
            report["orphan_lines"] += 1
            if len(report["orphan_examples"]) < 10:
                report["orphan_examples"].append(line[:200])

    return record


def enrich(record, report):
    row = dict(record)

    # Parse helpfulness while retaining the original value.
    row["HelpfulnessNumerator"] = ""
    row["HelpfulnessDenominator"] = ""

    raw = row["HelpfulnessRaw"].strip()
    if raw:
        try:
            numerator, denominator = map(int, raw.split("/", 1))
            row["HelpfulnessNumerator"] = numerator
            row["HelpfulnessDenominator"] = denominator

            if numerator < 0 or denominator < 0:
                report["negative_helpfulness_counts"] += 1
            if numerator > denominator:
                report["numerator_gt_denominator"] += 1
        except ValueError:
            report["helpfulness_parse_failures"] += 1

    # Preserve raw score; create numeric score separately.
    row["Score"] = ""
    if row["ScoreRaw"].strip():
        try:
            row["Score"] = float(row["ScoreRaw"])
            if row["Score"] not in (1, 2, 3, 4, 5):
                report["unexpected_scores"] += 1
        except ValueError:
            report["score_parse_failures"] += 1

    # Preserve raw timestamp; add readable UTC date.
    row["ReviewDateUTC"] = ""
    if row["TimeRaw"].strip():
        try:
            timestamp = int(row["TimeRaw"])
            row["ReviewDateUTC"] = datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).date().isoformat()
        except (ValueError, OverflowError, OSError):
            report["time_parse_failures"] += 1

    row["ReviewLengthChars"] = len(row["Text"])
    row["SummaryLengthChars"] = len(row["Summary"])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "records_written": 0,
        "continuation_lines_attached": 0,
        "orphan_lines": 0,
        "orphan_examples": [],
        "repeated_fields": Counter(),
        "missing_required_fields": Counter(),
        "missing_optional_fields": Counter(),
        "helpfulness_parse_failures": 0,
        "negative_helpfulness_counts": 0,
        "numerator_gt_denominator": 0,
        "score_parse_failures": 0,
        "unexpected_scores": 0,
        "time_parse_failures": 0,
        "note": "Parsing only; no intentional row filtering or imputation.",
    }

    required = [
        "ProductId", "UserId", "HelpfulnessRaw",
        "ScoreRaw", "TimeRaw", "Text"
    ]
    optional = ["ProfileName", "Summary"]

    with open_source(args.input) as source, gzip.open(
        args.output, "wt", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS)
        writer.writeheader()
        lines = []

        def flush(record_lines):
            if not record_lines:
                return

            record = parse_record(record_lines, report)
            writer.writerow(enrich(record, report))
            report["records_written"] += 1

            for field in required:
                if record[field] == "":
                    report["missing_required_fields"][field] += 1
            for field in optional:
                if record[field] == "":
                    report["missing_optional_fields"][field] += 1

        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if line.strip():
                lines.append(line)
            elif lines:
                flush(lines)
                lines = []

        flush(lines)

    for key in (
        "repeated_fields",
        "missing_required_fields",
        "missing_optional_fields",
    ):
        report[key] = dict(report[key])

    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Records written:", f"{report['records_written']:,}")
    print("Continuation lines preserved:",
          report["continuation_lines_attached"])
    print("Orphan lines:", report["orphan_lines"])
    print("Repeated known fields:", sum(report["repeated_fields"].values()))
    print("Numerator > denominator:", report["numerator_gt_denominator"])
    print("Output:", args.output)
    print("Report:", args.report)


if __name__ == "__main__":
    main()
