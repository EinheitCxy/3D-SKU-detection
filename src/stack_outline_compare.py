"""Compare a saved candidate against a saved human reference."""
import argparse
import json
from pathlib import Path
from utils.outline_exchange import compare_outlines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_outlines(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f'outline comparison failed: {error}\n')


if __name__ == '__main__':
    main()
