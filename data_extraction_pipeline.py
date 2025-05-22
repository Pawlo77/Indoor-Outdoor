"""Data Extraction Pipeline for Preparing Initial Dataset."""

import argparse

from src.data_extraction_pipeline import prepare_initial_set


def main():
    """Main function to parse arguments and prepare the initial dataset."""

    parser = argparse.ArgumentParser(description="Prepare the initial dataset.")
    parser.add_argument(
        "--start", type=int, default=0, help="Starting part id (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Ending part id (default: 330)"
    )
    parser.add_argument(
        "--remove-original-files",
        action="store_true",
        help="Flag to remove original files after processing",
        default=False,
    )
    args = parser.parse_args()

    prepare_initial_set(
        start_id=args.start,
        end_id=args.end,
        remove_original_files=args.remove_original_files,
    )


if __name__ == "__main__":
    main()
