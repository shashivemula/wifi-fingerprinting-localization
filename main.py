"""Command-line entry point for the project scaffold."""

import logging


def main() -> None:
    """Report that the project scaffold is ready for the next phase."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(
        "Project scaffold is ready. Application functionality will be added "
        "in later phases."
    )


if __name__ == "__main__":
    main()
