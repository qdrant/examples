from pathlib import Path

import typer
from helpers.markdown import NotebookToHugoMarkdownConverter, ParsingException
from loguru import logger

MAIN_DIR = Path(__file__).resolve().parent.parent

# TODO: allow passing a list of notebooks to convert, not just all


def main(
    overwrite: bool = False,
):
    converter = NotebookToHugoMarkdownConverter()

    for notebook_path in MAIN_DIR.glob("**/*.ipynb"):
        relative_notebook_dir = notebook_path.relative_to(MAIN_DIR).parent.parent

        # Normalized filename is used to create the output file and it will be a part of the URL
        new_filename = converter.normalize_filename(notebook_path.stem)

        # Output directory mimics the structure of the landing_page repo
        output_dir = (
            MAIN_DIR
            / ".dist"
            / "qdrant-landing"
            / "content"
            / "documentation"
            / str(relative_notebook_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_md_file = output_dir / f"{new_filename}.md"

        # Assets are stored to mimic the landing_page repo structure as well
        assets_dir = (
            MAIN_DIR
            / ".dist"
            / "qdrant-landing"
            / "static"
            / "documentation"
            / str(relative_notebook_dir)
            / new_filename
        )
        assets_dir.mkdir(parents=True, exist_ok=True)

        if output_md_file.exists() and not overwrite:
            logger.info(
                "Skipping {} as {} already exists",
                notebook_path.relative_to(MAIN_DIR),
                output_md_file.relative_to(MAIN_DIR),
            )
            continue

        logger.info(
            "Converting {} to {}", notebook_path.relative_to(MAIN_DIR), output_md_file
        )

        try:
            converter.convert(notebook_path, output_md_file, assets_dir)
        except ParsingException as e:
            logger.error("Could not convert {}: {}", notebook_path, e)


if __name__ == "__main__":
    typer.run(main)
