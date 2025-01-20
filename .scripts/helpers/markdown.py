import base64
import itertools
import mimetypes
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse

import frontmatter
import magic
import markdown_it.token
import requests
from loguru import logger
from markdown_it import MarkdownIt
from mdformat.renderer import MDRenderer
from mdformat_frontmatter import plugin as mdformat_front_matter_plugin
from mdformat_tables import plugin as tables_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from nbconvert import MarkdownExporter

from .plugins.word_count import word_count_plugin

MAIN_DIR = Path(__file__).parent.parent.parent


@dataclass
class ParsedMarkdown:
    """
    A class for keeping a consistent representation of a Markdown document.
    """

    raw_content: str
    tokens: list[markdown_it.token.Token]
    metadata: dict | None = field(default_factory=dict)
    env: dict | None = field(default_factory=dict)
    resources: dict | None = field(default_factory=dict)

    def iter_tokens(self) -> Generator[markdown_it.token.Token, None, None]:
        """
        Iterate over all the tokens in the document, including the frontmatter, if there is metadata.
        :return: A generator of tokens.
        """
        if self.metadata:
            post = frontmatter.Post("", **self.metadata)
            doc_frontmatter = frontmatter.dumps(post).strip().strip("-").strip()
            yield markdown_it.token.Token(
                type="front_matter",
                tag="",
                nesting=0,
                content=doc_frontmatter,
                markup="---",
                block=True,
                hidden=True,
            )
        for token in self.tokens:
            yield token


class ParsingException(Exception):
    """
    An exception that is raised when the parsing of the Markdown content fails.
    """


class NotebookToHugoMarkdownConverter:
    """
    A converter that converts Jupyter notebooks to Hugo markdown files, including the frontmatter.
    It additionally performs some formatting fixes to the generated markdown.
    """

    def __init__(self):
        self._exporter = MarkdownExporter()
        self._md = (
            MarkdownIt(
                "gfm-like",
                {"parser_extension": [mdformat_front_matter_plugin, tables_plugin]},
                renderer_cls=MDRenderer,
            )
            .use(front_matter_plugin)
            .use(word_count_plugin)
            .enable("front_matter")
            .enable("table")
        )

    def normalize_filename(self, filename: str) -> str:
        """
        Normalize the filename to be compatible with the landing page conventions.
        :param filename: The filename to normalize.
        :return: The normalized filename.
        """
        normalized_filename = filename.lower()
        normalized_filename = unicodedata.normalize("NFKD", normalized_filename)

        # Normalize different kinds of hyphens
        hyphens = ["-", "–", "—", "―"]
        for hyphen in hyphens:
            normalized_filename = normalized_filename.replace(hyphen, "-")

        # Replace all whitespace and underscores with single hyphens
        normalized_filename = re.sub(r"\s+", "-", normalized_filename)
        normalized_filename = re.sub(r"_+", "-", normalized_filename)
        normalized_filename = re.sub("-+", "-", normalized_filename)

        # Remove all non-alphanumeric characters
        normalized_filename = re.sub(r"[^a-z0-9-]", "", normalized_filename)

        return normalized_filename

    def convert(
        self, notebook_path: Path, output_path: Path, assets_dir: Path | None = None
    ):
        """
        Run the conversion process for a selected Jupyter notebook and save the output to a markdown file.
        :param notebook_path: The path to the Jupyter notebook to convert.
        :param output_path: The path to save the converted markdown file.
        :param assets_dir: The directory where the assets should be saved. If None, the assets are not saved.
        """
        if not notebook_path.exists():
            raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

        with open(notebook_path) as fp:
            body, resources = self._exporter.from_file(fp)

        # Parse the document and pass it through all the processing steps
        env_dict = {}
        parsed_markdown = ParsedMarkdown(
            body,
            self._md.parse(body, env_dict),
            env=env_dict,
            resources=resources,
        )

        # Perform additional modifications so the Markdown is compatible with Hugo
        parsed_markdown = self._separate_code_blocks(parsed_markdown)
        parsed_markdown = self._remove_empty_code_blocks(parsed_markdown)

        # Add the frontmatter to the Markdown content
        parsed_markdown = self._add_frontmatter(notebook_path, parsed_markdown)

        # Save the assets to the specified directory
        if assets_dir is not None:
            parsed_markdown.tokens = self._process_assets(
                parsed_markdown, parsed_markdown.tokens, notebook_path, assets_dir
            )

        # Render the finalized Markdown content to a file. The MDRenderer will take care of the formatting.
        with open(output_path, "w") as f:
            rendered_md = self._md.renderer.render(
                list(parsed_markdown.iter_tokens()),
                self._md.options,
                parsed_markdown.env,
            )
            f.write(rendered_md)
        logger.info(f"Converted notebook to markdown: {output_path}")

        # Generate the _index.md file, if it doesn't exist, so the new markdown file is included in the menu
        self._write_index_file(output_path.parent)

    def _separate_code_blocks(self, markdown: ParsedMarkdown) -> ParsedMarkdown:
        """
        Separate code blocks in the markdown to ensure they are rendered correctly by Hugo.
        If there are multiple code blocks of the same language in a row, Hugo will render them
        as tabs, but they won't be functional. This function ensures that each such a pair of
        code blocks is separated by a horizontal rule.
        """
        new_tokens = []
        for first, second in itertools.pairwise(markdown.tokens):
            new_tokens.append(first)
            if (
                first.type == "fence"
                and second.type == "fence"
                and first.info == second.info
            ):
                new_tokens.append(
                    markdown_it.token.Token(
                        type="html_block", tag="", nesting=0, content="<hr />"
                    )
                )

        # The last token won't be added in a loop, so we need to add it manually
        if len(markdown.tokens) > 0:
            new_tokens.append(markdown.tokens[-1])

        return ParsedMarkdown(
            markdown.raw_content,
            new_tokens,
            markdown.metadata,
            markdown.env,
            markdown.resources,
        )

    def _remove_empty_code_blocks(self, markdown: ParsedMarkdown) -> ParsedMarkdown:
        """
        Remove empty code blocks in the markdown. They tend to be put at the very end of the notebook, and do not look
        good in the rendered markdown.
        :param markdown: The parsed markdown content.
        :return: The markdown content with the empty code blocks removed.
        """
        new_tokens = []
        for token in markdown.tokens:
            if token.type == "fence" and token.content.strip() == "":
                continue
            new_tokens.append(token)

        return ParsedMarkdown(
            markdown.raw_content,
            new_tokens,
            markdown.metadata,
            markdown.env,
            markdown.resources,
        )

    def _add_frontmatter(
        self, notebook_path: Path, markdown: ParsedMarkdown
    ) -> ParsedMarkdown:
        """
        Add document metadata to the Markdown content.
        :param notebook_path: The path to the notebook file.
        :param markdown: The parsed Markdown content.
        :return: The Markdown content with the frontmatter added.
        """
        # Add all the attributes to render in the metadata
        new_metadata = {**markdown.metadata}
        new_metadata["title"] = self._extract_title(markdown)
        new_metadata["notebook_path"] = str(notebook_path.relative_to(MAIN_DIR))
        new_metadata["reading_time_min"] = markdown.env["wordcount"]["minutes"]
        return ParsedMarkdown(
            markdown.raw_content,
            markdown.tokens,
            new_metadata,
            markdown.env,
            markdown.resources,
        )

    def _extract_title(self, markdown: ParsedMarkdown) -> str | None:
        """
        Extract the title from the markdown content. The fist level 1 heading is considered the title.
        :param markdown: The parsed markdown content.
        :return: The title of the document.
        """
        use_next = False
        for token in markdown.tokens:
            if use_next and token.type == "inline":
                return token.content
            # If the current token is a heading, the next inline token will be the title
            use_next = token.type == "heading_open" and token.markup == "#"
        return None

    def _process_assets(
        self,
        markdown: ParsedMarkdown,
        tokens: list[markdown_it.token.Token] | None,
        notebook_path: Path,
        assets_dir: Path,
    ) -> list[markdown_it.token.Token] | None:
        """
        Iterate over all the assets in the markdown content, download them and update the paths in Markdown, so they
        point to the downloaded files. As a side effect, the assets are downloaded to the local directory specified
        in the configuration.
        :param markdown: The parsed markdown content.
        :param tokens: List of tokens to parse.
        :param notebook_path: Path to the notebook.
        :param assets_dir: Path to store all the assets to.
        :return: The updated markdown content with the paths to the assets updated
        """
        if tokens is None:
            return None

        new_tokens = []
        for token_num, token in enumerate(tokens):
            # Recursively process the children of the token
            token.children = self._process_assets(
                markdown, token.children, notebook_path, assets_dir
            )

            # If the token is not an image or a link, just add it to the new tokens
            if token.type == "link_open":
                new_token = self._process_link_opening(
                    markdown, token, notebook_path, assets_dir
                )
            elif token.type == "image":
                new_token = self._process_image(
                    markdown, token, notebook_path, assets_dir
                )
            else:
                new_token = token

            new_tokens.append(new_token)

        return new_tokens

    def _process_link_opening(
        self,
        markdown: ParsedMarkdown,
        token: markdown_it.token.Token,
        notebook_path: Path,
        assets_dir: Path,
    ) -> markdown_it.token.Token:
        """
        Process the opening link token to ensure that they point to local files, whenever possible.
        :param markdown: The parsed markdown content.
        :param token: The link_open token to process.
        :param notebook_path: The path to the notebook.
        :param assets_dir: The directory where the assets are stored.
        :return: The updated link_open token.
        """
        # Only local links should be updated. If we have such a link, then it may be a local asset or another notebook.
        link_address = token.attrGet("href") or ""
        parsed_link = urlparse(link_address)
        if parsed_link.scheme:
            # Remote link, don't process it
            return token

        # Get full path the link points to
        target_path = notebook_path.parent / link_address
        if not target_path.exists():
            raise ParsingException(
                f"Path {target_path} not found in the notebook directory"
            )

        # TODO: Handle the case when the link points to another notebook or local file
        return token

    def _process_image(
        self,
        markdown: ParsedMarkdown,
        token: markdown_it.token.Token,
        notebook_path: Path,
        assets_dir: Path,
    ) -> markdown_it.token.Token:
        """
        Process the image token to download the image and update the path in the token.
        :param markdown: The parsed markdown content.
        :param token: The image token to process.
        :param notebook_path: The path to the notebook.
        :param assets_dir: The directory where the assets are stored.
        :return: The updated image token.
        """
        # Only process local assets, as remote ones may be hosted there on purpose (like big datasets)
        asset_link = token.attrGet("src") or ""
        parsed_link = urlparse(asset_link)

        # Perfectly all the assets should be local, but we can't guarantee that. Remote and base64 images are also
        # valid in Jupyter notebooks, so we need to handle them properly.
        resource_outputs = markdown.resources.get("outputs", {})
        if asset_link in resource_outputs:
            # Asset is resource generated by the markdown exporter, so we need to process it
            asset_content = resource_outputs.get(asset_link)
            asset_location = assets_dir / Path(asset_link).name
            with open(asset_location, "wb") as f:
                f.write(asset_content)
        elif asset_link.startswith("data:image"):
            # We have a base64 image that we're not going to store as a file and then process as a local asset
            head, tail = asset_link.split(";", 1)
            encoding, body = tail.split(",", 1)
            unique_slug = uuid.uuid4().hex
            asset_location = assets_dir / f"{unique_slug}"
            with open(asset_location, "wb") as f:
                f.write(base64.b64decode(body))
            asset_location = self._guess_file_extension(asset_location)
        elif parsed_link.scheme:
            # Download the remote asset and save it to the assets directory, then process as a local asset
            response = requests.get(asset_link, timeout=30)
            if not response.ok:
                raise ParsingException(f"Failed to download asset {asset_link}.")

            # Save file in a local directory and try to derive the mime type from the response
            asset_location = assets_dir / Path(parsed_link.path).name
            with open(asset_location, "wb") as f:
                f.write(response.content)

            # File has no extension, try to derive the mime type from the response
            if not asset_location.suffix:
                asset_location = self._guess_file_extension(asset_location)
        else:
            # Local asset, just get the path
            asset_location = notebook_path.parent / asset_link

        # Check if the asset exists in the notebook directory
        if not asset_location.exists():
            raise ParsingException(
                f"Asset {asset_location} not found in the notebook directory"
            )

        # Copy the asset and update the path in the token
        new_asset_location = assets_dir / asset_location.name
        if new_asset_location != asset_location:
            shutil.copyfile(asset_location, new_asset_location)

        # Create a new token with the updated path
        new_token = markdown_it.token.Token(
            type=token.type,
            tag=token.tag,
            nesting=token.nesting,
            attrs={**token.attrs},
            map=token.map,
            level=token.level,
            children=token.children,
            content=token.content,
        )

        relative_asset_location = new_asset_location.relative_to(MAIN_DIR)
        # Relative web url does not contain the .dist/qdrant-landing/static prefix
        relative_web_url = Path(*relative_asset_location.parts[3:])
        new_token.attrSet("src", f"/{relative_web_url}")
        return new_token

    def _guess_file_extension(self, path: Path) -> Path:
        """
        Guess the file extension based on the mime type of the file.
        :param path: The path to the file.
        :return: The path to the file with the correct extension.
        """
        mimetypes.init()
        file_mime = magic.from_file(path, mime=True)
        file_extension = mimetypes.guess_extension(file_mime)
        logger.debug(f"Guessed extension {file_extension} for {path}")
        path = path.rename(path.with_suffix(file_extension))
        return path

    def _write_index_file(self, output_dir: Path):
        """
        Write the _index.md file to the output directory.
        :param output_dir: The directory where the _index.md file should be saved.
        """
        index_file = output_dir / "_index.md"
        if index_file.exists():
            return

        # Create a new index file and write it to the output directory
        index_markdown = ParsedMarkdown(
            "",
            [],
            metadata={
                "title": output_dir.name,
                "is_empty": False,
                "partition": "qdrant",
            },
        )
        with open(index_file, "w") as f:
            rendered_md = self._md.renderer.render(
                list(index_markdown.iter_tokens()), self._md.options, index_markdown.env
            )
            f.write(rendered_md)
