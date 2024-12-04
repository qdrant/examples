import configparser
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

MAIN_DIR = Path(__file__).parent.parent.parent


class GitHubRepository:
    """
    A class for interacting with a Git repository.
    """

    def __init__(self, main_dir: Path | str = MAIN_DIR):
        if isinstance(main_dir, str):
            main_dir = Path(main_dir)
        self._main_dir = main_dir

    def repository_name(self) -> str:
        """
        Get the name of the repository.
        :return: The name of the repository.
        """
        config_file = self._main_dir / ".git" / "config"
        config = configparser.ConfigParser()
        config.read(config_file)

        repo_url = None
        for section_name, section in config.items():
            if not section_name.startswith("remote"):
                continue
            logger.debug("Reading remote section: {}", section_name)
            repo_url = section.get("url")
            break

        if repo_url is None:
            raise RuntimeError("Could not find the repository URL.")

        # Log the repository URL for debugging purposes
        logger.info("Repository URL: {}", repo_url)

        # Parse the repo url to extract the repository name (without the .git suffix)
        if repo_url.startswith("git@"):
            repo_name = repo_url.partition(":")[2]
        else:
            parsed_url = urlparse(repo_url)
            repo_name = parsed_url.path[1]

        # Remove the .git suffix
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        return repo_name

    def current_branch_name(self) -> str:
        """
        Get the name of the current branch.
        :return: The name of the current branch.
        """
        head_file = self._main_dir / ".git" / "HEAD"
        with head_file.open("r") as f:
            content = f.read().splitlines()

        for line in content:
            if line[0:4] == "ref:":
                return line.partition("refs/heads/")[2]

        raise RuntimeError("Could not determine the current branch name.")

    def relative_path(self, path: Path) -> Path:
        """
        Get the relative path from the main directory.
        :param path: The path to get the relative path for.
        :return: The relative path.
        """
        return path.relative_to(self._main_dir)
