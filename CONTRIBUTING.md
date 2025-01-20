# Contributing to the Qdrant examples

This repository is intended for keeping the sources of the tutorials presenting how Qdrant may be used in different
scenarios. These tutorials are also available on the [Qdrant website](https://qdrant.tech), so they should be rather
comprehensive and well-documented, not pure code snippets.

> As a rule of thumb, the notebooks should present an idea, not focus too much on handling lots of data or complex
> scenarios. The goal is to show how Qdrant may be used in a specific context, not to present a full-fledged solution.

## Repository structure

All the notebooks are divided based on their complexity. The current structure divides the notebooks into three groups,
but that may change in the future. The current structure is as follows:

- `101-foundations`
- `201-intermediate`
- `301-advanced`

There are no clear boundaries between the groups, however the foundation notebooks are intended to present just the
Qdrant basics, while the more advanced notebooks may present integrations with external tools or even end-to-end
solutions.

### Python environment

[Poetry](https://python-poetry.org/) is used to manage the Python environment. The `pyproject.toml` file contains the
common dependencies for all the notebooks. **If a specific notebook requires additional dependencies, they should be
installed in the notebook itself, usually using pip.** The main project is not supposed to accumulate all the possible
dependencies, as it may lead to conflicts and long installation times.

## Environmental variables

Whenever interacting with Qdrant, you should configure the connection using environmental variables. That helps to run
the notebooks automatically without the need to change the code. The following variables are used:

### Qdrant

- `QDRANT_LOCATION` - the location of the Qdrant instance, e.g. `http://localhost:6333`, but `:memory:` is also possible
- `QDRANT_API_KEY` - the API key used to authenticate the requests

### HuggingFace

- `HF_API_KEY` - the API key used to authenticate the requests to the HuggingFace API

## Good practices

1. Qdrant version should not be restricted in the individual notebooks. That applies to the `pyproject.toml` file as
   well. The notebooks should be able to run with the latest version of Qdrant, and if there are any breaking changes,
   the notebooks should be updated accordingly.
2. The notebooks should be self-contained. That means that all the necessary data should be downloaded or generated
   within the notebook. If the data is too large to be stored in the repository, it should be downloaded from an
   external source.

## Running the notebooks

Although the notebooks are intended to be run in an interactive Jupyter environment, they should not require user input
to run, unless it's necessary for the tutorial. The notebooks should be able to run from top to bottom without any
interruptions, and perfectly an execution of the whole notebook should not take more than 30 minutes. That simplifies
the CI/CD process and makes the notebooks more user-friendly.

You can test the automatic execution of the notebooks using the `jupyter execute` command. The following command will
execute the`test.ipynb` notebook and run all the cells, saving the changes in place:

```bash
jupyter execute test.ipynb --inplace
```

## Code style

`pre-commit` is used to enforce the code style. The configuration is stored in the `.pre-commit-config.yaml` file.
