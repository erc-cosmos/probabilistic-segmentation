# Python setup and cosmos-python

## What cosmos-python provides

[cosmos-python](https://github.com/erc-cosmos/cosmos-python) supplies the setup,
activation, and update scripts used in this repository. Pixi installs Python and
packages into a project-local environment so collaborators can use the same saved
dependency versions. You do not need Python or Conda installed beforehand.

The helper scripts are committed here. Ordinary users only need this repository;
they do not need to clone cosmos-python separately. Maintainers use a cosmos-python
checkout when updating the helper scripts.

`pixi.toml` lists this project's requirements, and `pixi.lock` records exact
platform-specific packages. They are authoritative for this workflow; the older
requirements files remain historical reference. The current environment uses
Python 3.11 and includes NumPy, SciPy, pandas, Matplotlib, IPython, ipykernel,
pytest, pytest-xdist, Hypothesis, pylint, autopep8, and mantel.

## Install and activate

Clone the project if you have not already done so:

```sh
git clone https://github.com/erc-cosmos/probabilistic-segmentation.git
cd probabilistic-segmentation
```

Run these commands from the repository directory.

### Mac/Linux

```sh
bash bootstrap.sh
bash cosmos.sh setup
source ./activate.sh
pixi run --locked python --version
pixi run --locked test
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\cosmos.ps1 setup
. .\activate.ps1
pixi run --locked python --version
pixi run --locked test
```

If PowerShell blocks activation, launch an allowed session with
`powershell -ExecutionPolicy Bypass`, return to the repository directory, and run
`. .\activate.ps1`. This setting applies to that process only and cannot override
organisation policy.

Bootstrap installs the required Pixi version in `.tools/`. Setup installs the
committed environment in `.pixi/`. Downloaded packages and plotting caches live in
`.cache/`. These local directories are ignored by Git. Downloads require internet
access; installation does not change system Python or your shell profile.

Activation makes Pixi available in the current terminal. Repeat activation in each
new terminal, and use `pixi run --locked python your_script.py` to run code. Pixi
requires `run`; `pixi python` is not a native command. `--locked` prevents a run
from silently changing saved dependency versions.

## Apply updates published by maintainers

Pull the repository's changes, then repeat bootstrap and setup using the commands
above. They install the required Pixi and synchronize your local environment with
the committed lockfile. Activate again and run the project's tests.

You do not need to run `update.sh` or regenerate the lockfile just to use a project
update. Those operations are for maintainers changing dependencies.

## Tests and notebooks

`pixi run --locked test` runs the 58 self-contained tests. The equivalent command is:

```sh
pixi run --locked python -m pytest tests --ignore=tests/test_readers.py
```

`pixi run --locked test-all` also includes seven reader tests requiring the original
Mazurka datasets under `data/`. Bare pytest may collect research notebook scripts
that need external files, so use the defined tasks for this project.

For VS Code, select `.pixi/envs/default/bin/python` on Mac/Linux or
`.pixi\envs\default\python.exe` on Windows as the Python/Jupyter interpreter.
Pixi commands already set the repository import path; no `.env` edits are needed.
Original notebooks can still contain old imports or machine-specific data paths.
Installing the environment does not fix those scripts or supply their datasets.

## Maintainers: change dependencies

For guided package addition or pinning, run `bash cosmos.sh add` on Mac/Linux, or:

```powershell
powershell -ExecutionPolicy Bypass -File .\cosmos.ps1 add
```

Enter the name, exact version, and source; leave the source blank for conda-forge.
Alternatively, edit `pixi.toml`. Put Python and conda-forge packages under
`[dependencies]`, and packages unavailable there under `[pypi-dependencies]`.
Do not list a package in both sections or edit the lockfile manually.

Apply the edited requirements:

```sh
bash update.sh
source ./activate.sh
pixi run --locked test
```

Windows: run `powershell -ExecutionPolicy Bypass -File .\update.ps1`, activate,
and run the same test task. To deliberately refresh all versions permitted by
the requirements, use `cosmos.sh upgrade` / `cosmos.ps1 upgrade`. Exact pins stay
fixed until changed. Clear any shell-level `PIXI_LOCKED` or `PIXI_FROZEN` settings
for the intentional update step.

Review the results and commit **both `pixi.toml` and `pixi.lock`**. Collaborators
then pull and run setup. To roll back, restore both files from a known working
revision and run setup again.

## Maintainers: update the cosmos-python helpers

Obtain the latest trusted cosmos-python checkout, then preview and apply:

```sh
bash /path/to/cosmos-python/update-repos.sh /path/to/probabilistic-segmentation --dry-run
bash /path/to/cosmos-python/update-repos.sh /path/to/probabilistic-segmentation --sync
```

Windows, in a PowerShell session that permits the scripts:

```powershell
& C:\path\cosmos-python\update-repos.ps1 -Repositories C:\path\probabilistic-segmentation -DryRun
& C:\path\cosmos-python\update-repos.ps1 -Repositories C:\path\probabilistic-segmentation -Sync
```

This installation is already registered in `.cosmos-python.json`; no adoption is
needed. The updater preserves project dependencies and detects edited helper files.
`--sync` also runs bootstrap and locked setup. Run the tests, review changes, and
commit the updated helpers and metadata. Collaborators pull and repeat bootstrap
and setup on their machines.

See [COSMOS-TOOLS.md](COSMOS-TOOLS.md) for managed files, conflicts, and backups.
This project guide is preserved during template updates.

## Sharing and validation

Commit the helper scripts, manifest, lockfile, `.cosmos-python.json`, ignore rules,
and documentation. Do not commit `.tools/`, `.pixi/`, `.cache/`, or credentials.
Setup runs locally and downloads tools/packages; it does not upload code to a chat
or create commits or pushes.

The lockfile declares Apple Silicon and Intel Mac, Windows x64, and Linux x64.
Only this Mac's runtime has been validated; Windows, Intel Mac, and Linux execution
remain untested. On 3 October 2026, all 58 self-contained tests passed. The suite
emitted 11 existing divide-by-zero-in-log warnings. Pixi reported a host
architecture-detection warning, but Python and tests executed successfully.
The seven external-dataset tests were not run.
