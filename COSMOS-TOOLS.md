# Updating the COSMOS setup tools

This file and the bootstrap, cosmos, update, and activate scripts are managed by
cosmos-python. `.cosmos-python.json` records their installed template version and
checksums. Commit that file with the scripts. Keep project notes in
PYTHON-ENVIRONMENT.md, which is not overwritten by template updates.

After obtaining a newer trusted cosmos-python checkout, run from your terminal:

```sh
bash /path/to/cosmos-python/update-repos.sh /path/to/this-repo --dry-run
bash /path/to/cosmos-python/update-repos.sh /path/to/this-repo
```

Windows PowerShell:

```powershell
& C:\path\cosmos-python\update-repos.ps1 -Repositories C:\path\this-repo -DryRun
& C:\path\cosmos-python\update-repos.ps1 -Repositories C:\path\this-repo
```

The updater stops on locally changed helper files. Reconcile them with the trusted
new template before retrying; it does not silently overwrite them. Dependencies,
Python version, lockfile, tasks, and project documentation remain project-owned.
Only the Pixi tool-version constraint in pixi.toml may be migrated to match new
bootstrap scripts, provided it still matches the recorded previous constraint.

Add `--sync` (Windows: `-Sync`) to the updater to also bootstrap and install
the saved environment in each target. Otherwise, after updating, run `bash bootstrap.sh` and `bash cosmos.sh setup` (Windows:
bootstrap.ps1 and cosmos.ps1 setup) to install the required tool and saved
environment on this machine. Run the project's tests, review the diff, and commit
the changed scripts and state. Collaborators pull and run bootstrap/setup too.
This never forces newer Python or library versions: use the project's update
workflow separately when requirements change.

Backups of replaced files and the old state live under
`.cache/cosmos-python-updates/<id>/`. To undo a helper update, copy those files back
to the repository root and remove files listed as newly created in RESTORE.json.
Do not copy RESTORE.json itself to the root. Reapply the old bootstrap/setup if the
Pixi tool version changed. Backups do not include installed environments.
Updates are explicit, local operations; there is no background watcher, automatic
Git pull, commit, or push. Only paths you supply are accessed.
