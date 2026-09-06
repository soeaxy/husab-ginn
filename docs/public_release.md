# Code-only publication

Repository: <https://github.com/soeaxy/husab-ginn>.

The author authorized publication of the research code in a new repository.
The data have not been authorized for public release. This repository contains
algorithms, configuration examples, dependency metadata and synthetic tests.
It excludes research datasets, real coordinates and bounds, field records,
point predictions, learned weights, rendered maps and private audit manifests.
Aggregate settings and published manuscript summary statistics document the
protocol; they do not provide the underlying spatial observations.

No open-source license is selected or implied. The code's availability is not
a grant of unrestricted reuse rights. Research inputs remain controlled by
their owners.

## Checks before publishing changes

Stage only reviewed source files, then run:

```sh
python scripts/check_public_release.py
python scripts/check_public_release.py --all-history
```

The default command scans bytes in the Git index, including staged changes.
The history command scans every tree reachable from local refs. It does not
scan an unrelated old repository or remote caches. Both reject data/output
paths, archives, non-text/binary files, symlinks, model files, private keys and
recognizable credential formats. They print file paths and finding categories,
never matching credential values. This is a conservative automated guard,
not proof that every possible identifying string has been detected; review
text changes for real coordinates and restricted observations before staging.

Before Git initialization, use `python scripts/check_public_release.py
--worktree`. That mode skips build/dependency/cache directories and does not
follow symlinks. Normal releases and CI use the index/history modes instead.

The CI workflow checks source publication boundaries with the standard library.
The scientific regression suite additionally requires the dependencies declared
in `pyproject.toml`; run `python -m unittest discover -s tests -v` after installing
them. No private dataset is needed. The real-data artifact audit is disabled by
default and can only be run by an authorized holder of the private archives.
