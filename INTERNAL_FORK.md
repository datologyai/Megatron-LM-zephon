# DatologyAI integration fork

This repository is a private copy of NVIDIA Megatron-LM with an opt-in Zephon
training dataloader. It is intentionally not part of GitHub's public fork
network. The intended end state is a public reference integration.

## Continuous integration

The upstream NVIDIA GitHub and GitLab pipelines are not used here. `Zephon CI`
runs only when the integration adapter, examples, fixtures, tests, or workflow
change. It provides:

- formatting and lint checks for the Zephon integration files;
- public adapter contract tests that do not require Zephon.

While Zephon is private, run `scripts/validate_zephon_install.sh` locally from
the Megatron development container with normal GitHub credentials. It installs
the pinned private dependency into a clean temporary environment and runs the
real batch and elastic-resume tests. Hosted CI needs no private credential;
enable the same integration validation there once Zephon is public. There are
no scheduled, release, Slack, on-call, DCO, container-publishing, or
NVIDIA-internal CI jobs.

Cryptographic commit signing and DCO sign-off are optional in this internal
repository.

## Updating from NVIDIA

The local checkout uses `origin` for this repository and `upstream` for
`NVIDIA/Megatron-LM`.

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

After an upstream sync, rebase the Zephon integration branch and resolve any
changes around the external-dataloader seam. Do not restore upstream automation
unless its behavior is deliberately adopted by this project.
