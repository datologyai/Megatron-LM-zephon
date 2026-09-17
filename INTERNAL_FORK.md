# DatologyAI integration fork

This repository is a private copy of NVIDIA Megatron-LM with an opt-in Zephon
training dataloader. It is intentionally outside GitHub's public fork network.
The intended end state is a public reference integration that demonstrates how
to replace an existing framework dataloader without changing its model or
training stack.

User documentation belongs in the [Zephon example guide](examples/zephon/README.md)
and [integration reference](docs/zephon.md). This file is for maintainers of the
fork.

## Repository policy

- `origin` is `datologyai/Megatron-LM-zephon`; `upstream` is
  `NVIDIA/Megatron-LM`.
- Zephon changes are reviewed through a pull request rather than committed
  directly to `main`.
- Preserve upstream copyright and licensing notices.
- Follow Megatron's contribution policy. Commits must include a DCO sign-off
  and should be cryptographically signed as required by the repository
  workflow.
- Keep Zephon-specific behavior confined to the adapter, entry point, examples,
  and focused tests so upstream synchronization remains tractable.

## Continuous integration

The upstream NVIDIA GitHub and GitLab pipelines are not used here. `Zephon CI`
runs only when the integration adapter, examples, fixtures, tests, dependency
pin, validation script, or workflow changes. It provides formatting and lint
checks plus adapter tests that do not require the private Zephon package.

While Zephon is private, run the release-facing validation locally from the
Megatron development environment with GitHub access to `datologyai/zephon`:

```bash
scripts/validate_zephon_install.sh
```

The script builds a clean temporary environment, installs the pinned Zephon
release, runs the focused adapter test, and executes the CPU elastic demo. Set
`ZEPHON_WHEEL=/path/to/zephon.whl` to validate a release candidate wheel.
Hosted CI deliberately has no private package credential. Once Zephon is
public, enable the same release-facing validation there. This fork has no
scheduled, release, Slack, on-call, container-publishing, or NVIDIA-internal CI
jobs.

## Updating from NVIDIA

```bash
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

After an upstream sync, rebase the Zephon integration branch and resolve any
changes around the external-dataloader seam. Do not restore upstream automation
unless its behavior is deliberately adopted by this project.

## Public-release checklist

Complete every item before changing the repository visibility to public.

### Dependency and access

- [ ] Publish an approved public Zephon release and replace the private Git pin
      in `requirements-zephon.txt` with its public installation source.
- [ ] Confirm a clean machine without DatologyAI GitHub or package credentials
      can install every dependency used by the examples and tests.
- [ ] Remove private package indexes, repository URLs, credentials, internal
      hostnames, account identifiers, and employee-specific paths from tracked
      files and GitHub configuration.
- [ ] Search the complete Git history, not only the current tree, for secrets or
      private artifacts; rewrite history if anything sensitive was committed.

### Licensing and provenance

- [ ] Obtain approval to publish the integration, fixtures, scripts, and Zephon
      API examples.
- [ ] Verify that all added files have appropriate copyright and license
      treatment and that third-party fixtures are redistributable.
- [ ] Preserve attribution to NVIDIA Megatron-LM and document the upstream
      repository and synchronization model.
- [ ] Confirm that the repository name, description, topics, and license do not
      imply endorsement by NVIDIA or upstream ownership of Zephon.

### Documentation and usability

- [ ] Replace every statement that Zephon is private with public installation
      and support information.
- [ ] Run the CPU elastic demo from a fresh clone using only documented commands.
- [ ] Run the full GPU checkpoint/resume demo from a fresh clone, including a
      two-GPU to one-GPU topology change.
- [ ] Verify all Markdown links, commands, paths, expected output, and current
      limitations in `README.md`, `examples/zephon/README.md`, and
      `docs/zephon.md`.
- [ ] Ensure the Megatron and TorchTitan Zephon guides use the same terminology
      for recipes, token proportions, token estimation, packing, canonical
      lanes, and elastic-resume guarantees.

### Tests and GitHub configuration

- [ ] Enable release-facing Zephon tests in hosted CI without repository
      secrets and confirm they pass on pull requests.
- [ ] Review workflow permissions, third-party actions, branch protection,
      required checks, CODEOWNERS, issue settings, and pull-request settings for
      a public repository.
- [ ] Run secret scanning, dependency review, and an appropriate source/security
      scan; resolve or explicitly accept every finding.
- [ ] Verify that no upstream workflow capable of publishing packages,
      containers, releases, or notifications was accidentally re-enabled.
- [ ] Confirm the public default branch contains the intended CI-surgery changes
      and that the Zephon integration PR has a reviewable, signed history.

### Publication

- [ ] Decide whether to retain this standalone repository or recreate it as a
      GitHub fork, and document the consequence for upstream synchronization.
- [ ] Add public issue/support guidance and identify maintainers responsible for
      the demo.
- [ ] Record the final private commit SHA, create a release tag for the reviewed
      public baseline, and save the successful clean-clone validation results.
- [ ] Have a second maintainer review this checklist and the visibility change.
- [ ] Change visibility only after every preceding item is complete.
