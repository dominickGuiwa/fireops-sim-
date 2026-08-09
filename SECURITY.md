# Security policy

FIREOPS-SIM is a synthetic research simulator. It does not require credentials,
process personal data, or provide authentication in v0.1.

## Supported versions

Security fixes are made on the latest release and the default branch. Older
research snapshots may not receive patches.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| Earlier snapshots | No |

## Reporting a vulnerability

Please use GitHub's **Security** tab to submit a private vulnerability report.
Do not open a public issue for a suspected vulnerability, leaked secret, or a
report that includes exploit details. Include the affected revision, a minimal
reproduction, expected impact, and any suggested mitigation. Maintainers will
acknowledge the report, assess it, and coordinate disclosure through the
private advisory; response timing depends on maintainer availability.

If a credential is ever committed, revoke and rotate it immediately. Removing
it from the current tree is not sufficient because Git history and forks may
retain it.

## Scope and security assumptions

- The FastAPI service is a local research API, not an internet-hardened or
  multi-tenant service. Put it behind appropriate access controls before any
  network exposure.
- Request validation limits configuration values, but the simulator is not a
  sandbox for untrusted Python policies or model files.
- Treat third-party PPO model files as untrusted input. Load only artifacts from
  sources you control; Python and PyTorch serialization formats can execute
  code during deserialization.
- The browser client is a visualization and control surface, not an authority
  for life-safety decisions.
- The guided UI stores only synthetic trial summaries and a tutorial-seen flag
  in that browser's local storage. It does not ask for names, real locations,
  medical or financial details, or other sensitive data; it sends no analytics
  or cloud telemetry. **Clear practice data** removes the saved browser record.
- No API key or external service is needed. Store any future local settings in
  environment variables and never commit `.env` files or credentials.

## Responsible use

FIREOPS-SIM models abstract actions on fictional maps. It must not be used for
real emergency response, evacuation, firefighting, military planning, or other
life-safety decisions. Reports about unsafe defaults, misleading claims, or
ways the project could encourage operational misuse are welcome through the
same private reporting channel.

## Dependency hygiene

GitHub Actions tests the supported application paths, and Dependabot monitors
Python, npm, workflow, and container-base dependencies. Before publishing a
release, review dependency alerts, rebuild containers from current base images,
and verify the generated lockfile and Python dependency resolution.
