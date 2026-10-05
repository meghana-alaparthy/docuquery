# Engineering Handbook

## Code review

Every pull request needs one approval before merging. Keep PRs under 400 lines when possible — smaller PRs get reviewed in under 4 hours on average, large ones sit for days. Reviewers: comment on the code, not the person.

## Testing

New features ship with unit tests. Aim for 80% line coverage on business logic; 100% is not required and chasing it wastes time. Run the full suite with `make test` before pushing.

## Deployments

We deploy continuously. Merges to `main` go out automatically after CI passes (about 12 minutes). If you need to hold a release, add the `hold-release` label. Roll back with one click in the deploy dashboard — no approval needed, then tell the team in #eng.

## On-call

On-call rotates weekly, Monday 9 AM to Monday 9 AM Central. Acknowledge pages within 15 minutes. If an alert fires more than twice a week and nobody acts on it, tune the alert instead of muting it.

## Incident reviews

Blameless by default. The write-up is due within 48 hours of resolution and must include: timeline, root cause, and at least one action item with an owner.
