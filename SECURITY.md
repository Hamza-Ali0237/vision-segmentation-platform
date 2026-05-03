# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, open a [GitHub Security Advisory](https://github.com/Hamza-Ali0237/vision-segmentation-platform/security/advisories/new) so it can be addressed privately before public disclosure.

Include as much detail as possible:
- A description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Any suggested mitigations

You can expect an acknowledgement within 48 hours and a resolution plan within 7 days.

## Credential Safety

This project interacts with AWS services. Keep the following in mind:

- **Never commit real credentials.** `training/configs/base.yaml` is git-ignored for this reason.
- Use IAM roles with least-privilege permissions — the README documents the minimum required policies.
- Rotate credentials immediately if you suspect they have been exposed.
- Always delete SageMaker endpoints when not in use to limit exposure and cost.
