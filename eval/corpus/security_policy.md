# Information Security Policy

## Password Requirements

All passwords must be at least 14 characters long and unique per system.
Passwords must be stored in the company-approved password manager. Password
rotation is required only after a suspected compromise, not on a fixed
schedule.

## Multi-Factor Authentication

MFA is mandatory for email, VPN, the code repository, and all production
systems. Hardware security keys are the preferred second factor; TOTP
authenticator apps are an acceptable alternative. SMS-based MFA is prohibited.

## Incident Reporting

Suspected security incidents must be reported to the security team within
1 hour of discovery via the #security-incidents channel or the 24/7 hotline.
Do not attempt to investigate or remediate on your own before reporting.

## Data Classification

Data is classified into three tiers: Public, Internal, and Restricted.
Restricted data (customer PII, credentials, financial records) must never be
stored on personal devices and may only be shared through approved encrypted
channels.

## Laptop Security

Company laptops must have full-disk encryption enabled, automatic screen lock
at 5 minutes, and the endpoint protection agent installed. Lost or stolen
devices must be reported within 1 hour so they can be remotely wiped.
