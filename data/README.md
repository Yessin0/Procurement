# Data Requirements & Verification

## Dataset Details
- **Source:** Mendeley Data (Version 2)
- **DOI:** 10.17632/f3y4nrn3s6.2
- **Download Link:** https://data.mendeley.com/datasets/f3y4nrn3s6
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Internal Shared Drive Link:** <INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE>

## Verification & Checksum
To verify your local copy matches the team version, run in PowerShell:
`Get-FileHash -Algorithm SHA256 data\raw\GTI_labelled_cartel_data_NOV2023.csv`

Expected File Name: `GTI_labelled_cartel_data_NOV2023.csv`
Expected SHA-256 Hash: `13AB9E8EF511AC5F9B8258875D1596C4AFBE60D5F4EAE7CCCAFD6B1E0559A1D3` 

## Database Setup Note for Teammates (PostgreSQL 15+)
If you get `permission denied for schema public` during `python manage.py migrate`, connect to PostgreSQL as superuser `postgres` and run:

```sql
GRANT ALL PRIVILEGES ON DATABASE fraud_detection TO app;
GRANT ALL ON SCHEMA public TO app;
ALTER SCHEMA public OWNER TO app;