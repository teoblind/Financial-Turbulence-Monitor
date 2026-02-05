#!/bin/bash
# Run turbulence dashboard and email
cd ~/Financial-turbulence-monitoring-system-with-Mahalanobis-distance
source venv/bin/activate
python scripts/email_dashboard.py
