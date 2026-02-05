#!/usr/bin/env python3
"""
Email Dashboard Script

Runs the turbulence monitoring system and emails the dashboard PNG.
Designed to be run daily via cron or scheduler.

Setup:
1. Create a Gmail App Password:
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the 16-character password

2. Set environment variables:
   export GMAIL_ADDRESS="your.email@gmail.com"
   export GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
   export RECIPIENT_EMAIL="recipient@example.com"  # optional, defaults to GMAIL_ADDRESS

3. Run:
   python scripts/email_dashboard.py

4. For daily automation (cron example - runs at 6 PM ET on weekdays):
   0 18 * * 1-5 cd /path/to/repo && /path/to/venv/bin/python scripts/email_dashboard.py
"""

import os
import smtplib
import sys
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.main import run_pipeline


def send_email_with_dashboard(
    dashboard_path: str,
    sender_email: str,
    app_password: str,
    recipient_email: str,
    subject: str = None,
    body: str = None
) -> bool:
    """
    Send an email with the dashboard PNG attached.

    Args:
        dashboard_path: Path to the dashboard.png file
        sender_email: Gmail address
        app_password: Gmail app password (not regular password)
        recipient_email: Recipient email address
        subject: Email subject (default: auto-generated with date)
        body: Email body text (default: auto-generated)

    Returns:
        True if successful, False otherwise
    """
    today = datetime.now().strftime("%Y-%m-%d")

    if subject is None:
        subject = f"Market Turbulence Dashboard - {today}"

    if body is None:
        body = f"""Market Turbulence Monitoring Report

Date: {today}

The attached dashboard shows:
- Current turbulence level and regime status
- Divergence detection (turbulence high while markets rising)
- Historical turbulence with threshold levels

This is an automated daily report.
"""

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach body
    msg.attach(MIMEText(body, 'plain'))

    # Attach dashboard image
    dashboard_file = Path(dashboard_path)
    if not dashboard_file.exists():
        print(f"[ERROR] Dashboard not found: {dashboard_path}")
        return False

    with open(dashboard_file, 'rb') as f:
        img_data = f.read()

    image = MIMEImage(img_data, name=dashboard_file.name)
    image.add_header('Content-Disposition', 'attachment', filename=dashboard_file.name)
    msg.attach(image)

    # Send via Gmail SMTP
    try:
        print(f"[EMAIL] Connecting to Gmail SMTP...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            print(f"[EMAIL] Logging in as {sender_email}...")
            server.login(sender_email, app_password)
            print(f"[EMAIL] Sending to {recipient_email}...")
            server.send_message(msg)
            print(f"[EMAIL] Successfully sent dashboard to {recipient_email}")
            return True

    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Authentication failed. Check your app password.")
        print("        Make sure you're using an App Password, not your regular Gmail password.")
        print("        Create one at: https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False


def main():
    """Run pipeline and send email."""
    # Get credentials from environment (or .env file)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    sender_email = os.environ.get('EMAIL_ADDRESS')
    app_password = os.environ.get('EMAIL_PASSWORD')
    recipient_email = os.environ.get('EMAIL_RECIPIENT', sender_email)

    if not sender_email or not app_password:
        print("[ERROR] Missing credentials in .env file.")
        print("        Create .env file with:")
        print("")
        print("        EMAIL_ADDRESS=your.email@gmail.com")
        print("        EMAIL_PASSWORD=xxxx xxxx xxxx xxxx")
        print("        EMAIL_RECIPIENT=recipient@gmail.com")
        print("")
        sys.exit(1)

    print("=" * 60)
    print("MARKET TURBULENCE DASHBOARD - EMAIL REPORT")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sender: {sender_email}")
    print(f"Recipient: {recipient_email}")
    print("=" * 60)

    # Run the pipeline with fresh dates
    print("\n[1/2] Running turbulence pipeline...")
    try:
        config = get_config()  # This now uses current date dynamically
        run_pipeline(config)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    # Send email
    print("\n[2/2] Sending email...")
    dashboard_path = Path(config.output_dir) / config.dashboard_filename

    success = send_email_with_dashboard(
        dashboard_path=str(dashboard_path),
        sender_email=sender_email,
        app_password=app_password,
        recipient_email=recipient_email
    )

    if success:
        print("\n" + "=" * 60)
        print("COMPLETE - Dashboard emailed successfully!")
        print("=" * 60)
    else:
        print("\n[ERROR] Email sending failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
