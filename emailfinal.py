import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import creds

def send_email_all_files(receiver_email, folder_path):
    """
    Sends an email with all files in the specified folder as attachments.
    
    Args:
        receiver_email (str): The recipient's email address.
        folder_path (str): The path to the folder containing the files to attach.
    """
    sender_email = creds.sender_email  # Replace with your email
    sender_password = creds.sender_password 

    # Email content
    subject = "All Files in Folder"
    body = f"The folder '{folder_path}' contains the following files, which are attached to this email."

    # Create the email
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            with open(file_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={filename}",
            )
            msg.attach(part)

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"Email sent successfully to {receiver_email} with all files in '{folder_path}'!")
    except Exception as e:
        print(f"Failed to send email: {e}")