#!/usr/bin/env python3
"""
Simple Upload Script for ARC Challenge to AWS Instance

This script uploads your project files to the AWS instance you're already connected to.
"""

import os
import subprocess
import sys
from pathlib import Path

def upload_files_to_aws():
    """Upload project files to AWS instance."""
    
    # Get instance IP from user
    print("🔧 AWS File Upload for ARC Challenge")
    print("=" * 40)
    
    instance_ip = input("Enter your AWS instance IP (e.g., 3.250.123.45): ").strip()
    
    if not instance_ip:
        print("❌ Instance IP is required!")
        return False
    
    # Check if SSH key exists
    key_file = "arc-challenge-key.pem"
    if not os.path.exists(key_file):
        print(f"❌ SSH key file not found: {key_file}")
        print("Please create it first with: aws ec2 create-key-pair --key-name arc-challenge-key --query 'KeyMaterial' --output text > arc-challenge-key.pem")
        return False
    
    # Set correct permissions for SSH key
    try:
        os.chmod(key_file, 0o400)
        print(f"✅ Set SSH key permissions: {key_file}")
    except Exception as e:
        print(f"⚠️ Could not set SSH key permissions: {e}")
    
    # Create rsync command to upload files
    rsync_cmd = [
        'rsync', '-avz', '--progress',
        '--exclude', '*.pth',  # Exclude large model files
        '--exclude', '__pycache__',
        '--exclude', '.git',
        '--exclude', '*.log',
        '--exclude', 'results/',
        '--exclude', 'logs/',
        '--exclude', 'notebooks/',
        '--exclude', '.venv',
        '--exclude', 'venv',
        '-e', f'ssh -i {key_file}',
        './',
        f'ubuntu@{instance_ip}:/home/ubuntu/arc-africa/'
    ]
    
    print(f"📤 Uploading files to {instance_ip}...")
    print(f"Command: {' '.join(rsync_cmd)}")
    
    try:
        result = subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)
        print("✅ Upload completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function."""
    success = upload_files_to_aws()
    
    if success:
        print("\n🎉 Upload completed!")
        print("\n📋 Next steps on AWS instance:")
        print("1. SSH to your instance:")
        print("   ssh -i arc-challenge-key.pem ubuntu@<your-instance-ip>")
        print("2. Navigate to project:")
        print("   cd /home/ubuntu/arc-africa")
        print("3. Run setup:")
        print("   chmod +x aws_quick_setup.sh && ./aws_quick_setup.sh")
        print("4. Start training:")
        print("   source arc-env/bin/activate")
        print("   python aws_train_enhanced.py")
    else:
        print("\n❌ Upload failed. Please check your connection and try again.")

if __name__ == "__main__":
    main() 