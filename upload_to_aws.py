#!/usr/bin/env python3
"""
Upload ARC Challenge Project to AWS Instance

This script uploads your project files to the AWS instance for training.
"""

import os
import subprocess
import sys
from pathlib import Path

def upload_files_to_aws(instance_ip, key_path, local_dir=".", remote_dir="/home/ubuntu/arc-africa"):
    """Upload project files to AWS instance using scp."""
    
    print(f"🚀 Uploading project files to AWS instance {instance_ip}")
    print(f"📁 Local directory: {local_dir}")
    print(f"📁 Remote directory: {remote_dir}")
    
    # Files to upload (exclude large files and unnecessary directories)
    exclude_patterns = [
        '*.pth',  # Model files (too large)
        '__pycache__',
        '.git',
        'node_modules',
        '*.log',
        '*.tmp',
        'results/',
        'logs/',
        'notebooks/',
        '.venv',
        'venv'
    ]
    
    # Build rsync command with exclusions
    exclude_args = []
    for pattern in exclude_patterns:
        exclude_args.extend(['--exclude', pattern])
    
    # Create rsync command
    rsync_cmd = [
        'rsync', '-avz', '--progress',
        *exclude_args,
        '-e', f'ssh -i {key_path}',
        f'{local_dir}/',
        f'ubuntu@{instance_ip}:{remote_dir}/'
    ]
    
    print(f"📤 Running: {' '.join(rsync_cmd)}")
    
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
    """Main function to handle file upload."""
    
    # Get instance details from user
    print("🔧 AWS Instance File Upload")
    print("=" * 40)
    
    instance_ip = input("Enter your AWS instance IP: ").strip()
    key_path = input("Enter path to your SSH key (.pem file): ").strip()
    
    if not instance_ip or not key_path:
        print("❌ Instance IP and key path are required!")
        return
    
    if not os.path.exists(key_path):
        print(f"❌ SSH key file not found: {key_path}")
        return
    
    # Set correct permissions for SSH key
    try:
        os.chmod(key_path, 0o400)
        print(f"✅ Set SSH key permissions: {key_path}")
    except Exception as e:
        print(f"⚠️ Could not set SSH key permissions: {e}")
    
    # Upload files
    success = upload_files_to_aws(instance_ip, key_path)
    
    if success:
        print("\n🎉 Upload completed!")
        print("\n📋 Next steps on AWS instance:")
        print("1. SSH to your instance:")
        print(f"   ssh -i {key_path} ubuntu@{instance_ip}")
        print("2. Run the setup script:")
        print("   chmod +x aws_quick_setup.sh && ./aws_quick_setup.sh")
        print("3. Activate environment and start training:")
        print("   source arc-env/bin/activate")
        print("   python aws_train_enhanced.py")
    else:
        print("\n❌ Upload failed. Please check your connection and try again.")

if __name__ == "__main__":
    main() 