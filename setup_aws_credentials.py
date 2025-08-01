#!/usr/bin/env python3
"""
AWS Credentials Setup for ARC Challenge

This script helps you configure AWS credentials for uploading files
and managing your ARC Challenge AWS instance.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_aws_cli():
    """Check if AWS CLI is installed."""
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ AWS CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ AWS CLI not found")
            return False
    except FileNotFoundError:
        print("❌ AWS CLI not installed")
        return False

def install_aws_cli():
    """Install AWS CLI if not present."""
    print("📦 Installing AWS CLI...")
    
    if sys.platform.startswith('win'):
        # Windows installation
        print("🪟 Installing AWS CLI for Windows...")
        print("Please download and install from: https://awscli.amazonaws.com/AWSCLIV2.msi")
        print("Or run: msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi")
    else:
        # Linux/Mac installation
        try:
            subprocess.run(['curl', 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip', '-o', 'awscliv2.zip'], check=True)
            subprocess.run(['unzip', 'awscliv2.zip'], check=True)
            subprocess.run(['sudo', './aws/install'], check=True)
            print("✅ AWS CLI installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install AWS CLI: {e}")
            return False
    
    return True

def configure_aws_credentials():
    """Configure AWS credentials interactively."""
    print("\n🔑 AWS Credentials Configuration")
    print("=" * 40)
    
    print("Please enter your AWS credentials:")
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()
    region = input("Default region (e.g., us-east-1): ").strip() or "us-east-1"
    output_format = input("Default output format (json): ").strip() or "json"
    
    if not access_key or not secret_key:
        print("❌ Access Key ID and Secret Access Key are required!")
        return False
    
    # Configure AWS CLI
    try:
        subprocess.run([
            'aws', 'configure', 'set', 'aws_access_key_id', access_key
        ], check=True)
        
        subprocess.run([
            'aws', 'configure', 'set', 'aws_secret_access_key', secret_key
        ], check=True)
        
        subprocess.run([
            'aws', 'configure', 'set', 'default.region', region
        ], check=True)
        
        subprocess.run([
            'aws', 'configure', 'set', 'default.output', output_format
        ], check=True)
        
        print("✅ AWS credentials configured successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to configure AWS credentials: {e}")
        return False

def test_aws_connection():
    """Test AWS connection by listing EC2 instances."""
    print("\n🧪 Testing AWS Connection...")
    
    try:
        result = subprocess.run([
            'aws', 'ec2', 'describe-instances', 
            '--query', 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]',
            '--output', 'table'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ AWS connection successful!")
            print("📋 Your EC2 instances:")
            print(result.stdout)
            return True
        else:
            print("❌ AWS connection failed")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to test AWS connection: {e}")
        return False

def create_ssh_key_pair():
    """Create an SSH key pair for EC2 access."""
    print("\n🔐 Creating SSH Key Pair...")
    
    key_name = "arc-challenge-key"
    
    try:
        # Check if key pair already exists
        result = subprocess.run([
            'aws', 'ec2', 'describe-key-pairs', '--key-names', key_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Key pair '{key_name}' already exists")
            return key_name
        
        # Create new key pair
        result = subprocess.run([
            'aws', 'ec2', 'create-key-pair',
            '--key-name', key_name,
            '--query', 'KeyMaterial',
            '--output', 'text'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Save private key to file
            key_file = f"{key_name}.pem"
            with open(key_file, 'w') as f:
                f.write(result.stdout)
            
            # Set correct permissions (Unix-like systems)
            if not sys.platform.startswith('win'):
                os.chmod(key_file, 0o400)
            
            print(f"✅ SSH key pair created: {key_file}")
            print(f"🔑 Private key saved to: {os.path.abspath(key_file)}")
            return key_name
        else:
            print(f"❌ Failed to create key pair: {result.stderr}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create SSH key pair: {e}")
        return None

def main():
    """Main function to set up AWS credentials."""
    print("🚀 AWS Setup for ARC Challenge")
    print("=" * 40)
    
    # Check if AWS CLI is installed
    if not check_aws_cli():
        print("\n📦 AWS CLI not found. Installing...")
        if not install_aws_cli():
            print("❌ Failed to install AWS CLI. Please install manually.")
            return
    
    # Configure credentials
    if not configure_aws_credentials():
        print("❌ Failed to configure AWS credentials")
        return
    
    # Test connection
    if not test_aws_connection():
        print("❌ AWS connection test failed. Please check your credentials.")
        return
    
    # Create SSH key pair
    key_name = create_ssh_key_pair()
    if not key_name:
        print("❌ Failed to create SSH key pair")
        return
    
    print("\n🎉 AWS setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Your AWS credentials are configured")
    print("2. SSH key pair created for EC2 access")
    print("3. You can now run: python aws_setup_arc.py")
    print("4. Or upload files with: python upload_to_aws.py")

if __name__ == "__main__":
    main() 