#!/usr/bin/env python3
"""
AWS Setup for ARC Challenge Africa

This script sets up an AWS instance optimized for:
1. Generating competition submissions
2. Training improved models
3. Running experiments with better hardware

Author: ARC Challenge Africa Team
Date: January 2025
"""

import boto3
import json
import time
import os
from pathlib import Path

def create_ec2_instance():
    """Create an EC2 instance optimized for ARC Challenge."""
    
    # AWS Configuration
    ec2 = boto3.client('ec2')
    
    # Instance configuration for ARC Challenge
    instance_config = {
        'ImageId': 'ami-0c7217cdde317cfec',  # Deep Learning AMI with PyTorch
        'InstanceType': 'g4dn.xlarge',        # GPU instance with 4 vCPUs, 16GB RAM, 1 GPU
        'KeyName': 'arc-challenge-key',       # Your SSH key
        'SecurityGroups': ['arc-challenge-sg'],
        'UserData': '''#!/bin/bash
# Update system
sudo yum update -y

# Install additional packages
sudo yum install -y git htop tmux

# Set up Python environment
cd /home/ec2-user
python3 -m venv arc-env
source arc-env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install numpy pandas scikit-learn matplotlib seaborn tqdm

# Clone your repository (replace with your actual repo)
git clone https://github.com/your-username/arc-africa.git
cd arc-africa

# Install project requirements
pip install -r requirements.txt

# Set up GPU monitoring
pip install gpustat

echo "ARC Challenge environment setup complete!"
''',
        'TagSpecifications': [
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': 'ARC-Challenge-Solver'},
                    {'Key': 'Project', 'Value': 'ARC-Challenge-Africa'},
                    {'Key': 'Purpose', 'Value': 'Competition-Submission'}
                ]
            }
        ]
    }
    
    try:
        response = ec2.run_instances(**instance_config)
        instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ Created EC2 instance: {instance_id}")
        return instance_id
    except Exception as e:
        print(f"❌ Failed to create instance: {e}")
        return None

def create_security_group():
    """Create security group for ARC Challenge."""
    
    ec2 = boto3.client('ec2')
    
    try:
        # Create security group
        response = ec2.create_security_group(
            GroupName='arc-challenge-sg',
            Description='Security group for ARC Challenge'
        )
        group_id = response['GroupId']
        
        # Add SSH access
        ec2.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        print(f"✅ Created security group: {group_id}")
        return group_id
    except Exception as e:
        print(f"⚠️ Security group creation failed (may already exist): {e}")
        return None

def upload_project_files(instance_id):
    """Upload project files to AWS instance."""
    
    # This would use S3 or direct file transfer
    print("📁 Project files will be uploaded via S3 or direct transfer")
    
    # Create S3 bucket for project files
    s3 = boto3.client('s3')
    bucket_name = f'arc-challenge-{int(time.time())}'
    
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
        )
        print(f"✅ Created S3 bucket: {bucket_name}")
        return bucket_name
    except Exception as e:
        print(f"❌ Failed to create S3 bucket: {e}")
        return None

def main():
    """Main setup function."""
    
    print("🚀 Setting up AWS environment for ARC Challenge")
    print("=" * 60)
    
    # 1. Create security group
    print("🔒 Creating security group...")
    sg_id = create_security_group()
    
    # 2. Create EC2 instance
    print("🖥️ Creating EC2 instance...")
    instance_id = create_ec2_instance()
    
    if instance_id:
        print(f"\n✅ AWS setup completed!")
        print(f"📋 Instance ID: {instance_id}")
        print(f"🔗 Connect via SSH: ssh -i arc-challenge-key.pem ec2-user@<instance-ip>")
        print(f"💻 Instance type: g4dn.xlarge (GPU enabled)")
        print(f"💰 Estimated cost: ~$0.50/hour")
        
        # Save configuration
        config = {
            'instance_id': instance_id,
            'security_group_id': sg_id,
            'created_at': time.time(),
            'status': 'running'
        }
        
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📄 Configuration saved to aws_config.json")
        
    else:
        print("❌ AWS setup failed!")

if __name__ == "__main__":
    main() 