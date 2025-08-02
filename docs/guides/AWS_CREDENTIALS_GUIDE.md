# AWS Credentials Setup Guide for ARC Challenge

## 🔑 Quick Setup

### **Option 1: Automated Setup (Recommended)**

```bash
# Run the automated setup script
python setup_aws_credentials.py
```

### **Option 2: Manual Setup**

## 📋 Step-by-Step Manual Setup

### **1. Get Your AWS Access Keys**

1. **Go to AWS Console**: https://console.aws.amazon.com/
2. **Navigate to IAM**: Services → IAM → Users → Your User
3. **Security credentials tab**
4. **Create access key** → **Command Line Interface (CLI)**
5. **Download the CSV file** with your keys

**Important**: Keep your access keys secure and never share them!

### **2. Install AWS CLI**

#### **Windows:**

```bash
# Download and install from:
# https://awscli.amazonaws.com/AWSCLIV2.msi

# Or use command line:
msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
```

#### **macOS/Linux:**

```bash
# Download and install
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### **3. Configure AWS Credentials**

```bash
# Configure your credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: us-east-1
# Default output format: json
```

### **4. Test Your Configuration**

```bash
# Test AWS connection
aws sts get-caller-identity

# List your EC2 instances
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]' --output table
```

### **5. Create SSH Key Pair**

```bash
# Create SSH key pair for EC2 access
aws ec2 create-key-pair --key-name arc-challenge-key --query 'KeyMaterial' --output text > arc-challenge-key.pem

# Set correct permissions (Linux/Mac)
chmod 400 arc-challenge-key.pem
```

## 🚀 Quick Commands

### **Check AWS CLI Version**

```bash
aws --version
```

### **Configure Credentials (Alternative Method)**

```bash
# Set credentials individually
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-east-1
aws configure set default.output json
```

### **List Your Resources**

```bash
# List EC2 instances
aws ec2 describe-instances

# List S3 buckets
aws s3 ls

# List IAM users
aws iam list-users
```

## 🔧 Troubleshooting

### **Common Issues**

1. **"aws: command not found"**

   - Install AWS CLI first
   - Add AWS CLI to your PATH

2. **"Access Denied"**

   - Check your IAM permissions
   - Verify your access keys are correct
   - Ensure your user has EC2 permissions

3. **"Invalid credentials"**
   - Double-check your access key and secret key
   - Make sure you copied them correctly
   - Check if your keys are expired

### **IAM Permissions Required**

Your AWS user needs these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ec2:*", "s3:*", "iam:ListUsers"],
      "Resource": "*"
    }
  ]
}
```

## 📁 File Locations

### **AWS Credentials Location**

- **Windows**: `%UserProfile%\.aws\credentials`
- **Linux/Mac**: `~/.aws/credentials`

### **SSH Key Location**

- **Windows**: `C:\Users\YourUsername\arc-challenge-key.pem`
- **Linux/Mac**: `~/arc-challenge-key.pem`

## 🎯 Next Steps

Once your credentials are configured:

1. **Test connection**: `aws sts get-caller-identity`
2. **Create EC2 instance**: `python aws_setup_arc.py`
3. **Upload files**: `python upload_to_aws.py`
4. **Start training**: SSH to your instance and run training

## 🔒 Security Best Practices

1. **Never commit credentials to git**
2. **Use IAM roles when possible**
3. **Rotate access keys regularly**
4. **Use least privilege principle**
5. **Enable MFA for your AWS account**

---

**Need help?** Check the AWS CLI documentation: https://docs.aws.amazon.com/cli/
