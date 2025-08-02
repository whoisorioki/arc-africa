#!/usr/bin/env python3
"""
Sync the restructured ARC Challenge Africa project to AWS S3.

This script will:
1. Upload the cleaned project structure to S3
2. Ignore development files (.git, .cursor, .aws, etc.)
3. Maintain proper organization in the S3 bucket
"""

import os
import boto3
import subprocess
import sys
from pathlib import Path

def create_s3_sync_script():
    """Create an S3 sync script that ignores unnecessary files."""
    
    sync_script = """#!/bin/bash

# S3 Sync Script for ARC Challenge Africa Project
# This script syncs the project to S3 while ignoring development files

BUCKET_NAME="arc-africa-clean-2024"
LOCAL_DIR="."
S3_PREFIX=""

echo "🔄 Syncing ARC Challenge Africa project to S3..."

# Create .s3ignore file if it doesn't exist
cat > .s3ignore << 'EOF'
# Development and IDE files
.cursor/
.git/
.aws/
.pytest_cache/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Virtual environments
.venv/
venv/
env/
ENV/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
*.log
*.bak

# AWS credentials and config
.aws/
aws_credentials.json
aws_config.json

# Large files that shouldn't be in S3
*.pth
*.pt
*.h5
*.hdf5
*.pkl
*.pickle

# Results and logs (keep structure but not content)
outputs/logs/*
outputs/results/*
!outputs/logs/.gitkeep
!outputs/results/.gitkeep

# Notebook outputs
notebooks/.ipynb_checkpoints/

# Test artifacts
.pytest_cache/
.coverage
htmlcov/

# Documentation build artifacts
docs/_build/
docs/build/

# Backup files
*.backup
*.old
*~
EOF

echo "✅ Created .s3ignore file"

# Sync to S3 using AWS CLI
echo "📤 Starting S3 sync..."

# Use aws s3 sync with exclude patterns
aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME/$S3_PREFIX" \\
    --exclude ".cursor/*" \\
    --exclude ".git/*" \\
    --exclude ".aws/*" \\
    --exclude ".pytest_cache/*" \\
    --exclude "__pycache__/*" \\
    --exclude "*.pyc" \\
    --exclude "*.pyo" \\
    --exclude "*.pyd" \\
    --exclude ".venv/*" \\
    --exclude "venv/*" \\
    --exclude "env/*" \\
    --exclude "ENV/*" \\
    --exclude ".vscode/*" \\
    --exclude ".idea/*" \\
    --exclude "*.swp" \\
    --exclude "*.swo" \\
    --exclude "*~" \\
    --exclude ".DS_Store" \\
    --exclude "Thumbs.db" \\
    --exclude "*.tmp" \\
    --exclude "*.temp" \\
    --exclude "*.log" \\
    --exclude "*.bak" \\
    --exclude "aws_credentials.json" \\
    --exclude "aws_config.json" \\
    --exclude "*.pth" \\
    --exclude "*.pt" \\
    --exclude "*.h5" \\
    --exclude "*.hdf5" \\
    --exclude "*.pkl" \\
    --exclude "*.pickle" \\
    --exclude "outputs/logs/*" \\
    --exclude "outputs/results/*" \\
    --exclude "notebooks/.ipynb_checkpoints/*" \\
    --exclude ".pytest_cache/*" \\
    --exclude ".coverage" \\
    --exclude "htmlcov/*" \\
    --exclude "docs/_build/*" \\
    --exclude "docs/build/*" \\
    --exclude "*.backup" \\
    --exclude "*.old" \\
    --exclude "*~" \\
    --delete \\
    --dryrun

echo ""
echo "🔍 Dry run completed. Review the changes above."
echo ""
echo "To perform the actual sync, run:"
echo "aws s3 sync . s3://$BUCKET_NAME/ --exclude 'pattern1' --exclude 'pattern2' ... --delete"
echo ""
echo "Or use the interactive sync function below:"
echo ""
read -p "Do you want to perform the actual sync? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Performing actual sync..."
    aws s3 sync "$LOCAL_DIR" "s3://$BUCKET_NAME/$S3_PREFIX" \\
        --exclude ".cursor/*" \\
        --exclude ".git/*" \\
        --exclude ".aws/*" \\
        --exclude ".pytest_cache/*" \\
        --exclude "__pycache__/*" \\
        --exclude "*.pyc" \\
        --exclude "*.pyo" \\
        --exclude "*.pyd" \\
        --exclude ".venv/*" \\
        --exclude "venv/*" \\
        --exclude "env/*" \\
        --exclude "ENV/*" \\
        --exclude ".vscode/*" \\
        --exclude ".idea/*" \\
        --exclude "*.swp" \\
        --exclude "*.swo" \\
        --exclude "*~" \\
        --exclude ".DS_Store" \\
        --exclude "Thumbs.db" \\
        --exclude "*.tmp" \\
        --exclude "*.temp" \\
        --exclude "*.log" \\
        --exclude "*.bak" \\
        --exclude "aws_credentials.json" \\
        --exclude "aws_config.json" \\
        --exclude "*.pth" \\
        --exclude "*.pt" \\
        --exclude "*.h5" \\
        --exclude "*.hdf5" \\
        --exclude "*.pkl" \\
        --exclude "*.pickle" \\
        --exclude "outputs/logs/*" \\
        --exclude "outputs/results/*" \\
        --exclude "notebooks/.ipynb_checkpoints/*" \\
        --exclude ".pytest_cache/*" \\
        --exclude ".coverage" \\
        --exclude "htmlcov/*" \\
        --exclude "docs/_build/*" \\
        --exclude "docs/build/*" \\
        --exclude "*.backup" \\
        --exclude "*.old" \\
        --exclude "*~" \\
        --delete
    
    echo "✅ Sync completed!"
else
    echo "❌ Sync cancelled."
fi

echo ""
echo "🔗 S3 Bucket URL: https://s3.console.aws.amazon.com/s3/buckets/$BUCKET_NAME"
echo "📁 Project structure uploaded to: s3://$BUCKET_NAME/$S3_PREFIX"
"""
    
    with open("sync_to_s3.sh", "w") as f:
        f.write(sync_script)
    
    # Make the script executable
    os.chmod("sync_to_s3.sh", 0o755)
    print("✅ Created sync_to_s3.sh script")

def create_python_sync_script():
    """Create a Python-based S3 sync script."""
    
    python_sync_script = """#!/usr/bin/env python3
\"\"\"
Python S3 Sync Script for ARC Challenge Africa Project
\"\"\"

import boto3
import os
import sys
from pathlib import Path

def sync_to_s3():
    \"\"\"Sync the project to S3 bucket.\"\"\"
    
    bucket_name = "arc-africa-clean-2024"
    local_dir = "."
    s3_prefix = ""
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
        print(f"✅ Connected to S3")
    except Exception as e:
        print(f"❌ Failed to connect to S3: {e}")
        return False
    
    # Files/directories to exclude
    exclude_patterns = [
        ".cursor/*",
        ".git/*", 
        ".aws/*",
        ".pytest_cache/*",
        "__pycache__/*",
        "*.pyc",
        "*.pyo", 
        "*.pyd",
        ".venv/*",
        "venv/*",
        "env/*",
        "ENV/*",
        ".vscode/*",
        ".idea/*",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        "*.tmp",
        "*.temp",
        "*.log",
        "*.bak",
        "aws_credentials.json",
        "aws_config.json",
        "*.pth",
        "*.pt",
        "*.h5",
        "*.hdf5",
        "*.pkl",
        "*.pickle",
        "outputs/logs/*",
        "outputs/results/*",
        "notebooks/.ipynb_checkpoints/*",
        ".pytest_cache/*",
        ".coverage",
        "htmlcov/*",
        "docs/_build/*",
        "docs/build/*",
        "*.backup",
        "*.old",
        "*~"
    ]
    
    print(f"🔄 Syncing to s3://{bucket_name}/{s3_prefix}")
    print(f"📁 Local directory: {os.path.abspath(local_dir)}")
    
    # Build aws s3 sync command
    cmd = [
        "aws", "s3", "sync",
        local_dir,
        f"s3://{bucket_name}/{s3_prefix}",
        "--delete"
    ]
    
    # Add exclude patterns
    for pattern in exclude_patterns:
        cmd.extend(["--exclude", pattern])
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sync completed successfully!")
            print(result.stdout)
        else:
            print("❌ Sync failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during sync: {e}")
        return False
    
    print(f"🔗 S3 Bucket URL: https://s3.console.aws.amazon.com/s3/buckets/{bucket_name}")
    print(f"📁 Project uploaded to: s3://{bucket_name}/{s3_prefix}")
    
    return True

if __name__ == "__main__":
    sync_to_s3()
"""
    
    with open("sync_to_s3.py", "w") as f:
        f.write(python_sync_script)
    
    # Make the script executable
    os.chmod("sync_to_s3.py", 0o755)
    print("✅ Created sync_to_s3.py script")

def create_gitignore():
    """Create a comprehensive .gitignore file."""
    
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# AWS
.aws/
aws_credentials.json
aws_config.json

# Model files (large)
*.pth
*.pt
*.h5
*.hdf5
*.pkl
*.pickle

# Temporary files
*.tmp
*.temp
*.bak
*.backup
*.old

# Cursor IDE
.cursor/

# Project specific
outputs/logs/*
outputs/results/*
!outputs/logs/.gitkeep
!outputs/results/.gitkeep

# Wandb
wandb/

# Local development
local_*
debug_*
test_*
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ Updated .gitignore file")

def main():
    """Main function to prepare S3 sync."""
    print("🔄 Preparing S3 sync for ARC Challenge Africa project...")
    
    # Create sync scripts
    create_s3_sync_script()
    create_python_sync_script()
    
    # Create comprehensive .gitignore
    create_gitignore()
    
    print("\n✅ S3 sync preparation complete!")
    print("\n📋 Available sync options:")
    print("1. Bash script: ./sync_to_s3.sh")
    print("2. Python script: python sync_to_s3.py")
    print("3. Manual AWS CLI: aws s3 sync . s3://arc-africa-clean-2024/ --exclude 'pattern' --delete")
    
    print("\n🔧 Before syncing:")
    print("1. Ensure AWS CLI is installed and configured")
    print("2. Verify you have access to the S3 bucket: arc-africa-clean-2024")
    print("3. Run the restructuring script first: python restructure_project.py")
    
    print("\n🚀 To sync:")
    print("1. First restructure: python restructure_project.py")
    print("2. Then sync: ./sync_to_s3.sh")

if __name__ == "__main__":
    main() 