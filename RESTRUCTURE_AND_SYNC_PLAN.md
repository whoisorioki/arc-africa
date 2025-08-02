# Project Restructuring and S3 Sync Plan

## 🎯 Overview

This plan outlines the restructuring of the ARC Challenge Africa project for better organization and syncing to AWS S3 bucket `arc-africa-clean-2024`.

## 📁 Current vs. New Structure

### Current Structure (Cluttered)

```
arc-africa/
├── src/                          # Core source code
├── scripts/                      # Mixed scripts
├── data/                         # ARC datasets
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── results/                      # Results
├── logs/                         # Logs
├── validation_results/           # Validation results
├── quick_validation_results/     # Quick validation results
├── validation_train_results/     # Validation training results
├── aws/                          # AWS files
├── wandb/                        # Wandb logs
├── .cursor/                      # Cursor IDE
├── .git/                         # Git repository
├── .venv/                        # Virtual environment
├── .pytest_cache/                # Pytest cache
├── Various loose files           # Configuration, guides, etc.
└── Multiple submission files     # Duplicate submissions
```

### New Structure (Organized)

```
arc-africa/
├── src/                          # Core source code
│   ├── neural_guide/             # Neural guide implementation
│   ├── solver/                   # Solver implementations
│   ├── dsl/                      # Domain-specific language
│   ├── data_pipeline/            # Data processing
│   ├── symbolic_search/          # Symbolic search algorithms
│   └── external/                 # External libraries
├── scripts/                      # Production scripts
│   ├── generate_submission.py    # Main submission generator
│   ├── aws/                      # AWS deployment scripts
│   │   ├── setup_aws_credentials.py
│   │   ├── aws_quick_setup.sh
│   │   └── create_training_script.sh
│   ├── training/                 # Training scripts
│   └── validation/               # Validation scripts
│       └── validate_submission_format.py
├── config/                       # Configuration files
│   ├── config.json              # Main configuration
│   ├── requirements.txt         # Dependencies
│   ├── requirements.in          # Dependencies source
│   └── variable_description.csv # Variable descriptions
├── docs/                         # Documentation
│   ├── guides/                  # User guides
│   │   ├── AWS_DEPLOYMENT_GUIDE.md
│   │   ├── AWS_CREDENTIALS_GUIDE.md
│   │   ├── SUBMISSION_GUIDE.md
│   │   ├── TRAINING_GUIDE.md
│   │   ├── README_COLAB.md
│   │   └── AWS_TTT_INTEGRATION_PLAN.md
│   └── reports/                 # Project reports
│       ├── SOLVER_STATUS_REPORT.md
│       ├── CLEANUP_PLAN.md
│       └── CLEANUP_SUMMARY.md
├── outputs/                      # Output files
│   ├── submissions/             # Generated submissions
│   │   ├── submission.csv
│   │   └── SampleSubmission.csv
│   ├── results/                 # Training results
│   └── logs/                    # Training logs
├── data/                         # ARC datasets
├── models/                       # Trained models
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── README.md                     # Main documentation
└── LICENSE                       # License file
```

## 🗂️ File Movement Plan

### Files to Move to `scripts/aws/`

- `setup_aws_credentials.py`
- `aws_quick_setup.sh`
- `create_training_script.sh`

### Files to Move to `docs/guides/`

- `AWS_DEPLOYMENT_GUIDE.md`
- `AWS_CREDENTIALS_GUIDE.md`
- `SUBMISSION_GUIDE.md`
- `TRAINING_GUIDE.md`
- `README_COLAB.md`
- `AWS_TTT_INTEGRATION_PLAN.md`

### Files to Move to `docs/reports/`

- `SOLVER_STATUS_REPORT.md`
- `CLEANUP_PLAN.md`
- `CLEANUP_SUMMARY.md`

### Files to Move to `config/`

- `config.json`
- `requirements.txt`
- `requirements.in`
- `variable_description.csv`

### Files to Move to `outputs/submissions/`

- `submission.csv`
- `SampleSubmission.csv`

### Files to Move to `scripts/validation/`

- `validate_submission_format.py`

### Directories to Remove

- `.cursor/` - Cursor IDE files
- `.pytest_cache/` - Pytest cache
- `.venv/` - Virtual environment
- `wandb/` - Wandb logs
- `aws/` - Old AWS directory
- `validation_results/` - Moved to outputs
- `quick_validation_results/` - Moved to outputs
- `validation_train_results/` - Moved to outputs
- `results/` - Moved to outputs/results
- `logs/` - Moved to outputs/logs

## 🚀 Implementation Steps

### Step 1: Restructure the Project

```bash
python restructure_project.py
```

This will:

- Create new directory structure
- Move files to appropriate locations
- Remove unnecessary directories
- Update README.md with new structure

### Step 2: Prepare S3 Sync

```bash
python sync_to_s3.py
```

This will:

- Create sync scripts (`sync_to_s3.sh` and `sync_to_s3.py`)
- Update `.gitignore` with comprehensive patterns
- Create `.s3ignore` file for S3 exclusions

### Step 3: Sync to S3

```bash
./sync_to_s3.sh
```

This will:

- Perform dry run first
- Ask for confirmation
- Sync to `s3://arc-africa-clean-2024/`
- Exclude development files

## 🚫 Files to Exclude from S3

### Development Files

- `.cursor/` - Cursor IDE
- `.git/` - Git repository
- `.aws/` - AWS configuration
- `.pytest_cache/` - Pytest cache
- `__pycache__/` - Python cache
- `*.pyc`, `*.pyo`, `*.pyd` - Compiled Python files

### Virtual Environments

- `.venv/`, `venv/`, `env/`, `ENV/` - Virtual environments

### IDE Files

- `.vscode/`, `.idea/` - IDE configurations
- `*.swp`, `*.swo`, `*~` - Editor temporary files

### OS Files

- `.DS_Store` - macOS
- `Thumbs.db` - Windows

### Large Files

- `*.pth`, `*.pt` - PyTorch models
- `*.h5`, `*.hdf5` - HDF5 files
- `*.pkl`, `*.pickle` - Pickle files

### Temporary Files

- `*.tmp`, `*.temp`, `*.log`, `*.bak`
- `*.backup`, `*.old`, `*~`

### AWS Credentials

- `aws_credentials.json`, `aws_config.json`

### Results and Logs

- `outputs/logs/*` - Log files (keep structure)
- `outputs/results/*` - Result files (keep structure)

## 📋 S3 Bucket Details

- **Bucket Name**: `arc-africa-clean-2024`
- **Region**: Default AWS region
- **Access**: Requires appropriate AWS credentials
- **URL**: `https://s3.console.aws.amazon.com/s3/buckets/arc-africa-clean-2024`

## 🔧 Prerequisites

### AWS Setup

1. Install AWS CLI: `pip install awscli`
2. Configure AWS credentials: `aws configure`
3. Verify access to bucket: `aws s3 ls s3://arc-africa-clean-2024`

### Python Dependencies

```bash
pip install boto3
```

## ✅ Success Criteria

### Restructuring Success

- [ ] All files moved to appropriate directories
- [ ] Unnecessary directories removed
- [ ] README.md updated with new structure
- [ ] Project structure is clean and organized

### S3 Sync Success

- [ ] All essential files uploaded to S3
- [ ] Development files excluded
- [ ] Directory structure maintained in S3
- [ ] No sensitive information uploaded

### Verification

- [ ] Project can be downloaded from S3 and run
- [ ] All documentation links work
- [ ] Configuration files are in correct locations
- [ ] Scripts reference correct paths

## 🎯 Benefits

### Organization Benefits

1. **Clear Structure**: Logical grouping of files by purpose
2. **Easy Navigation**: Intuitive directory hierarchy
3. **Maintainable**: Easier to find and update files
4. **Professional**: Clean, production-ready appearance

### S3 Benefits

1. **Backup**: Secure cloud storage of project
2. **Sharing**: Easy sharing with team members
3. **Version Control**: S3 versioning for file history
4. **Access Control**: Fine-grained permissions

### Development Benefits

1. **Cleaner Development**: No clutter from unnecessary files
2. **Faster Operations**: Reduced file scanning time
3. **Better Collaboration**: Clear structure for team members
4. **Easier Deployment**: Organized structure for deployment

## 🚨 Important Notes

1. **Backup**: Ensure you have a backup before restructuring
2. **Testing**: Test the restructured project locally before S3 sync
3. **Credentials**: Never upload AWS credentials to S3
4. **Large Files**: Model files (`.pth`, `.pt`) are excluded from S3
5. **Documentation**: Update any hardcoded paths in documentation

## 📞 Support

If issues arise during restructuring or S3 sync:

1. Check the generated scripts for error messages
2. Verify AWS credentials and permissions
3. Ensure all prerequisites are installed
4. Review the exclusion patterns if files are missing
