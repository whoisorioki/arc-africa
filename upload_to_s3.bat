@echo off
REM upload_to_s3.bat
echo Setting AWS credentials...
aws configure set aws_access_key_id AKIA5V23GCAIHHYQBZVC
aws configure set aws_secret_access_key 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
aws configure set default.region us-east-1

echo Uploading data...
aws s3 cp data/ s3://arc-ttt-training-1753897120/data/ --recursive

echo Uploading source code...
aws s3 cp src/ s3://arc-ttt-training-1753897120/src/ --recursive

echo Uploading models...
aws s3 cp models/ s3://arc-ttt-training-1753897120/models/ --recursive

echo Uploading scripts...
aws s3 cp scripts/ s3://arc-ttt-training-1753897120/scripts/ --recursive

echo Uploading requirements...
aws s3 cp requirements.txt s3://arc-ttt-training-1753897120/
aws s3 cp requirements_cpu.txt s3://arc-ttt-training-1753897120/

echo Uploading results...
aws s3 cp ttt_results_*.json s3://arc-ttt-training-1753897120/results/

echo Upload complete!
pause
