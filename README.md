# risk-engine-docker

1. Sign in to AWS using CLI
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com

1. Build the docker image
docker build -t risk-score-v1 . 

2. Tag the image locally
docker tag risk-score-v1 <account_id>.dkr.ecr.<region>.amazonaws.com/risk-score-v1:latest

3. Push the image to ECR
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/risk-score-v1:latest 

4. To push changes using Git
git add README.md app.py 
git commit -m "JSON Refactor"
git push docker main