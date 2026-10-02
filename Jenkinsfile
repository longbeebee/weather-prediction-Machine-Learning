pipeline {
    agent any

    parameters {
        choice(
            name: 'CANARY_ACTION',
            choices: ['DEPLOY_CANARY', 'SET_WEIGHT', 'PROMOTE', 'ROLLBACK'],
            description: 'Short-lived canary operation. Monitoring happens outside this build.'
        )
        string(name: 'CHAMPION_WEIGHT', defaultValue: '90', description: 'ALB weight for champion')
        string(name: 'CANDIDATE_WEIGHT', defaultValue: '10', description: 'ALB weight for candidate')
        string(name: 'CANDIDATE_RELEASE_ID', defaultValue: '', description: 'MLflow parent run/release id to deploy or promote')
    }

    environment {
        AWS_REGION = credentials('aws-region')
        APP_HOST = credentials('weather-app-host')
        APP_USER = credentials('weather-app-user')
        APP_DIR = credentials('weather-app-dir')
        ECR_REGISTRY = credentials('weather-ecr-registry')
        ECR_REPOSITORY = credentials('weather-ecr-repository')
        S3_STORAGE_BUCKET = credentials('s3-storage-bucket')
        ALB_LISTENER_ARN = credentials('weather-alb-listener-arn')
        CHAMPION_TG_ARN = credentials('weather-champion-target-group-arn')
        CANDIDATE_TG_ARN = credentials('weather-candidate-target-group-arn')
    }

    stages {
        stage('Validate parameters') {
            steps {
                script {
                    def champion = params.CHAMPION_WEIGHT as Integer
                    def candidate = params.CANDIDATE_WEIGHT as Integer
                    if (champion < 0 || candidate < 0 || champion > 999 || candidate > 999 || champion + candidate == 0) {
                        error('ALB weights must be between 0 and 999 and cannot both be zero')
                    }
                    env.EFFECTIVE_CHAMPION_WEIGHT = params.CANARY_ACTION == 'ROLLBACK' ? '100' : params.CHAMPION_WEIGHT
                    env.EFFECTIVE_CANDIDATE_WEIGHT = params.CANARY_ACTION == 'ROLLBACK' ? '0' : params.CANDIDATE_WEIGHT
                    if (['DEPLOY_CANARY', 'PROMOTE'].contains(params.CANARY_ACTION) && !(params.CANDIDATE_RELEASE_ID ==~ /^[A-Za-z0-9_-]+$/)) {
                        error('CANDIDATE_RELEASE_ID is required and must contain only letters, numbers, underscores, or hyphens')
                    }
                }
            }
        }

        stage('Build and push ECR image') {
            when {
                expression { params.CANARY_ACTION == 'DEPLOY_CANARY' }
            }
            steps {
                sh '''
                    aws ecr get-login-password --region "$AWS_REGION" \\
                      | docker login --username AWS --password-stdin "$ECR_REGISTRY"
                    docker build -t "$ECR_REGISTRY/$ECR_REPOSITORY:$GIT_COMMIT" .
                    docker push "$ECR_REGISTRY/$ECR_REPOSITORY:$GIT_COMMIT"
                '''
            }
        }

        stage('Deploy canary') {
            when {
                expression { params.CANARY_ACTION == 'DEPLOY_CANARY' }
            }
            steps {
                sshagent(credentials: ['weather-app-ssh-key']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=accept-new "$APP_USER@$APP_HOST" \\
                          "cd '$APP_DIR' && aws ecr get-login-password --region '$AWS_REGION' | docker login --username AWS --password-stdin '$ECR_REGISTRY' && WEATHER_API_IMAGE='$ECR_REGISTRY/$ECR_REPOSITORY:$GIT_COMMIT' CANARY_MODEL_MANIFEST='/app/models/seven_day_production/releases/$CANDIDATE_RELEASE_ID/candidate_serving_manifest.json' docker compose pull weather-api-canary && WEATHER_API_IMAGE='$ECR_REGISTRY/$ECR_REPOSITORY:$GIT_COMMIT' CANARY_MODEL_MANIFEST='/app/models/seven_day_production/releases/$CANDIDATE_RELEASE_ID/candidate_serving_manifest.json' docker compose up -d --no-build weather-api-canary && (timeout 90 bash -c 'until curl -fsS http://localhost:8001/ready; do sleep 2; done' || { docker compose logs --tail=200 weather-api-canary; exit 1; })"
                    '''
                }
            }
        }

        stage('Set ALB weight') {
            when {
                anyOf {
                    expression { params.CANARY_ACTION == 'DEPLOY_CANARY' }
                    expression { params.CANARY_ACTION == 'SET_WEIGHT' }
                    expression { params.CANARY_ACTION == 'ROLLBACK' }
                }
            }
            steps {
                sh '''
                    AWS_REGION="$AWS_REGION" \\
                    ALB_LISTENER_ARN="$ALB_LISTENER_ARN" \\
                    CHAMPION_TG_ARN="$CHAMPION_TG_ARN" \\
                    CANDIDATE_TG_ARN="$CANDIDATE_TG_ARN" \\
                    ./scripts/set_alb_weights.sh "$EFFECTIVE_CHAMPION_WEIGHT" "$EFFECTIVE_CANDIDATE_WEIGHT"
                '''
            }
        }

        stage('Promote candidate') {
            when {
                expression { params.CANARY_ACTION == 'PROMOTE' }
            }
            steps {
                sshagent(credentials: ['weather-app-ssh-key']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=accept-new "$APP_USER@$APP_HOST" \\
                          "cd '$APP_DIR' && docker compose exec -T airflow-scheduler python -m src.production.cli_promote --evaluation-manifest /opt/airflow/models/seven_day_production/releases/$CANDIDATE_RELEASE_ID/evaluation_manifest.json --registry-manifest /opt/airflow/models/seven_day_production/releases/$CANDIDATE_RELEASE_ID/registry_manifest.json --output-dir /opt/airflow/models/seven_day_production --mlflow-tracking-uri http://mlflow:5000 && docker compose exec -T airflow-scheduler python -m src.production.cli_archive --path /opt/airflow/models/seven_day_production/production_manifest.json --path /opt/airflow/models/seven_day_production/champion_manifest.json --s3-bucket \"$S3_STORAGE_BUCKET\" --s3-prefix \"weather-7d/runs/jenkins-${BUILD_NUMBER}/training/production/release=$CANDIDATE_RELEASE_ID\" --region \"$AWS_REGION\" && docker compose up -d --force-recreate weather-api"
                    '''
                }
            }
        }
    }

    post {
        always {
            echo "Canary action completed: ${params.CANARY_ACTION}"
        }
    }
}
