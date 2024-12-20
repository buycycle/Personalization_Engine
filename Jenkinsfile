// Define the mappings
def map_branch_to_env = [
    "dev": "dev",
    "staging": "staging",
    "main": "live"
]
def map_branch_to_ab = [
    "dev": "canary",
    "staging": "canary",
    "main": "stable"
]
// Set dev as default
def image_tag = "dev-${env.BUILD_NUMBER}"
def environment = "dev"
def ab = "canary"
image_tag = "${env.BRANCH_NAME}-${env.BUILD_NUMBER}"
environment = "${map_branch_to_env[env.BRANCH_NAME]}"
ab = "${map_branch_to_ab[env.BRANCH_NAME]}"
// Simple switch to control skipping the test execution, default is false
def skipTests = false
pipeline {
    agent any
    options {
        disableConcurrentBuilds()
    }
    stages {
        stage('Build') {
            when {
                not { changeset pattern: "Jenkinsfile" }
                not { changeset pattern: "Makefile" }
            }
            steps {
                withCredentials([string(credentialsId: 'recommendation-s3-path-credential', variable: 'S3_CONFIG_PATH')]) {
                    sh "aws s3 cp ${S3_CONFIG_PATH} config/config.ini"
                }
                script {
                    if (environment == "live") {
                        app = docker.build("recommendation-api", "-f docker/main.dockerfile --build-arg ENVIRONMENT=${environment} --build-arg AB=${ab} .")
                    } else {
                        app = docker.build("recommendation-api", "-f docker/dev.dockerfile --build-arg ENVIRONMENT=${environment} --build-arg AB=${ab} .")
                    }
                    test = docker.build("recommendation-api-test", "-f docker/test.dockerfile .")
                }
            }
        }
        stage('Test') {
            when {
                not { changeset pattern: "Jenkinsfile" }
                not { changeset pattern: "Makefile" }
                expression { !skipTests } // Only run tests if skipTests is false
            }
            steps {
                script {
                    test.inside {
                        sh 'mkdir -p data'
                        sh 'python create_data.py ./data/ test'
                        sh 'pytest -v tests/test_fastapi.py'
                    }
                }
            }
        }
        stage('Push Docker image') {
            when {
                not { changeset pattern: "Jenkinsfile" }
                not { changeset pattern: "Makefile" }
            }
            steps {
                withCredentials([string(credentialsId: 'recommendation-ecr-url', variable: 'ECR_URL')]) {
                    script {
                        docker.withRegistry("${ECR_URL}", 'ecr:eu-central-1:aws-credentials-ecr') {
                            app.push(image_tag)
                            app.push("latest")
                        }
                    }
                }
            }
        }
        stage("Modify HELM chart") {
            when {
                not { changeset pattern: "Jenkinsfile" }
                not { changeset pattern: "Makefile" }
            }
            steps {
                sh "make push IMAGE_TAG=${image_tag} ENV=${environment}"
            }
        }
        stage("Sync Chart") {
            when {
                not { changeset pattern: "Jenkinsfile" }
                not { changeset pattern: "Makefile" }
            }
            steps {
                withCredentials([string(credentialsId: 'argocd-token', variable: 'TOKEN')]) {
                    script {
                        env.namespace = environment
                    }
                    sh '''
                      set +x
                      argocd app sync recommendation-api-ab-test-$namespace --server argocd.cube-gebraucht.com --auth-token $TOKEN
                      argocd app wait recommendation-api-ab-test-$namespace --server argocd.cube-gebraucht.com --auth-token $TOKEN
                    '''
                }
            }
        }
    }
}
