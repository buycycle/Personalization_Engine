# Local and Development

setup:
	conda create  --name recommendation --file requirements.txt python=3.9

activate:
	conda activate .env/recommendation

juypter:
	@cd notebook; PYTHONPATH=".." jupyter notebook notebook.ipynb

test_data:
	python create_data.py './data/' test
integration:
	## run integration tests
	pytest tests/test_fastapi.py
test:
	pytest tests/test_fastapi.py
lint:
	pylint --disable=R,C,W1203,W1202 src/
	pylint --disable=R,C,W1203,W1202 model/app.py



	# check style with flake8
	mypy .
	flake8 .

format:
	# Check Python formatting
	black --line-length 130 .



build-docker:
	## Run Docker locally

	docker compose build

	docker compose up app


all: install lint test

# Deployment

APP_NAME := recommendation-api-ab-test
VALUES_FILE = values.yaml
CLONE_DIR := /tmp/$(APP_NAME)/$(ENV)
REPO_NAME := buycycle-helm
TARGET_DIR := $(CLONE_DIR)/$(REPO_NAME)
GIT_USERNAME := Jenkins
GIT_EMAIL := jenkins@buycycle.de
BRANCH := main
REPO_URL := git@gitlab.com:buycycle

configure-git:
	git config --global user.name $(GIT_USERNAME)
	git config --global user.email $(GIT_EMAIL)

clone: clear configure-git
	mkdir -p $(CLONE_DIR) && cd $(CLONE_DIR) && git clone $(REPO_URL)/$(REPO_NAME).git

clear:
	test -d $(TARGET_DIR) && rm -R -f $(TARGET_DIR) || true

modify:
	yq -i ".image.tag = \"$(IMAGE_TAG)\"" $(TARGET_DIR)/$(ENV)/$(APP_NAME)/$(VALUES_FILE)

push: clone modify
	cd $(TARGET_DIR) && git add $(ENV)/$(APP_NAME)/$(VALUES_FILE) && git commit -m "updated during build $(APP_NAME) $(IMAGE_TAG)" && git pull --rebase && git push -u origin $(BRANCH)
