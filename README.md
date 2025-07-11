# Buycycle Pre-Owned Bike Market Recommendation System

## Overview
Buycycle's recommendation system enhances the user experience in the pre-owned bike market by providing personalized bike suggestions. It leverages machine learning to tailor recommendations based on user preferences and behavior, addressing challenges like information inefficiencies and high transaction costs in the used bike market.

## System Architecture
The system is divided into two main components: Content-Based Filtering and Collaborative Filtering. Each component periodically processes data to update models and generate recommendations.

## Content-Based Filtering
This component processes bike data from the database, applies feature engineering, and constructs a similarity matrix. It supports several recommendation strategies:

1. **get_top_n_quality_prefiltered**: Returns top recommendations based on quality, filtered by price, frame size, and family ID.
2. **get_top_n_recommendations**: Provides recommendations for a specific bike ID using a similarity matrix.
3. **get_top_n_recommendations_prefiltered**: Similar to the above but allows prefiltering by features like model and brand.

### Features
- **Categorical Features**: Motor, bike component ID, bike category ID, brand, family ID, etc.
- **Numerical Features**: Price, frame size code, year.
- **Feature Overweighting**: Certain features are given more weight to reflect their importance. Family ID has the highest weight (20x) to ensure same-family bikes are strongly prioritized.

## Collaborative Filtering
This component aggregates user interaction data to update models. It uses implicit feedback to generate recommendations and supports strategies like CollaborativeRandomized and CollaborativeRerank.

### Features
- **User and Item Features**: Includes user ID, bike ID, family ID, rider height, and price.
- **Implicit Feedback**: Weights interactions like views, purchases, and favorites to calculate user preferences.

## Recommendation Strategies
The system supports various strategies to ensure robust recommendations:

1. **ContentMixed**: Content-based filtering with family-first prioritization, followed by brand and similarity-based recommendations
2. **FallbackContentMixed**: Similar to ContentMixed but used as a fallback when primary strategies fail
3. **QualityFilter**: Filters by quality score with support for category, price, rider height, e-bike status, frameset status, and brand
4. **Collaborative**: User-based collaborative filtering
5. **CollaborativeRandomized**: Randomized collaborative filtering for diversity
6. **CollaborativeRerank**: Re-ranks a given list of bikes based on user preferences

### Family-First Prioritization
The ContentMixed and FallbackContentMixed strategies implement a "family-first" approach:
- **Same-family bikes are always shown first** when available
- If there are 5 bikes from the same family and you request 10 recommendations, those 5 will appear first
- Remaining slots are filled with similar bikes based on the similarity matrix
- Family ID has 20x overweighting in similarity calculations to ensure strong family grouping

### Brand Filtering
All content-based strategies (ContentMixed, FallbackContentMixed, and QualityFilter) support brand filtering. When a brand is specified (and not "null"), the system will prioritize bikes from that brand in the recommendations while still maintaining diversity through the similarity matrix and quality scores.

**Auto-detection for Product Pages**: When using the `product_page` strategy (ContentMixed/FallbackContentMixed), if no brand is explicitly provided in the request, the system will automatically detect and use the brand of the viewed bike (bike_id). This ensures that product page recommendations naturally favor the same brand without requiring the client to pass the brand parameter.

**API Usage**:
- Explicit brand filtering: Pass `"brand": "specialized"` to filter by a specific brand
- No brand filtering: Pass `"brand": "null"` or omit the parameter entirely
- Auto-detection: For product_page strategy, omit brand to auto-detect from bike_id

## AB Testing
Implemented with Istio on Kubernetes, allowing for controlled testing of different recommendation models.

## Usage
The system is configured via driver files and a config.ini for database credentials. It supports REST API endpoints for health checks and generating recommendations.

## Setup and Deployment
- **Development Environment**: Set up with `make setup` and `make install`.
- **Docker**: Build and run the application using Docker Compose.
- **Linting and Formatting**: Ensure code quality with `make lint` and `make format`.

## To-do
- Merge `get_top_n_quality_prefiltered_bot` and `get_top_n_quality_prefiltered`.

## Content Based
Create_data.py implements reads in the DB of bikes periodically and saves the queried data to disc.
It then applies feature engineering and constructs the m x n (m = bike_ids with status != new, n= bike_ids with status active)
similarity matrix (default metric is pariwise cosine).

The above descibed data generation and read-in process is done periodically with a pvc cron job (k8s) and thread (app.py).

The model implements three algorithm for content based recommendation.

1. get_top_n_quality_prefiltered
	Returns the top n recommendation based on overall quality (quality_score), filtered by price, frame_size_code and family_id

    Since the click data in the DB is discontinued, we randomized the order of the df until the new quality DB is red.
	Currently this is relevant for the edge case that the bike_id is not yet in the constructed similarity matrix
    The prefiltered features need to be send in the request.

2. get_top_n_recommendations
	Returns the top n recommendations for a bike_id

	Features considered are categorical_features and numerical_features defined in src/driver_content.py
	Based on the constructed similarity matrix

4. get_top_n_recommendations_prefiltered
    Returns the top n recommendations as described above for a bike_id prefiltered by the prefilter_features defined in src/driver_content.py. This enables to prefilter the recommendation by for example model and brand, making the recommendation more specific.
	Currently we only filter for family_id. Currently only one prefilter_feature is supported, multiple lead to docker container performance issues (300ms).

### Features
Finally, get_top_n_recommendations_mix implements a mix of prefilters and generic recommendations. If interveav_prefiltered_general these selected these two lists are interwoven, meaning they are mixed alternatevly. Otherwise the generic bikes are appended to the prefiltered one. The ratio between the two can be determined with the ratio argument. If there are not enough recommendations available, populatity is used to append to the recommendation list.

    Currently these features are use to construct the similarity matrix.
    categorical_features = [
        "motor",
        "bike_component_id",
        "bike_category_id",
        "bike_type",
        "brake_type_code",
        "frame_material_code",
        "shifting_code",
        "color",
    ]
    numerical_features = ["price", "frame_size_code", "year"]

    We overwieight certain features to account for their different relevance.
    numerical_features_to_overweight = ["price", "frame_size_code"]
    numerical_features_overweight_factor = 4
    categorical_features_to_overweight = ["bike_component_id", "bike_category_id", "bike_type"]
    categorical_features_overweight_factor = 2


## Collaborative Filtering

The data generation is implemented periodicaly in create_data.py.
The script checks which data is available on the pod's pvc and compares to the mixpanel based click date on S3.
It then updates the local data and saves it to the pvc, again the model reads this data periodically and retrains the model.

The data generation condenses all click data of unique user_id (distinct_id in the current case) and calculates the feedback according to the weights below.

### Features
    Currently, the features used are:
        user_id = 'distinct_id'
        bike_id = 'bike.id'

    item_features = [
        "family_id",
        "rider_height_min",
        "rider_height_max",
        "price",
    ]

    implicit_feedback = {'Bike_view': 1,
                         'Choose_service': 5,
                         'Choose_shipping_method': 5,
                         'Save_filter': 5,
                         'add_discount': 10,
                         'add_payment_info': 20,
                         'add_shipping_info': 20,
                         'add_to_compare': 3,
                         'add_to_favorite': 10,
                         'ask_question': 10,
                         'begin_checkout': 12,
                         'choose_condition': 0,
                         'click_filter_no_result': 0,
                         'close_login_page': 0,
                         'comment_show_original': 3,
                         'counter_offer': 10,
                         'delete_from_favourites': -2,
                         'home_page_open': 0,
                         'login': 5,
                         'open_login_page': 2,
                         'purchase': 50,
                         'receive_block_comment_pop_up': 0,
                         'recom_bike_view': 3,
                         'register': 10,
                         'remove_bike_from_compare': -1,
                         'request_leasing': 20,
                         'sales_ad_created': 0,
                         'search': 0,
                         'search_without_result': 0,
                         'sell_click_family': 0,
                         'sell_click_template': 0,
                         'sell_condition': 0,
                         'sell_details': 0,
                         'sell_search': 0,
                         'sell_templates': 0,
                         'sell_toggle_components_changed': 0,
                         'sellpage_open': 0,
                         'share_bike': 10,
                         'shop_view': 2,
                         'toggle_language': 2,
                         }
## CollaborativeRandomized
Same as above but draw from a sample of more recommendations and randomize.
## CollaborativeRerank
Rerank a given list of bike_rerank_id according to collaborative filtering model.
## Preference filter
To account for business logic we introduce a preference dict that applies custom user specific filters such as continent_id. Currently this is used to avoid cross continent ebike shipping.
We also filter the recommendations by the user specified preferenes in the df_user_preferences.
These two filters are combined into the preference_mask.

## Bot
The chat bot buyers guide is relying on the QualityFilter strategy that returns slugs of progressivly prefiltered bikes in desc order of quality_score.

## AB test

The AB test is implemented with istio on Kubernetes.
Istio ensures a weighted assignment of users by browser. Istio returns a 'version', if no or unknown --header "version: xx" is sent, that is then saved in local storage of the browser.
Make sure the app_version in the model matches the Kubernetes version.
There are three scenarios:

If traffic should be routed to the dev environment, name them -dev for the version name.
This allows the Load Balancer to route to the dev environment.

    Test two versions:

    Prepare two versions and add -stable, -canary to the app_version

    Define your image tag and name and weight under Values.api.version.
    use the same name as in the model app_version
    Sync Kubernetes.

    After test finished:

    set canary to 0% to stop allocation of new people

    and/or rename Values.api.version.name to force also already assigned to 0%.

    Change model image but keep user assignment:

    This only works if the model input data is not changing.

    Set Values.newVersion to false and change Values.api.version.tag to different value.
    Sync Kubernets.

    Start a new ab test:

    same as 'test two versions'
    The PVCs and cronjobs for the data creation are kept and assigned through meta_name, so change the metaname.
    If data requirements are chaning for the model version first start version with
    0 value and wait until the old data from the pvc is delete and new one created.
    Then set to desired weight.


## Model Logic

### Strategies
Try to apply the specified strategy in the request to get recommendations.
If no stategy is passed, use the default strategy (product page).
If an unknow strategy is passed use the fallback strategy (product_page).
If the specified strategy or fallback strategy leads to < n recommendations, use a ContentMixed to ensure that always enough recommendations are generated.

#### Strategy use-case mapping

    strategy_dict = {
        "product_page": CollaborativeRandomized,
        "braze": Collaborative,
        "homepage": CollaborativeRandomized,
        "FallbackContentMixed": FallbackContentMixed,
        "bot": QualityFilter,
    }


#### Available stategies
1. `FallbackContentMixed`:
   - This strategy is a fallback that uses the content based model.
   - It uses a similarity matrix to find bikes similar to the given bike ID.
   - If not enough similar bikes are found, quality-based recommendations are added.
   - The recommendations are generated by the `get_top_n_recommendations_mix` function, which takes into account various parameters such as family ID, price, and frame size code.
   - The strategy returns a tuple containing the strategy name, a list of recommendations, and an optional error message.
2. `ContentMixed`:
   - Similar to `FallbackContentMixed`, this strategy also mixes content-based recommendations with quality-based recommendations.
   - It uses the same `get_top_n_recommendations_mix` function and parameters.
   - The strategy is designed to return quality-based recommendations if there are too few specific recommendations available.
3. `Collaborative`:
   - This strategy attempts to use collaborative filtering to generate recommendations.
   - It uses the `get_top_n_collaborative` function, which leverages a pre-trained model and a dataset to predict items that a user might like.
   - The strategy returns a tuple with the strategy name, a list of recommendations, and a placeholder for an error message (which is always `None` in this case).
4. `CollaborativeRandomized`:
   - This strategy is a variation of collaborative filtering that includes randomized sampling.
   - It uses the `get_top_n_collaborative_randomized` function to generate recommendations, which introduces randomness to potentially increase the diversity of recommendations.
   - The strategy returns a tuple with the strategy name, a list of recommendations, and an optional error message.
5. `QualityFilter`:
   - This strategy applies filters to the bikes list sorted by quality score and returns the slugs.
   - Currently we consider, category, price and rider_height.
   - The strategy uses the `get_top_n_quality_prefiltered_bot` function to generate recommendations after applying the specified filters.
   - The strategy returns a tuple with the strategy name, a list of recommendations, and an optional error message.

### Usage

Most parameters are determined in the driver_.py files, such as numerical and categorical features to consider and which feature to prefilter for.
The function create_data_model_content allows the overweighing num and cat feature, chose in the driver.
get_data replaces NA values with the median, currently only for 'motor'.

The DB parameters are read-in from a config.ini, this should be moved to a vault based approach.

Currently, the status to recommend bike is active and the similarity metric is eucledian.

### Requirements

* [Docker version 20 or later](https://docs.docker.com/install/#support)

### Setup development environment

We setup the conda development environment with the following command.

- `make setup`

Install requirements

- `make install`

Prepare test data


```bash
python create_data.py data/ test
```


### Lint and formatting

- `make lint`

- `make format`


### Docker

when creating an docker image the data is downloaded and prepared. Build test and production stages and runs tests.

- `docker compose build`

Run app.

- `docker compose up app`


### Driver and Config

src/driver_content.py defines the SQL queries, categorical and numerical features as well as the prefilter_features.
config/config.ini holdes DB credentials.
src/driver_collaborative.py defines the bike and user column, which item and user features to use and the implicit feedback weights.


### Endpoint

REST API

#### Health

	Path: /health
	HTTP Verb: GET

#### Get recommendation

	Path: /recommendation
	HTTP Verb: POST

Description: Create n recommednations given a bike_id, user_id and distinct_id, family_id, price and frame_size_code.
Parameters:

    strategy: str, recommendation strategy to apply
	bike_id: int, bike_id for which recommendations are generated
	user_id: int, if null translated to 0, user_id for which recommednations are generated
	distinct_id: str, distinct_id for which recommednations are generated, mixpanel ID
	n: int, number of recommendations to return
    family_id: int, family id
    price, int, price
    frame_size_code, str, frame size

	{   "strategy": "CollaborativeRandomized",
		"bike_type": 1234,
		"bike_id": 1234,
		"user_id": 123,
        "distinct_id": "abc",
		"n": 4,
        "family_id": 12334,
        "price": 1999,
        "frame_size_code": "56"
    }
Return:

    jsonify({"status": "success", "strategy": strategy, "recommendation": recommendation, "app_name": app_name, "app_version": app_version}),

strategy:

    'product_page': ContentMixed
    'braze': Collaborative
    'homepage': CollaborativeRandomized

recommendation:

    bike_id_1,
    .
    .
    .
    bike_id_n

HTTP:

	200: Sucessfull

    400: Bad request or request argument
    404: Recommendation failed


Example:
```bash
ab
curl -i -X POST https://ab.recommendation.buycycle.com/recommendation \
     -H "Content-Type: application/json" \
     -d '{
           "strategy": "product_page",
           "bike_id": 78041,
           "user_id": 123,
           "distinct_id": "3bf240f7-aead-4227-8538-b204aaa58692",
           "n": 8,
           "family_id": 403,
           "continent_id": 1
         }' \
     --header "version: stable"
dev
curl -i -X POST https://dev.recommendation.buycycle.com/recommendation \
     -H "Content-Type: application/json" \
     -d '{
           "strategy": "product_page",
           "bike_id": 78041,
           "user_id": 123,
           "distinct_id": "3bf240f7-aead-4227-8538-b204aaa58692",
           "n": 8,
           "family_id": 403,
           "continent_id": 1
         }' \
     --header "version: stable-001-dev"


deployed
curl -i -X POST https://recommendation.buycycle.com/recommendation ...                                                                                                              "user_id": 123,

local
curl -i -X POST 0.0.0.0:8000/recommendation ...

```


## To-does:

Merge get_top_n_quality_prefiltered_bot and get_top_n_quality_prefiltered.
