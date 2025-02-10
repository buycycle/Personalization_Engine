import pandas as pd
import configparser
from buycycle.data import (
    sql_db_read,
)

# Define the list of words to remove
words_to_remove = ["Disc", "Rim", "Mechanical", "Hydraulic"]
# Define your SQL queries
family_model_name_query = """
SELECT name as family_model_name
FROM family_models
"""
component_names_query = """
SELECT name as component_name
FROM bike_components
"""
config_paths = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_paths)
# Read data from the database
family_model_name_df = sql_db_read(
    query=family_model_name_query,
    DB="DB_BIKES",
    config_paths=config_paths,
)
component_names_df = sql_db_read(
    query=component_names_query,
    DB="DB_BIKES",
    config_paths=config_paths,
)
# Create a new DataFrame to store all substrings of component names
component_substrings_list = []
# Extract all substrings from component_names
for index, row in component_names_df.iterrows():
    words = row["component_name"].split()
    for word in words:
        component_substrings_list.append({"component_substring": word})
component_substrings_df = pd.DataFrame(component_substrings_list)
# Convert component substrings to a set of lowercase strings for faster lookup
component_substrings_set = set(component_substrings_df["component_substring"].str.lower())
# Add the words to remove to the set
component_substrings_set.update(word.lower() for word in words_to_remove)


# Function to remove components from a given name while preserving the original case
def remove_components_from_name(name, components):
    words = name.split()
    filtered_words = []
    for word in words:
        # Check if the lowercase version of the word is in the components set
        if word.lower() not in components:
            filtered_words.append(word)
    return " ".join(filtered_words)


# Apply the function to each row in the family_model_name DataFrame
family_model_name_df["cleaned_family_model_name"] = family_model_name_df["family_model_name"].apply(
    lambda x: remove_components_from_name(x, component_substrings_set)
)
# Display the component substrings DataFrame
print("Component Substrings DataFrame:")
print(component_substrings_df)
# Display the cleaned family model names
print("\nCleaned Family Model Names:")
print(family_model_name_df)
family_model_name_df.to_csv("family_model_name_df.csv")
