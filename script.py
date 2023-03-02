import bs4
import nltk
import requests
from bs4 import BeautifulSoup
from langdetect import detect

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from helpers import *

RECIPES_PER_LANGUAGE = 50

print("Scraping English recipes...")

site_request = requests.get("https://allrecipes.com/recipes")
soup = BeautifulSoup(site_request.text, "html.parser")

links = soup.find_all('a')
recipe_links = [link for link in links if
                link.has_key("href") and link['href'].startswith(
                    "https://www.allrecipes.com/recipe/")]  # only keep links which are actually recipes

recipe_text = {"title": [], "ingredients": [], "directions": []}

for i in range(RECIPES_PER_LANGUAGE):
    recipe_request = requests.get(recipe_links[i]["href"])
    recipe_soup = BeautifulSoup(recipe_request.text, "html.parser")

    # extract title
    title = recipe_soup.find("h1", {"id": "article-heading_2-0"})
    recipe_text["title"].append(title.contents[0][1:])

    # extract ingredient list
    ingredients = recipe_soup.find_all("li", {"class": "mntl-structured-ingredients__list-item"})
    ingredients = [ingredient.findChildren("span", recursive=True) for ingredient in ingredients]
    ingredients = [" ".join([ingr.contents[0] if ingr.contents else "" for ingr in ingredient]) for ingredient in
                   ingredients]
    recipe_text["ingredients"].append(ingredients)

    # extract list of directions
    directions = recipe_soup.find_all("li",
                                      {"class": "comp mntl-sc-block-group--LI mntl-sc-block mntl-sc-block-startgroup"})
    directions = [direction.findChildren("p")[0].contents[0][1:-1] for direction in directions]
    recipe_text["directions"].append(directions)

df_recipes_en = pd.DataFrame.from_dict(recipe_text)

print(f"Scraped {RECIPES_PER_LANGUAGE} English recipes!")

print("Scraping French recipes...")

site_request = requests.get("https://www.marmiton.org/recettes/top-internautes.aspx")
soup = BeautifulSoup(site_request.text, "html.parser")

recipe_links = soup.find_all('a', {"class": "recipe-card-link"})

recipe_text = {"title": [], "ingredients": [], "directions": []}

for i in range(RECIPES_PER_LANGUAGE):
    recipe_request = requests.get(recipe_links[i]["href"])
    recipe_soup = BeautifulSoup(recipe_request.text, "html.parser")

    # extract title
    title = recipe_soup.find("h1", {"class": "SHRD__sc-10plygc-0 itJBWW"})
    recipe_text["title"].append(
        str(title.contents[0].string))  # NavigableStrings need to be converted to regular strings to be able to pickle

    # extract ingredients
    ingredients = recipe_soup.find_all("span", {"class": "SHRD__sc-10plygc-0 kWuxfa"})
    ingredients = [ingredient.findChildren("span", recursive=True) for ingredient in ingredients]
    ingredients = [
        " ".join([str(ingr.contents[0].string) if ingr.contents and len(ingr.contents) == 1 else "" for ingr in
                  ingredient]).strip()
        for ingredient in ingredients]
    recipe_text["ingredients"].append(ingredients)

    # extract directions
    directions = recipe_soup.find_all("p", {"class": "RCP__sc-1wtzf9a-3 jFIVDw"})
    directions = [str(direction.contents[0].string) for direction in directions]
    recipe_text["directions"].append(directions)

df_recipes_fr = pd.DataFrame.from_dict(recipe_text)

print(f"Scraped {RECIPES_PER_LANGUAGE} French recipes!")

print("Scraping Spanish recipes...")

site_request = requests.get("https://www.comedera.com/100-platos-comida-espanola/")
soup = BeautifulSoup(site_request.text, "html.parser")

recipe_links = soup.find_all('h3')
recipe_links = [r.contents[1] for r in recipe_links if r.find("a")][:-1]
del recipe_links[42]  # one of the recipes has a different html structure, so ignore it for simplicity

recipe_text = {"title": [], "ingredients": [], "directions": []}
for i in range(RECIPES_PER_LANGUAGE):
    recipe_request = requests.get(recipe_links[i]["href"])
    recipe_soup = BeautifulSoup(recipe_request.text, "html.parser")

    # extract recipe title
    title = str(recipe_links[i].contents[0].string)
    recipe_text["title"].append(title)

    # extract recipe ingredients
    ingredients = recipe_soup.find_all("li", {"class": "wprm-recipe-ingredient"})
    ingredients = [ingredient.findChildren("span") for ingredient in ingredients]
    ingredients = [" ".join([ingr.contents[0] if ingr.contents else "" for ingr in ingredient]) for ingredient in
                   ingredients]
    recipe_text["ingredients"].append(ingredients)

    # extract recipe directions
    directions = recipe_soup.find_all("div", {"class": "wprm-recipe-instruction-text"})
    directions = [direction.contents[0] for direction in directions]
    directions = [str(direction.contents[0].string) if type(direction) is bs4.element.Tag else str(direction.string) for
                  direction in directions]  # some directions are already strings and others are Spans
    recipe_text["directions"].append(directions)

df_recipes_sp = pd.DataFrame.from_dict(recipe_text)

print(f"Scraped {RECIPES_PER_LANGUAGE} Spanish recipes!")

print("\nDetecting languages...")

# store all recipes in a single dataframe
df_recipes = pd.concat([df_recipes_en, df_recipes_fr, df_recipes_sp], ignore_index=True)

detected_languages = []
for directions in df_recipes["directions"].str.join(" "):  # use the directions section to detect the language
    detected_languages.append(detect(directions))

df_recipes["language"] = detected_languages

counts = df_recipes.language.value_counts()

print(f"Found {counts['en']} English recipes, {counts['fr']} French recipes and {counts['es']} Spanish recipes!")

print("\nTranslating recipes...")

MAX_TRANS_LEN = 5000  # googletrans only accepts texts of length up to 5000

# lists to store translated sections
translated_title = []
translated_ingredients = []
translated_directions = []

for i, recipe in df_recipes.iterrows():
    if i > 0 and i % 30 == 0:
        print(f"Translated {int(i / 1.5)}% of recipes...")
    if recipe["language"] != "en":  # only translate non-english text
        # translate title
        translate_chunk(recipe["title"], translated_title, recipe["language"])

        # translate list of ingredients
        joined_ingredients = ", ".join(recipe["ingredients"])
        translate_chunk(joined_ingredients, translated_ingredients, recipe["language"])

        # translate list of directions
        joined_directions = ". ".join(recipe["directions"])
        if len(joined_directions) <= MAX_TRANS_LEN:
            translate_chunk(joined_directions, translated_directions, recipe["language"])
        else:
            translated = []
            for i in range((
                                   len(joined_directions) // MAX_TRANS_LEN) + 1):  # break up text into chunks of length 5000 and translate one by one
                translate_chunk(joined_directions[MAX_TRANS_LEN * i:MAX_TRANS_LEN * (i + 1)], translated,
                                recipe["language"])
            translated_directions.append(''.join(translated))
    else:
        translated_title.append(recipe["title"])
        translated_ingredients.append(", ".join(recipe["ingredients"]))
        translated_directions.append(". ".join(recipe["directions"]))

df_recipes["translated_title"] = translated_title
df_recipes["translated_ingredients"] = translated_ingredients
df_recipes["translated_directions"] = translated_directions

print(f"Translated 100% of recipes.")

print("\nCleaning text...")

df_recipes["translated_text"] = df_recipes["translated_title"] + ". " + df_recipes["translated_ingredients"] + ". " + \
                                df_recipes["translated_directions"]

preprocess_section(SECTIONS[0], df_recipes, freq_thresh_upper=1, freq_thresh_lower=0)
for section in SECTIONS[1:]:
    preprocess_section(section, df_recipes)

print(
    f"Cleaning done! Total text length was divided by {len(df_recipes.translated_text.str.cat(sep=' ')) / len(df_recipes.preprocessed_text.str.cat(sep=' ')):.1f}.")

print("\nClustering recipes...")

clusters = get_clusters("ingredients", df_recipes, 2)

print("Found 2 clusters of recipes! One of them corresponds to salty recipes and the other to sweet recipes...")
print("Cluster 1 contains recipes such as:\n{0}\n".format('\n'.join(df_recipes.translated_title[clusters==0].sample(5).tolist())))
print("Cluster 2 contains recipes such as:\n{0}\n".format('\n'.join(df_recipes.translated_title[clusters==1].sample(5).tolist())))
