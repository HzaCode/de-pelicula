
import os
from dotenv import load_dotenv
from openai import OpenAI, APIError
from flask import Flask, jsonify, render_template
import syllables
import json
import logging
import requests

app = Flask(__name__)

MAX_TOKENS = 350
TEMPERATURE = 0.7
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_CONFIG = None
DEFAULT_POSTER_SIZE = "w342"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(funcName)s: %(message)s')

load_dotenv()

DEEPSEEK_API_KEY_NAME = 'DEEPSEEK_API_KEY'
DEEPSEEK_API_KEY_VALUE = os.environ.get(DEEPSEEK_API_KEY_NAME)
if not DEEPSEEK_API_KEY_VALUE:
    logging.error(f"{DEEPSEEK_API_KEY_NAME} not found in environment variables.")
    raise ValueError(f"{DEEPSEEK_API_KEY_NAME} not found. Please set it in your .env file.")
else:
    masked_key = DEEPSEEK_API_KEY_VALUE[:5] + "****" + DEEPSEEK_API_KEY_VALUE[-4:]
    logging.info(f"Loaded DeepSeek API Key ({DEEPSEEK_API_KEY_NAME}): {masked_key}")

TMDB_API_KEY_NAME = 'TMDB_API_ACCESS_TOKEN'
TMDB_ACCESS_TOKEN = os.environ.get(TMDB_API_KEY_NAME)
if not TMDB_ACCESS_TOKEN:
    logging.warning(f"{TMDB_API_KEY_NAME} not found in environment variables. Movie posters cannot be fetched.")
else:
     logging.info(f"Loaded TMDB Access Token ({TMDB_API_KEY_NAME}).")

deepseek_client = None
try:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY_VALUE,
        base_url=DEEPSEEK_BASE_URL,
    )
    logging.info(f"OpenAI client initialized successfully for DeepSeek API (Base URL: {DEEPSEEK_BASE_URL}).")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client for DeepSeek: {e}", exc_info=True)
    raise

def fetch_tmdb_config():
    global TMDB_CONFIG
    if not TMDB_ACCESS_TOKEN:
        logging.warning("TMDB Access Token not available, skipping config fetch.")
        return None

    url = f"{TMDB_BASE_URL}/configuration"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        config_data = response.json()
        if "images" in config_data and "secure_base_url" in config_data["images"]:
             TMDB_CONFIG = config_data["images"]
             if DEFAULT_POSTER_SIZE not in TMDB_CONFIG.get("poster_sizes", []):
                 logging.warning(f"Default poster size '{DEFAULT_POSTER_SIZE}' not found in TMDB config. Available: {TMDB_CONFIG.get('poster_sizes')}. Check TMDB API.")
             logging.info(f"Successfully fetched TMDB configuration. Image base URL: {TMDB_CONFIG['secure_base_url']}")
             return TMDB_CONFIG
        else:
            logging.error("TMDB configuration fetched but missing 'images' or 'secure_base_url'.")
            TMDB_CONFIG = None
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching TMDB configuration: {e}", exc_info=True)
        TMDB_CONFIG = None
        return None

def search_movie_poster(movie_title):
    if not TMDB_ACCESS_TOKEN or not TMDB_CONFIG or not TMDB_CONFIG.get('secure_base_url'):
        logging.warning("TMDB config or token missing, cannot search for poster.")
        return None

    search_url = f"{TMDB_BASE_URL}/search/movie"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"
    }
    params = {
        "query": movie_title,
        "include_adult": "false",
        "language": "en-US",
        "page": 1
    }

    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()

        if results and results.get("results"):
            first_movie = results["results"][0]
            poster_path = first_movie.get("poster_path")
            if poster_path:
                full_poster_url = f"{TMDB_CONFIG['secure_base_url']}{DEFAULT_POSTER_SIZE}{poster_path}"
                logging.info(f"Found poster for '{movie_title}': {full_poster_url}")
                return full_poster_url
            else:
                logging.warning(f"Movie '{movie_title}' found on TMDB, but it has no poster path.")
                return None
        else:
            logging.warning(f"No results found on TMDB for movie title: '{movie_title}'")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching TMDB for '{movie_title}': {e}", exc_info=True)
        return None
    except IndexError:
         logging.warning(f"TMDB search for '{movie_title}' returned empty results list.")
         return None

fetch_tmdb_config()

full_prompt = """   .
Your task is to .

Instructions:
1. 
2. 
3. 
4. 
   - 
   - 
   -
5. 

Example format (ONLY the JSON object):
{

  
}

, formatted strictly as the JSON object described.
"""

@app.route('/')
def index():
    logging.info("Serving index.html")
    return render_template('index.html')

@app.route('/api/translate', methods=['GET'])
def get_quote():
    logging.info("Received request for /api/translate")

    if not deepseek_client:
         logging.error("DeepSeek API Client not initialized.")
         return jsonify({"error": "API Client (DeepSeek) not available"}), 503

    if DEEPSEEK_MODEL_NAME == "PLEASE_REPLACE_WITH_ACTUAL_DEEPSEEK_MODEL_NAME" or not DEEPSEEK_MODEL_NAME:
        logging.error("DEEPSEEK_MODEL_NAME is not set or still a placeholder in code.")
        return jsonify({"error": "Server configuration error: DeepSeek model name not set"}), 500

    try:
        messages = [{"role": "user", "content": full_prompt}]
        logging.info(f"Sending request to DeepSeek API (Model: {DEEPSEEK_MODEL_NAME})...")

        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        logging.info("Received response object from DeepSeek API.")

        if not response.choices or len(response.choices) == 0:
             logging.error("No valid choices found in DeepSeek API response object.")
             return jsonify({"error": "No valid choices in DeepSeek API response"}), 500

        result_string = response.choices[0].message.content.strip()
        logging.info(f"Raw content string from DeepSeek API: ---START---\n{result_string}\n---END---")

        try:
            if result_string.startswith("```json"):
                result_string = result_string[7:]
            if result_string.endswith("```"):
                result_string = result_string[:-3]
            result_string = result_string.strip()

            quote_data = json.loads(result_string)
            logging.info(f"Successfully parsed JSON from DeepSeek: {quote_data}")

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error from DeepSeek: {e}. Raw string was: {result_string}", exc_info=True)
            return jsonify({
                "error": f"Invalid JSON response received from DeepSeek API. Parsing failed: {e}",
                "raw_response": result_string
                }), 502

        required_fields = ["english_text", "translated_text", "movie_title"]
        missing_fields = [field for field in required_fields if field not in quote_data]
        if missing_fields:
            logging.error(f"Missing required fields {missing_fields} in parsed DeepSeek JSON: {quote_data}")
            return jsonify({
                "error": f"Missing required fields in DeepSeek API response JSON: {', '.join(missing_fields)}",
                "received_data": quote_data
                }), 500

        movie_title_from_llm = quote_data["movie_title"]

        poster_url = None
        if TMDB_ACCESS_TOKEN and TMDB_CONFIG:
             poster_url = search_movie_poster(movie_title_from_llm)
        elif not TMDB_ACCESS_TOKEN:
            logging.info("TMDB Token not configured, skipping poster search.")
        else:
             logging.warning("TMDB config failed to load, skipping poster search.")
             if not fetch_tmdb_config():
                 logging.warning("Attempt to reload TMDB config also failed.")
             else:
                 poster_url = search_movie_poster(movie_title_from_llm)


        syllabled_text = "Syllable processing failed"
        try:
            spanish_text = quote_data["translated_text"]
            spanish_words = spanish_text.split()
            processed_syllables_list = []
            for word in spanish_words:
                cleaned_word = ''.join(c for c in word if c.isalpha() or c == '-')
                if not cleaned_word:
                    processed_syllables_list.append(word)
                    continue
                try:
                    found_syllables = syllables.find_syllables(cleaned_word)
                    if found_syllables:
                        processed_syllables_list.append("-".join(found_syllables))
                    else:
                        processed_syllables_list.append(word)
                except Exception as syl_err:
                     logging.warning(f"Syllable finding error for word '{word}' (cleaned: '{cleaned_word}'): {syl_err}")
                     processed_syllables_list.append(f"{word}[err]")

            syllabled_text = " ".join(processed_syllables_list)
            logging.info("Syllable processing completed.")

        except Exception as e:
            logging.error(f"Syllable processing error: {e}", exc_info=True)

        final_response = {
            "english_text": quote_data["english_text"],
            "translated_text": quote_data["translated_text"],
            "syllables": syllabled_text,
            "movie_title": movie_title_from_llm,
            "poster_url": poster_url
        }
        logging.info(f"Sending final response: {final_response}")
        return jsonify(final_response)

    except APIError as e:
        logging.error(f"DeepSeek API error: Status={e.status_code} Response={e.response}", exc_info=True)
        status_code = e.status_code or 500
        try:
            err_payload = e.response.json()
            message = err_payload.get("error", {}).get("message", str(e))
        except:
             message = e.body or str(e)

        if status_code == 401:
             message = f'DeepSeek API Authentication Failed. Check API Key ({DEEPSEEK_API_KEY_NAME}).'
        elif status_code == 404:
              message = 'DeepSeek API Endpoint or Model Not Found. Check Base URL and Model Name.'
        elif status_code == 429:
              message = 'DeepSeek API Rate Limit Exceeded.'
        elif status_code == 400:
             if "model_not_found" in str(message).lower():
                 message = f'DeepSeek Model "{DEEPSEEK_MODEL_NAME}" not found or invalid. Please check the model name.'
             else:
                 message = f"DeepSeek API Bad Request: {message}"

        return jsonify({"error": f"API error (DeepSeek): {message}"}), status_code

    except Exception as e:
        logging.error(f"Unexpected error in get_quote: {e}", exc_info=True)
        err_msg = str(e) if app.debug else "An unexpected server error occurred."
        return jsonify({"error": f"Unexpected server error: {err_msg}"}), 500

if __name__ == '__main__':
    if not DEEPSEEK_API_KEY_VALUE:
        print(f"\n*** WARNING: Environment variable '{DEEPSEEK_API_KEY_NAME}' not set. DeepSeek API calls will fail. ***\n")
    if DEEPSEEK_MODEL_NAME == "PLEASE_REPLACE_WITH_ACTUAL_DEEPSEEK_MODEL_NAME" or not DEEPSEEK_MODEL_NAME:
        print(f"\n*** WARNING: DEEPSEEK_MODEL_NAME placeholder not replaced or is empty in app.py (currently set to '{DEEPSEEK_MODEL_NAME}'). API calls will likely fail. Please choose a valid DeepSeek model. ***\n")
    if not TMDB_ACCESS_TOKEN:
         print(f"\n*** WARNING: Environment variable '{TMDB_API_KEY_NAME}' not set. Movie posters cannot be fetched. ***\n")
    elif not TMDB_CONFIG:
         print(f"\n*** WARNING: Failed to fetch TMDB configuration at startup. Movie posters might not be available. Check TMDB token and network. ***\n")

    print(f"Starting Flask development server...")
    print(f"Using DeepSeek Model: {DEEPSEEK_MODEL_NAME}")
    print(f"TMDB Poster Fetching: {'Enabled' if TMDB_ACCESS_TOKEN and TMDB_CONFIG else 'Disabled (check logs/config)'}")
    print(f"Connect to http://localhost:5000 or http://<your-ip>:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
