from flask import Flask, request, jsonify
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
import requests
import re
from urllib.parse import quote
import time
from urllib.parse import quote

app = Flask(__name__)

# ✅ Load NLP model
nlp_model = None

def get_nlp_model():
    global nlp_model
    if nlp_model is None:
        print("Loading NLP model...")
        from transformers import pipeline
        nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("Model loaded.")
    return nlp_model

# ✅ Download WordNet resources
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

CONCEPTNET_API_URL = "http://api.conceptnet.io/c/en/"

# ✅ Helpers
def detect_and_translate(text):
    try:
        lang = detect(text)
        print(f"Detected language for '{text}': {lang}")
        from nltk.corpus import wordnet
        # Dacă nu e engleză sau nu există în WordNet, traduce
        if lang != "en" and not wordnet.synsets(text):
            translated_text = GoogleTranslator(source=lang, target="en").translate(text)
            print(f"Translated '{text}' from {lang} to '{translated_text}'")
            return translated_text
        # Dacă nu există în WordNet, dar e engleză, returnează totuși cuvântul
        return text
    except Exception as e:
        print(f"Error in detect_and_translate for '{text}': {e}")
        return text
    
def clean_label(label):
    """
    Normalize ConceptNet labels:
      - lowercase, remove leading articles
      - replace underscores with spaces
      - strip parenthetical info, punctuation except spaces/hyphens
      - collapse whitespace
    """
    if not label:
        return ""
    lbl = label.strip().lower()
    for a in ("a ", "an ", "the "):
        if lbl.startswith(a):
            lbl = lbl[len(a):]
            break
    lbl = lbl.replace("_", " ")
    # remove content in parentheses
    lbl = re.sub(r"\s*\(.*?\)\s*", " ", lbl)
    # remove punctuation except hyphen
    lbl = re.sub(r"[^\w\s\-]", " ", lbl, flags=re.UNICODE)
    # collapse whitespace
    lbl = re.sub(r"\s+", " ", lbl).strip()
    return lbl


def get_wordnet_categories(word):
    synsets = wordnet.synsets(word)
    # Use the second part of lexname when available (e.g. 'noun.person' -> 'person')
    cats = []
    for syn in synsets:
        parts = syn.lexname().split(".")
        if len(parts) > 1:
            cats.append(parts[1])
        else:
            cats.append(parts[0])
    return list(set(cats)) if synsets else []


def get_conceptnet_data(word):
    """
    Query ConceptNet for `word` with retries and a WordNet fallback when ConceptNet is unavailable.
    Returns (categories_list, relationships_list).
    """
    try:
        resource = quote(word.strip().lower().replace(" ", "_"))
        # prefer the query endpoint (more reliable) and allow retries on 5xx
        attempts = 3
        backoff = 0.6
        data = None
        for i in range(attempts):
            try:
                url = f"{CONCEPTNET_API_URL}/query?start=/c/en/{resource}&limit=50"
                resp = requests.get(url, timeout=6)
                # Retry on server errors
                if resp.status_code >= 500:
                    time.sleep(backoff * (2 ** i))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                print(f"ConceptNet request attempt {i+1} failed for '{word}': {e}")
                time.sleep(backoff * (2 ** i))
        if not data:
            # fallback to simple WordNet-based related tokens if ConceptNet is down
            print(f"ConceptNet unavailable for '{word}', falling back to WordNet heuristics")
            wn_related = []
            try:
                synsets = wordnet.synsets(word)
                for s in synsets[:4]:
                    for hy in s.hyponyms()[:6]:
                        for l in hy.lemmas()[:4]:
                            wn_related.append(l.name().replace("_", " "))
            except Exception as e:
                print(f"WordNet fallback failed for '{word}': {e}")
            return (sorted(set(wn_related)), [])
        # build categories and relationships from edges
        edges = data.get("edges", [])[:200]
        categories = set()
        relationships = []
        for edge in edges:
            rel_label = edge.get("rel", {}).get("label", "")
            start_label = edge.get("start", {}).get("label", "")
            end_label = edge.get("end", {}).get("label", "")
            for target_raw in (start_label, end_label):
                if not target_raw:
                    continue
                target = target_raw.replace("_", " ").strip().lower()
                if target == word.lower():
                    continue
                # accept targets that contain letters
                if not any(c.isalpha() for c in target):
                    continue
                categories.add(target)
                relationships.append({
                    "source": word,
                    "target": target,
                    "relation": rel_label or "related"
                })
        return (sorted(categories), relationships)
    except Exception as e:
        print(f"Unexpected error in get_conceptnet_data('{word}'): {e}")
        return ([], [])
        
def generate_economic_tags(words):
    tags = []
    for word in words:
        if any(w in word for w in ["market", "finance", "bank", "trade"]):
            tags.append("Finance")
        elif any(w in word for w in ["medicine", "health", "bio"]):
            tags.append("Healthcare")
        elif any(w in word for w in ["code", "tech", "data", "ai"]):
            tags.append("Technology")
        elif any(w in word for w in ["eco", "farm", "sustain"]):
            tags.append("Environment")
    return list(set(tags))

def generate_trendy_topics(words):
    trends = []
    if any("ai" in w or "machine" in w for w in words):
        trends.append("Artificial Intelligence")
    if any("climate" in w or "green" in w for w in words):
        trends.append("Climate Action")
    if any("crypto" in w or "block" in w for w in words):
        trends.append("Cryptocurrency")
    if any("remote" in w or "hybrid" in w for w in words):
        trends.append("Future of Work")
    return trends

# --- NEW: broader profession detection helpers ---
def fetch_wikidata_professions(term):
    """
    Use Wikidata search API to find entities whose description suggests
    an occupation/profession related to `term`.
    """
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": term,
            "type": "item",
            "limit": 12
        }
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        profs = set()
        for entry in data.get("search", []):
            label = entry.get("label")
            desc = (entry.get("description") or "").lower()
            if not label:
                continue
            # If description contains profession/occupation hints, accept label
            if any(k in desc for k in ["occupation", "profession", "job", "worker", "specialist", "role", "career"]):
                profs.add(label)
        return list(profs)
    except Exception as e:
        print(f"Wikidata lookup failed for '{term}': {e}")
        return []

def extract_professions_from_wordnet(word):
    """
    Explore WordNet synsets/hyponyms/lemmas to find tokens that look like professions
    (heuristic: common profession suffixes or noun.person lexnames).
    """
    profs = set()
    try:
        synsets = wordnet.synsets(word)
        # Also consider synsets for 'person' related lexnames
        for syn in synsets:
            # check if this synset or its hypernyms/hyponyms contain profession-like lemmas
            candidates = [syn] + syn.hyponyms() + syn.hypernyms()
            for c in candidates:
                # prefer lemmas that are nouns and look like role names
                for l in c.lemmas():
                    name = l.name().replace("_", " ").lower()
                    if len(name) < 3:
                        continue
                    # simple suffix heuristics for professions
                    if any(name.endswith(s) for s in ("er", "ist", "ian", "or", "ant", "ent", "man", "woman", "maker")):
                        profs.add(name)
                # also inspect lexname (e.g., 'noun.person') to surface person types
                try:
                    lex = c.lexname()
                    if "person" in lex:
                        for l in c.lemmas():
                            profs.add(l.name().replace("_", " ").lower())
                except Exception:
                    pass
    except Exception as e:
        print(f"WordNet profession extraction failed for '{word}': {e}")
    return [p.title() for p in profs]


# ✅ Main Route
@app.route("/process", methods=["POST"])
def process_words():
    try:
        data = request.get_json()
        if not data or "words" not in data:
            return jsonify({"error": "No words provided"}), 400

        input_words = [word.strip().lower() for word in data.get("words", "").split(",")]
        nodes, links = [], []
        existing_nodes = set()
        all_categories = set()
        processed_words = set()  # Track processed words to avoid infinite loops
        max_depth = 1  # Limit graph expansion to 1 level

        def process_word(word, depth=0):
            if word in processed_words or depth > max_depth:
                return
            processed_words.add(word)

            translated_word = detect_and_translate(word).lower()

            wordnet_cats = get_wordnet_categories(translated_word)
            conceptnet_cats, conceptnet_links = get_conceptnet_data(translated_word)

            # Debugging: Log the categories and links
            print(f"Processing word: {word}, Translated: {translated_word}")
            print(f"WordNet categories: {wordnet_cats}")
            print(f"ConceptNet categories: {conceptnet_cats}")
            print(f"ConceptNet links: {conceptnet_links}")

           
            combined_cats = list(set(wordnet_cats + conceptnet_cats))
            combined_cats = [c for c in combined_cats if c and c.lower() != translated_word.lower()]

            # Add found categories to the global set for broader career/economy detection
            for c in combined_cats:
                if isinstance(c, str) and c.strip():
                    all_categories.add(c.lower())


            if translated_word not in existing_nodes:
                nodes.append({
                    "id": translated_word,
                    "original": word,
                    "categories": combined_cats
                })
                existing_nodes.add(translated_word)

            # Add ConceptNet links
            for link in conceptnet_links:
                if link["target"] not in processed_words:
                    links.append(link)

            # Recursively process linked words (limited by depth)
            for link in conceptnet_links:
                process_word(link["target"], depth + 1)

        # Process each input word
        for original_word in input_words:
            process_word(original_word)

        # Add linked nodes if not yet in node list
        linked_words = {l["source"] for l in links} | {l["target"] for l in links}
        for lw in linked_words:
            if lw not in existing_nodes:
                nodes.append({"id": lw, "categories": ["auto-generated"]})
                existing_nodes.add(lw)

        # Remove duplicate links and self-loops
        links = [l for l in links if l["source"] != l["target"]]
        unique_links = {(l["source"], l["target"], l["relation"]) for l in links}
        links = [{"source": s, "target": t, "relation": r} for s, t, r in unique_links]

        # Generate panel data
        user_input_words = [n["id"] for n in nodes if "auto-generated" not in n.get("categories", [])]
        economy_tags = generate_economic_tags(user_input_words)
        trendy_tags = generate_trendy_topics(user_input_words)
        # Broader career detection: match category words and node ids against career keywords
        career_keywords = [
            "person","profession","occupation","job","role","worker","specialist",
            "engineer","doctor","teacher","lawyer","nurse","artist","scientist",
            "manager","developer","programmer","chef","driver","farmer","mechanic",
            "designer","sales","consultant","technician","producer","writer","musician"
        ]

        career_tags_set = set()
        for cat in all_categories:
            if any(kw in cat for kw in career_keywords):
                career_tags_set.add(cat)

        # also check node ids (some nodes may be direct profession names)
        for n in nodes:
            nid = n.get("id", "").lower()
            for kw in career_keywords:
                if kw in nid:
                    career_tags_set.add(nid)
                    break

        career_tags = sorted(career_tags_set)
        
        # Filtrare noduri neconectate
        #connected_nodes = {link["source"] for link in links} | {link["target"] for link in links}
        #nodes = [node for node in nodes if node["id"] in connected_nodes]

        response = {
            "nodes": [{"id": n["id"], "categories": n["categories"]} for n in nodes],
            "links": links,
            "words": input_words,  # Original input words
            "careers": career_tags,  # Career-related categories
            "economy": economy_tags,  # Economic tags
            "trends": trendy_tags  # Trendy topics
        }

        import json
        print("Flask response:", json.dumps(response, indent=2))

        return jsonify(response)

    except Exception as e:
        print(f"Error in /process: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
