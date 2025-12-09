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


def get_family_vocab(word):
    """
    Return related family vocabulary for `word` (things that come from it, parts, products).
    Uses ConceptNet edges filtered for relations like HasA, PartOf, UsedFor, DerivedFrom,
    and WordNet hyponyms/meronyms/derivational forms as a fallback.
    Returns (family_list, relationship_links).
    """
    family = set()
    relationships = []
    w = (word or "").strip().lower()
    if not w:
        return ([], [])

    try:
        # ConceptNet query for more specific relations
        resource = quote(w.replace(" ", "_"))
        url = f"{CONCEPTNET_API_URL}/query?start=/c/en/{resource}&limit=200"
        try:
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            data = resp.json()
            edges = data.get("edges", [])[:200]
            for edge in edges:
                rel_raw = edge.get("rel", {}).get("label", "")
                rel_l = (rel_raw or "").lower()
                start_label = edge.get("start", {}).get("label", "")
                end_label = edge.get("end", {}).get("label", "")

                # Candidate targets are start/end labels not equal to the word
                for target_raw, other_raw in ((start_label, end_label), (end_label, start_label)):
                    if not target_raw or not other_raw:
                        continue
                    target = clean_label(target_raw)
                    other = clean_label(other_raw)
                    if not target or target == w:
                        continue
                    if not any(c.isalpha() for c in target):
                        continue

                    # Accept relations indicative of product/part/derivative or usage
                    rel_tokens = ["hasa", "has a", "has", "part", "made", "made of", "used", "used for", "cause", "derived", "derived from", "isa", "instance", "form", "produce", "product", "produce", "substance", "meronym", "meronymous", "material"]
                    if any(tok in rel_l for tok in rel_tokens) or any(tok in rel_l for tok in ["has", "part", "made", "used"]):
                        family.add(other)
                        relationships.append({"source": w, "target": other, "relation": rel_raw or "related"})
        except Exception:
            # ConceptNet may fail — continue to WordNet fallbacks
            pass

        # WordNet-based fallbacks: hyponyms (specific kinds), meronyms (parts/substances), derivational forms
        try:
            synsets = wordnet.synsets(w)
            for s in synsets[:6]:
                # Hyponyms (kinds of the word)
                for hy in s.hyponyms()[:10]:
                    for l in hy.lemmas()[:6]:
                        t = l.name().replace("_", " ").lower()
                        if t and t != w:
                            family.add(t)
                            relationships.append({"source": w, "target": t, "relation": "is a"})

                # Part meronyms and substance meronyms (things made of / parts)
                for pm in list(s.part_meronyms())[:10] + list(s.substance_meronyms())[:10]:
                    for l in pm.lemmas()[:8]:
                        t = l.name().replace("_", " ").lower()
                        if t and t != w:
                            family.add(t)
                            relationships.append({"source": w, "target": t, "relation": "part of"})

                # Derivationally related forms (e.g., milk -> milky? or derive related words)
                for l in s.lemmas()[:8]:
                    try:
                        for dr in l.derivationally_related_forms()[:8]:
                            t = dr.name().replace("_", " ").lower()
                            if t and t != w:
                                family.add(t)
                                relationships.append({"source": w, "target": t, "relation": "derivative"})
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as e:
        print(f"get_family_vocab failed for '{word}': {e}")

    return (sorted(family), relationships)
        
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
    Extract profession/occupation suggestions from multiple sources:
    1. WordNet lemmas from noun.person synsets
    2. Related category words that suggest professions
    Returns list of unique professional terms
    """
    profs = set()
    
    profession_suffixes = ("er", "ist", "ian", "or", "ant", "ent", "man", "woman", 
                          "maker", "smith", "wright", "monger", "keeper", "guard", "master")
    profession_keywords = ("professional", "specialist", "expert", "practitioner", "technician",
                          "officer", "agent", "conductor", "performer", "consultant", "worker")
    
    # Blacklist of non-professions
    blacklist = {"person", "people", "human", "entity", "plant", "cross-dresser", "cow", 
                "appointment", "substance", "object", "thing", "item", "artifact",
                "masturbator", "onanist", "violator", "perpetrator", "deviant", "abuser",
                "heretic", "iconoclast", "miscreant", "traitor", "charlatan", "pretender",
                "idler", "loafer", "dunce", "blockhead", "fool", "simpleton"}
    
    word_clean = word.lower().strip()
    if not word_clean or len(word_clean) < 2:
        return []
    
    try:
        synsets = wordnet.synsets(word_clean)
        
        for syn in synsets:
            try:
                # Check if this is a person-type synset
                lex = syn.lexname()
                if "noun.person" in lex:
                    # Extract lemmas from this synset - more inclusive filtering
                    for l in syn.lemmas():
                        name = l.name().replace("_", " ").strip().lower()
                        if len(name) < 2 or name in blacklist:
                            continue
                        
                        # Accept if has profession suffix OR has profession keyword
                        has_suffix = any(name.endswith(suffix) for suffix in profession_suffixes)
                        has_keyword = any(kw in name for kw in profession_keywords)
                        
                        if has_suffix or has_keyword:
                            profs.add(name.title())
            except:
                pass
                
            # Also check hyponyms (more specific person types)
            try:
                for hypo in syn.hyponyms()[:5]:
                    if "noun.person" in hypo.lexname():
                        for l in hypo.lemmas():
                            name = l.name().replace("_", " ").strip().lower()
                            if len(name) < 2 or name in blacklist:
                                continue
                            
                            has_suffix = any(name.endswith(suffix) for suffix in profession_suffixes)
                            has_keyword = any(kw in name for kw in profession_keywords)
                            
                            if has_suffix or has_keyword:
                                profs.add(name.title())
            except:
                pass
    except Exception as e:
        pass
    
    # Expand with category-based professions
    category_profession_map = {
        "food": ["Chef", "Cook", "Baker", "Nutritionist", "Food Scientist"],
        "plant": ["Botanist", "Gardener", "Florist", "Horticulturist"],
        "animal": ["Zookeeper", "Veterinarian", "Animal Handler"],
        "art": ["Artist", "Painter", "Sculptor", "Designer"],
        "music": ["Musician", "Composer", "Conductor"],
        "sport": ["Athlete", "Coach", "Trainer"],
        "science": ["Scientist", "Researcher", "Technician"],
        "medicine": ["Doctor", "Physician", "Nurse"],
        "law": ["Lawyer", "Judge", "Attorney"],
        "business": ["Entrepreneur", "Manager", "Executive"],
        "education": ["Teacher", "Professor", "Educator"],
        "technology": ["Programmer", "Engineer", "Developer"],
    }
    
    if word_clean in category_profession_map:
        profs.update(category_profession_map[word_clean])
    
    # Final validation: ensure each candidate has at least one noun synset
    # that is person-related (to avoid demonyms, adjectives, or unrelated nouns)
    valid_profs = []
    try:
        for p in profs:
            p_clean = p.lower().strip()
            good = False
            try:
                for syn in wordnet.synsets(p_clean, pos='n'):
                    if 'person' in syn.lexname():
                        good = True
                        break
            except:
                pass
            if good:
                valid_profs.append(p)
    except:
        # If validation fails for any reason, fallback to original list
        valid_profs = list(profs)

    return sorted(valid_profs)[:10]


# ✅ Main Route
@app.route("/process", methods=["POST"])
def process_words():
    try:
        data = request.get_json()
        if not data or "words" not in data:
            return jsonify({"error": "No words provided"}), 400

        # API toggle: whether to produce inferred links server-side (default: true)
        inferred_enabled = bool(data.get('inferred', True))

        input_words = [word.strip().lower() for word in data.get("words", "").split(",")]
        nodes, links = [], []
        existing_nodes = set()
        all_categories = set()
        processed_words = set()  # Track processed words to avoid infinite loops
        max_depth = 1  # Limit graph expansion to 1 level

        def process_word(word, depth=0):
            """Process input word: add node and get direct relationships only (no expansion)"""
            if word in processed_words:
                return
            processed_words.add(word)

            translated_word = detect_and_translate(word).lower()
            
            # Get metadata for this word (categories for career/economy panels)
            wordnet_cats = get_wordnet_categories(translated_word)
            conceptnet_cats, _ = get_conceptnet_data(translated_word)  # Don't add links from conceptnet
            
            combined_cats = list(set(wordnet_cats + conceptnet_cats))
            combined_cats = [c for c in combined_cats if c and c.lower() != translated_word.lower()]
            generic_exclude = {"person", "people", "human", "entity", "thing", "object"}
            combined_cats = [c for c in combined_cats if c.lower() not in generic_exclude]
            
            # Add to global categories
            for c in combined_cats:
                if isinstance(c, str) and c.strip():
                    all_categories.add(c.lower())

            # Add only the input word itself as a node
            if translated_word not in existing_nodes:
                nodes.append({
                    "id": translated_word,
                    "original": word,
                    "categories": combined_cats[:3]  # Limit to 3 categories
                })
                existing_nodes.add(translated_word)

        # Process each input word
        for original_word in input_words:
            process_word(original_word)

        # Only server-side inference adds links between input words

        # --- Server-side inference pass (only if enabled) ---
        if inferred_enabled:
            try:
                # normalize node id lookup
                node_ids = {n['id'].lower(): n for n in nodes}
                # helper to add a link uniquely into links list
                seen_triples = {(l['source'], l['target'], l.get('relation', '')) for l in links}
                def add_link_once(s, t, rel):
                    if not s or not t:
                        return
                    sL = s.strip().lower(); tL = t.strip().lower()
                    if sL == tL:
                        return
                    key = (sL, tL, rel or '')
                    if key in seen_triples:
                        return
                    seen_triples.add(key)
                    links.append({'source': sL, 'target': tL, 'relation': rel or 'inferred'})

                # classification heuristics: simple keyword lists + WordNet hints
                animal_kw = set(['animal','mammal','cow','pig','chicken','sheep','goat','horse','bovine','cattle','swine','hen','dog','cat','fish'])
                plant_kw = set(['plant','flora','tree','grass','weed','herb','crop','leaf','fodder','hay','grassland','algae'])
                product_kw = set(['product','food','dairy','meat','milk','cheese','wool','egg','honey','leather','beef','pork','butter'])

                animals = set(); plants = set(); products = set()
                for n in nodes:
                    nid = (n.get('id') or '').strip().lower()
                    cats = [c.lower() for c in n.get('categories', []) if isinstance(c, str)]
                    text = nid + ' ' + ' '.join(cats)
                    # keyword match
                    if any(k in text for k in animal_kw): animals.add(nid)
                    if any(k in text for k in plant_kw): plants.add(nid)
                    if any(k in text for k in product_kw): products.add(nid)
                    # WordNet hint: check lexname categories
                    try:
                        wcats = get_wordnet_categories(nid)
                        for wc in wcats:
                            if wc and 'animal' in wc: animals.add(nid)
                            if wc and 'plant' in wc: plants.add(nid)
                            if wc and ('food' in wc or 'artifact' in wc or 'substance' in wc): products.add(nid)
                    except Exception:
                        pass

                # Add inferred plant->animal and animal->product links
                for pl in plants:
                    for an in animals:
                        add_link_once(pl, an, 'eaten_by')
                for an in animals:
                    for p in products:
                        add_link_once(an, p, 'produces')

                # transitive plant->product via animal
                for pl in plants:
                    for p in products:
                        # check if exists some animal linking pl->animal and animal->p
                        has_middle = False
                        for an in animals:
                            if any((l['source'].lower() == pl and l['target'].lower() == an) for l in links) and any((l['source'].lower() == an and l['target'].lower() == p) for l in links):
                                has_middle = True; break
                        if has_middle:
                            add_link_once(pl, p, 'contributes')

                # ensure inferred links produce nodes if missing
                for l in list(links):
                    for side in ('source','target'):
                        sid = (l.get(side) or '').strip().lower()
                        if sid and sid not in existing_nodes and sid not in {"person","people","human","entity","thing","object"}:
                            nodes.append({'id': sid, 'categories': ['auto-generated']})
                            existing_nodes.add(sid)
            except Exception as e:
                print('Server-side inference error:', e)

        # Remove duplicate links and self-loops
        links = [l for l in links if l["source"] != l["target"]]
        unique_links = {(l["source"], l["target"], l["relation"]) for l in links}
        links = [{"source": s, "target": t, "relation": r} for s, t, r in unique_links]

        # Generate panel data
        user_input_words = [n["id"] for n in nodes if "auto-generated" not in n.get("categories", [])]
        economy_tags = generate_economic_tags(user_input_words)
        trendy_tags = generate_trendy_topics(user_input_words)
        
        # Helper: Vet profession candidate using WordNet definitions and Wikidata
        def is_vetted_profession(candidate):
            """Return True if `candidate` looks like a profession.
            Checks:
            - WordNet noun synsets exist and any synset definition contains occupation hints
            - OR Wikidata search returns description containing occupation hints
            """
            try:
                cand = (candidate or "").strip().lower()
                if not cand:
                    return False

                # Check WordNet synsets for noun.person-like definitions
                try:
                    syns = wordnet.synsets(cand, pos='n')
                    for s in syns:
                        lex = s.lexname().lower()
                        defn = (s.definition() or "").lower()
                        if 'person' in lex and any(k in defn for k in ["occupation", "profession", "one who", "person who", "works as", "job", "practitioner", "specialist"]):
                            return True
                except Exception:
                    pass

                # Fallback: query Wikidata for this label and see if description suggests an occupation
                try:
                    wd = fetch_wikidata_professions(candidate)
                    if wd:
                        return True
                except Exception:
                    pass

            except Exception:
                return False
            return False

        # Career detection: strict whitelist mode — only include curated category professions
        career_set = []
        category_profession_map = {
            "food": ["Chef", "Cook", "Baker", "Nutritionist", "Food Scientist"],
            "plant": ["Botanist", "Gardener", "Florist", "Horticulturist"],
            "animal": ["Zookeeper", "Veterinarian", "Animal Handler"],
            "art": ["Artist", "Painter", "Sculptor", "Designer"],
            "music": ["Musician", "Composer", "Conductor"],
            "sport": ["Athlete", "Coach", "Trainer"],
            "science": ["Scientist", "Researcher", "Technician"],
            "medicine": ["Doctor", "Physician", "Nurse"],
            "law": ["Lawyer", "Judge", "Attorney"],
            "business": ["Entrepreneur", "Manager", "Executive"],
            "education": ["Teacher", "Professor", "Educator"],
            "technology": ["Programmer", "Engineer", "Developer"],
        }

        added = set()
        for cat in sorted(list(all_categories)):
            cat_l = (cat or "").strip().lower()
            if cat_l in category_profession_map:
                for p in category_profession_map[cat_l]:
                    if p.lower() not in added:
                        career_set.append(p)
                        added.add(p.lower())

        # If we have no category-based professions, fall back to a conservative default list
        if not career_set:
            fallback = ["Consultant", "Specialist", "Technician", "Operator"]
            career_set.extend(fallback)

        career_tags = career_set[:15]

        # Final safety filter: remove any generic/human tokens that slipped through
        generic_exclude_ids = {"person", "people", "human", "entity", "thing", "object"}
        links = [l for l in links if (l.get("source") or "").lower() not in generic_exclude_ids and (l.get("target") or "").lower() not in generic_exclude_ids]
        nodes = [n for n in nodes if (n.get("id") or "").lower() not in generic_exclude_ids]

        response = {
            "nodes": [{"id": n["id"], "categories": n.get("categories", [])} for n in nodes],
            "links": links,
            "words": input_words,
            "careers": career_tags,
            "economy": economy_tags,
            "trends": trendy_tags
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
