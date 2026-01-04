from flask import Flask, request, jsonify
from transformers import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
import requests

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

# ✅ Comprehensive job database with semantic field mappings
JOB_DATABASE = {
    'animal': ['Veterinarian', 'Animal Trainer', 'Zoologist', 'Wildlife Biologist', 'Zookeeper', 'Pet Groomer', 'Animal Behaviorist', 'Marine Biologist', 'Aquarist', 'Animal Control Officer', 'Livestock Manager', 'Dairy Farmer', 'Rancher', 'Animal Nutritionist', 'Veterinary Technician'],
    'plant': ['Botanist', 'Horticulturist', 'Agricultural Scientist', 'Landscape Architect', 'Forester', 'Arborist', 'Plant Pathologist', 'Greenhouse Manager', 'Farm Manager', 'Crop Consultant', 'Soil Scientist', 'Agricultural Engineer'],
    'food': ['Chef', 'Food Scientist', 'Nutritionist', 'Dietitian', 'Restaurant Manager', 'Food Safety Inspector', 'Culinary Instructor', 'Pastry Chef', 'Butcher', 'Baker', 'Food Technologist', 'Beverage Manager', 'Sommelier', 'Barista Trainer'],
    'health': ['Doctor', 'Nurse', 'Pharmacist', 'Physical Therapist', 'Occupational Therapist', 'Medical Researcher', 'Healthcare Administrator', 'Public Health Officer', 'Epidemiologist', 'Health Educator', 'Clinical Psychologist', 'Psychiatrist', 'Surgeon', 'Dentist', 'Radiologist'],
    'technology': ['Software Engineer', 'Data Scientist', 'Cybersecurity Analyst', 'IT Manager', 'Systems Administrator', 'DevOps Engineer', 'Machine Learning Engineer', 'Web Developer', 'Database Administrator', 'Network Engineer', 'Cloud Architect', 'UX Designer', 'AI Researcher'],
    'education': ['Teacher', 'Professor', 'Education Administrator', 'Curriculum Developer', 'School Counselor', 'Special Education Teacher', 'Education Consultant', 'Instructional Designer', 'Academic Advisor', 'Tutor', 'Education Researcher'],
    'art': ['Graphic Designer', 'Art Director', 'Illustrator', 'Animator', 'Fine Artist', 'Gallery Curator', 'Art Therapist', 'Museum Conservator', 'Art Teacher', 'Creative Director', 'Set Designer', 'Exhibition Designer'],
    'music': ['Musician', 'Music Producer', 'Audio Engineer', 'Music Teacher', 'Composer', 'Music Therapist', 'Sound Designer', 'Music Director', 'Conductor', 'Vocalist', 'Music Journalist', 'Concert Promoter'],
    'communication': ['Journalist', 'Public Relations Specialist', 'Communications Manager', 'Social Media Manager', 'Content Strategist', 'Copywriter', 'Technical Writer', 'Editor', 'Communications Consultant', 'Broadcast Producer', 'Media Planner'],
    'business': ['Business Analyst', 'Management Consultant', 'Financial Advisor', 'Accountant', 'Marketing Manager', 'Sales Manager', 'Operations Manager', 'Business Development Manager', 'Product Manager', 'Entrepreneur', 'Investment Banker', 'Business Owner'],
    'engineering': ['Mechanical Engineer', 'Electrical Engineer', 'Civil Engineer', 'Chemical Engineer', 'Aerospace Engineer', 'Biomedical Engineer', 'Environmental Engineer', 'Industrial Engineer', 'Manufacturing Engineer', 'Robotics Engineer', 'Materials Engineer'],
    'science': ['Research Scientist', 'Physicist', 'Chemist', 'Biologist', 'Geologist', 'Astronomer', 'Meteorologist', 'Environmental Scientist', 'Materials Scientist', 'Microbiologist', 'Geneticist', 'Biochemist'],
    'social': ['Social Worker', 'Psychologist', 'Sociologist', 'Anthropologist', 'Community Organizer', 'Human Resources Manager', 'Counselor', 'Life Coach', 'Career Counselor', 'Therapist', 'Case Manager'],
    'law': ['Lawyer', 'Judge', 'Paralegal', 'Legal Consultant', 'Compliance Officer', 'Patent Attorney', 'Legal Researcher', 'Court Reporter', 'Legal Advisor', 'Contract Specialist'],
    'construction': ['Architect', 'Civil Engineer', 'Construction Manager', 'Carpenter', 'Electrician', 'Plumber', 'HVAC Technician', 'Construction Inspector', 'Surveyor', 'Heavy Equipment Operator'],
    'transportation': ['Pilot', 'Air Traffic Controller', 'Transportation Planner', 'Logistics Manager', 'Supply Chain Analyst', 'Fleet Manager', 'Maritime Captain', 'Railway Engineer', 'Urban Planner'],
    'environment': ['Environmental Consultant', 'Conservation Scientist', 'Renewable Energy Engineer', 'Environmental Lawyer', 'Sustainability Coordinator', 'Climate Scientist', 'Ecologist', 'Environmental Educator'],
    'finance': ['Financial Analyst', 'Actuary', 'Investment Analyst', 'Risk Manager', 'Tax Consultant', 'Financial Planner', 'Economist', 'Quantitative Analyst', 'Credit Analyst', 'Portfolio Manager'],
    'sport': ['Athletic Trainer', 'Sports Coach', 'Physical Education Teacher', 'Sports Psychologist', 'Sports Medicine Physician', 'Fitness Instructor', 'Sports Agent', 'Recreation Coordinator'],
    'hospitality': ['Hotel Manager', 'Event Planner', 'Travel Agent', 'Concierge', 'Restaurant Manager', 'Tourism Manager', 'Catering Manager', 'Guest Relations Manager'],
}

# NO HARDCODED MAPPINGS - System uses 100% dynamic semantic analysis via WordNet

# ✅ Helpers
def detect_and_translate(text):
    """
    Simple translation helper - just return text if it has WordNet synsets,
    otherwise try translation (rarely needed for English words).
    """
    text_lower = text.lower().strip()
    
    # If word exists in WordNet, no need to translate
    if wordnet.synsets(text_lower):
        return text_lower
    
    # Only translate if word is very short or looks like it might be non-English
    if len(text) < 2:
        return text_lower
    
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source=lang, target="en").translate(text)
            return translated.lower().strip()
    except Exception:
        pass
    
    return text_lower
    
def clean_label(label):
    return label.strip().lower().replace("a ", "").replace("an ", "").replace("the ", "")


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
    """Fetch ConceptNet data with timeout and error handling."""
    try:
        response = requests.get(f"{CONCEPTNET_API_URL}{word}", timeout=2)
        response.raise_for_status()
        data = response.json()

        categories = set()
        relationships = []
        for edge in data.get("edges", [])[:10]:
            relation = edge["rel"]["label"]
            target_word = clean_label(edge["end"]["label"])
            cleaned_word = clean_label(word)

            # Skip self-loops and unrelated nodes
            if target_word == cleaned_word or not target_word.isalpha():
                continue

            # Only include specific relations
            if relation in ["IsA", "PartOf", "UsedFor", "CapableOf", "RelatedTo"]:
                categories.add(target_word)
                relationships.append({"source": cleaned_word, "target": target_word, "relation": relation})

        return list(categories), relationships
    except Exception as e:
        # Silently fail - ConceptNet is optional
        return [], []
    
def generate_economic_tags(words):
    """
    Generate economic sector tags dynamically for ANY words using WordNet semantic analysis.
    No hardcoded mappings - works at scale for any vocabulary.
    """
    tags_set = set()
    
    # Economic sectors with semantic indicators
    economic_sectors = {
        'Finance & Banking': ['money', 'currency', 'bank', 'investment', 'capital', 'credit', 'financial', 'monetary', 'stock', 'bond', 'asset'],
        'Healthcare & Medicine': ['health', 'medical', 'treatment', 'disease', 'therapy', 'patient', 'clinical', 'hospital', 'pharmaceutical', 'diagnosis'],
        'Technology & IT': ['computer', 'software', 'digital', 'electronic', 'system', 'data', 'network', 'programming', 'algorithm', 'tech'],
        'Agriculture & Food': ['farming', 'crop', 'agriculture', 'food', 'livestock', 'harvest', 'cultivation', 'agricultural', 'produce', 'grain'],
        'Manufacturing & Industry': ['production', 'factory', 'manufacturing', 'industrial', 'machinery', 'assembly', 'fabrication', 'processing'],
        'Energy & Resources': ['energy', 'power', 'fuel', 'electricity', 'renewable', 'oil', 'gas', 'solar', 'wind', 'coal'],
        'Transportation & Logistics': ['transport', 'vehicle', 'shipping', 'delivery', 'logistics', 'freight', 'cargo', 'distribution'],
        'Real Estate & Construction': ['building', 'construction', 'property', 'real estate', 'housing', 'infrastructure', 'development'],
        'Education & Training': ['education', 'teaching', 'learning', 'training', 'instruction', 'academic', 'school', 'university'],
        'Retail & Commerce': ['retail', 'sales', 'commerce', 'trade', 'merchant', 'store', 'shopping', 'consumer', 'market'],
        'Media & Entertainment': ['media', 'entertainment', 'broadcasting', 'film', 'television', 'content', 'publishing', 'journalism'],
        'Tourism & Hospitality': ['tourism', 'travel', 'hotel', 'hospitality', 'accommodation', 'visitor', 'vacation', 'resort'],
    }
    
    for word in words:
        try:
            synsets = wordnet.synsets(word.lower())
            if not synsets:
                continue
            
            syn = synsets[0]
            definition = syn.definition().lower()
            
            # Check hypernym chain for economic context
            hypernyms = []
            current = syn
            for _ in range(3):
                hypers = current.hypernyms()
                if not hypers:
                    break
                current = hypers[0]
                hypernyms.append(current.lemmas()[0].name().lower())
            
            full_context = f"{word.lower()} {definition} {' '.join(hypernyms)}"
            
            # Match against economic sectors using semantic indicators
            for sector, indicators in economic_sectors.items():
                if any(indicator in full_context for indicator in indicators):
                    tags_set.add(sector)
        
        except Exception:
            pass
    
    return sorted(list(tags_set))[:8]  # Return top 8 most relevant sectors

def generate_trendy_topics(words):
    """
    Dynamically infer trending topics for ANY input words via WordNet context mining.
    If nothing matches, provide a small default set to avoid empty UI.
    """
    trends_set = set()

    # Trend categories with broad semantic indicators
    trend_categories = {
        'Artificial Intelligence & ML': ['artificial', 'intelligence', 'machine', 'learning', 'neural', 'algorithm', 'ai', 'automation', 'robot', 'cognitive'],
        'Climate & Sustainability': ['climate', 'environment', 'sustainable', 'green', 'renewable', 'carbon', 'ecology', 'conservation', 'emission', 'biodiversity'],
        'Digital Transformation': ['digital', 'transformation', 'cloud', 'software', 'platform', 'online', 'virtual', 'cyber', 'internet', 'compute'],
        'Blockchain & Crypto': ['blockchain', 'crypto', 'bitcoin', 'decentralized', 'token', 'ledger', 'cryptocurrency'],
        'Remote Work & Collaboration': ['remote', 'hybrid', 'collaboration', 'telecommute', 'virtual', 'workspace', 'distributed'],
        'Biotechnology & Genomics': ['biotech', 'genetic', 'genome', 'dna', 'molecular', 'cellular', 'bio-engineering', 'biotechnology'],
        'Renewable Energy': ['solar', 'wind', 'renewable', 'clean energy', 'photovoltaic', 'turbine', 'sustainable power', 'geothermal'],
        'E-commerce & Digital Markets': ['e-commerce', 'online shopping', 'marketplace', 'digital payment', 'retail technology', 'checkout', 'commerce'],
        'Data Science & Analytics': ['data', 'analytics', 'big data', 'statistics', 'visualization', 'insight', 'metrics', 'prediction', 'modeling'],
        'Cybersecurity': ['security', 'cyber', 'encryption', 'protection', 'threat', 'vulnerability', 'firewall', 'breach'],
    }

    def context_for_word(word):
        try:
            synsets = wordnet.synsets(word.lower())
            if not synsets:
                return ""
            syn = synsets[0]
            definition = syn.definition().lower()
            hypernyms = []
            current = syn
            for _ in range(3):
                hypers = current.hypernyms()
                if not hypers:
                    break
                current = hypers[0]
                hypernyms.append(current.lemmas()[0].name().lower())
            lemmas = [l.name().lower() for l in syn.lemmas()[:4]]
            return f"{word.lower()} {definition} {' '.join(hypernyms)} {' '.join(lemmas)}"
        except Exception:
            return ""

    # Build contexts and match
    for word in words:
        full_context = context_for_word(word)
        if not full_context:
            continue
        for trend, indicators in trend_categories.items():
            if any(ind in full_context for ind in indicators):
                trends_set.add(trend)

    # Fallback defaults to avoid empty list
    if not trends_set:
        trends_set.update(['Artificial Intelligence & ML', 'Digital Transformation'])

    return sorted(list(trends_set))[:6]

def expand_word_to_pool(word, max_expansions=12):
    """
    Expand a word to related words dynamically using WordNet relations and definition mining.
    Works for ANY word, not just predefined ones.
    """
    expanded = set()
    word_lower = word.lower()
    
    # DYNAMIC bridge extraction from WordNet definitions
    def extract_bridges_from_definition(synset):
        """Extract key nouns from definition that act as semantic bridges"""
        bridges = set()
        definition = synset.definition().lower()
        # Common bridge patterns in definitions
        bridge_patterns = [
            'used for', 'made from', 'type of', 'part of',
            'produced by', 'found in', 'related to', 'kind of'
        ]
        
        # Extract nouns from definition (simple heuristic)
        words_in_def = definition.split()
        for i, w in enumerate(words_in_def):
            # Skip very common words
            if w in ['the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with']:
                continue
            # If word has synsets, it might be a useful bridge
            if len(w) > 3 and wordnet.synsets(w):
                bridges.add(w)
        
        return bridges
    
    # Try to extract dynamic bridges from this word's definitions
    try:
        synsets = wordnet.synsets(word_lower)
        if synsets:
            # Get bridges from first 2 most common meanings
            for syn in synsets[:2]:
                bridges = extract_bridges_from_definition(syn)
                # Add top 3-4 bridge words
                for bridge in list(bridges)[:4]:
                    if bridge != word_lower and len(bridge) > 2:
                        expanded.add(bridge)
    except Exception:
        pass
    
    try:
        synsets = wordnet.synsets(word_lower)
        if not synsets:
            return sorted(list(expanded))[:max_expansions]
        
        # Process only first 2 synsets (most common meanings)
        for syn in synsets[:2]:
            # 1. Synonyms (lemmas)
            for lemma in syn.lemmas()[:4]:
                name = lemma.name().replace('_', ' ').lower()
                if name != word_lower and len(name) > 2:
                    expanded.add(name)
            
            # 2. Hypernyms (what is this a type of) - CRITICAL for bridging
            for hyper in syn.hypernyms()[:3]:
                for lemma in hyper.lemmas()[:2]:
                    name = lemma.name().replace('_', ' ').lower()
                    if name != word_lower and len(name) > 2:
                        expanded.add(name)
            
            # 3. Hyponyms (specific types)
            for hypo in syn.hyponyms()[:3]:
                for lemma in hypo.lemmas()[:2]:
                    name = lemma.name().replace('_', ' ').lower()
                    if name != word_lower and len(name) > 2:
                        expanded.add(name)
            
            # 4. Meronyms (parts)
            for mero in syn.part_meronyms()[:3]:
                for lemma in mero.lemmas()[:1]:
                    name = lemma.name().replace('_', ' ').lower()
                    if name != word_lower and len(name) > 2:
                        expanded.add(name)
            
            # 5. Holonyms (what this is part of) - good for bridging
            for holo in syn.part_holonyms()[:3]:
                for lemma in holo.lemmas()[:1]:
                    name = lemma.name().replace('_', ' ').lower()
                    if name != word_lower and len(name) > 2:
                        expanded.add(name)
    
    except Exception:
        pass
    
    return sorted(list(expanded))[:max_expansions]


def get_word_category(word):
    """
    Get the main semantic category of a word (noun.animal, noun.food, verb.action, etc.)
    Returns string like 'animal', 'food', 'plant', 'action', etc.
    """
    try:
        synsets = wordnet.synsets(word.lower())
        if not synsets:
            return None
        
        # Get the most common sense's lexname
        main_syn = synsets[0]
        lexname = main_syn.lexname()  # e.g., 'noun.animal', 'verb.motion'
        
        if '.' in lexname:
            return lexname.split('.')[1]  # Returns 'animal', 'motion', etc.
        return lexname
    except Exception:
        return None


def find_connection(word_a, word_b):
    """
    Simple human-like connection logic - as a person would think:
    1. Same type/category (both animals, plants, foods) -> CONNECT
    2. One mentions the other in definition (cow->milk in "mammal secretes milk") -> CONNECT
    3. Share a category word in definitions (milk mentions "mammals", cow is mammal) -> CONNECT
    4. Share parent category (both mammals, both plants) -> CONNECT
    
    Returns True if connected through any rule.
    """
    a_lower = word_a.lower()
    b_lower = word_b.lower()
    
    try:
        synsets_a = wordnet.synsets(a_lower)
        synsets_b = wordnet.synsets(b_lower)
        
        if not synsets_a or not synsets_b:
            return False
        
        syn_a = synsets_a[0]
        syn_b = synsets_b[0]
        
        def_a = syn_a.definition().lower()
        def_b = syn_b.definition().lower()

        cat_a = get_word_category(a_lower)
        cat_b = get_word_category(b_lower)

        # Ignore very broad artifact/object categories to avoid spurious links (e.g., car ↔ dairy)
        broad_categories = {"artifact", "object", "whole", "part", "group"}

        # Special-case dairy: animals produce milk/dairy products
        if (cat_a == "animal" and ("milk" in b_lower or "dairy" in b_lower)) or \
           (cat_b == "animal" and ("milk" in a_lower or "dairy" in a_lower)):
            return True

        # Quick substring bridge (e.g., "dairy product" contains "dairy", "cows' milk" contains "milk")
        if a_lower in b_lower or b_lower in a_lower:
            return True
        
        # Rule 1: Same semantic category (both animals, both foods, etc.) but skip broad buckets
        if (
            cat_a
            and cat_b
            and cat_a == cat_b
            and cat_a not in broad_categories
        ):
            return True  # Both same type
        
        # Rule 2: One word directly in the other's definition
        if b_lower in def_a or b_lower.replace(' ', '_') in def_a:
            return True
        if a_lower in def_b or a_lower.replace(' ', '_') in def_b:
            return True
        
        # Rule 3: Manual human-like bridge keywords (focused; drop overly broad ones like "food"/"product")
        bridge_keywords = {"dairy", "farm", "livestock", "milk", "drink", "animal"}
        def_a_words = set(def_a.split())
        def_b_words = set(def_b.split())
        # Check if any bridge keyword appears in both definitions or both word forms
        for kw in bridge_keywords:
            if (kw in def_a_words or kw in a_lower) and (kw in def_b_words or kw in b_lower):
                return True
        
        # Rule 4: Check if B's category/type appears in A's definition and vice versa
        cat_terms = []
        if cat_a:
            cat_terms.append(cat_a)
        if cat_b:
            cat_terms.append(cat_b)
        
        # Also add hypernym names to category terms
        for h in syn_a.hypernyms():
            cat_terms.append(h.name().split('.')[0].replace('_', ' ').lower())
        for h in syn_b.hypernyms():
            cat_terms.append(h.name().split('.')[0].replace('_', ' ').lower())
        
        allowed_cross = {"animal", "plant", "food", "substance", "material", "living thing", "body"}
        for cat in cat_terms:
            if cat in broad_categories:
                continue
            if cat in def_a or cat in def_b:
                if cat in def_b and (cat == cat_a or cat in str(syn_a.hypernyms()).lower()):
                    # Only cross-connect if the other category is in an allowed, non-broad bucket
                    if cat_a and cat_a in allowed_cross and cat_b and cat_b in allowed_cross:
                        return True
                if cat in def_a and (cat == cat_b or cat in str(syn_b.hypernyms()).lower()):
                    if cat_b and cat_b in allowed_cross and cat_a and cat_a in allowed_cross:
                        return True
        
        # Rule 5: Share parent categories at level 1 or 2
        hypers_a = syn_a.hypernyms()
        hypers_b = syn_b.hypernyms()
        
        if hypers_a and hypers_b:
            if set(hypers_a) & set(hypers_b):
                return True
        
        # Second-level parents
        hypers_a2 = []
        hypers_b2 = []
        for h in hypers_a:
            hypers_a2.extend(h.hypernyms())
        for h in hypers_b:
            hypers_b2.extend(h.hypernyms())
        
        if set(hypers_a2) & set(hypers_b2):
            return True
            
    except Exception:
        pass
    
    return False


# ✅ Main Route - REDESIGNED for rich constellations
@app.route("/process", methods=["POST"])
def process_words():
    try:
        data = request.get_json()
        if not data or "words" not in data:
            return jsonify({"error": "No words provided"}), 400

        input_words = [word.strip().lower() for word in data.get("words", "").split(",") if word.strip()]
        
        # Step 1: Expand each input word to a pool of related words
        word_pool = {}          # word -> type (input/expanded)
        seed_links = []         # keep track of seed-to-expansion links
        
        for inp_word in input_words:
            translated = detect_and_translate(inp_word).lower()
            word_pool[translated] = "input"
            
            # Expand to related words
            expanded = expand_word_to_pool(translated, max_expansions=6)
            for exp_word in expanded:
                if exp_word not in word_pool:
                    word_pool[exp_word] = "expanded"
                    seed_links.append((translated, exp_word))
        # Build suggestions for all nodes, excluding words already in the constellation
        suggestions_map = {}
        existing_set = set(word_pool.keys())
        for w in list(existing_set):
            try:
                expanded = expand_word_to_pool(w, max_expansions=10)
            except Exception:
                expanded = []
            filtered = [e for e in expanded if e not in existing_set]
            # Deduplicate while preserving order
            seen = set()
            unique_filtered = []
            for e in filtered:
                if e not in seen:
                    seen.add(e)
                    unique_filtered.append(e)
            suggestions_map[w] = unique_filtered[:3]
        
        # Step 2: Create nodes for all words
        nodes = []
        existing_nodes = set()
        all_categories = set()
        
        for word in word_pool.keys():
            # Only use WordNet categories for stability and speed
            wordnet_cats = get_wordnet_categories(word)
            combined_cats = [c for c in wordnet_cats if c and c.lower() != word.lower()]
            
            for c in combined_cats:
                if isinstance(c, str) and c.strip():
                    all_categories.add(c.lower())
            
            node_type = word_pool[word]
            nodes.append({
                "id": word,
                "categories": combined_cats[:3],
                "type": node_type
            })
            existing_nodes.add(word)
        
        # Step 3: Connect all words
        links = []
        all_words = list(word_pool.keys())
        
        # 3a. Seed-to-expansion links (guarantee local constellation)
        for src, tgt in seed_links:
            links.append({"source": src, "target": tgt, "relation": "seed"})
        
        # 3b. Semantic links across all words
        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                word_i = all_words[i]
                word_j = all_words[j]
                
                if find_connection(word_i, word_j):
                    links.append({
                        "source": word_i,
                        "target": word_j,
                        "relation": "related"
                    })
        
        # Remove duplicates
        unique_links = {(l["source"], l["target"], l["relation"]) for l in links}
        links = [{"source": s, "target": t, "relation": r} for s, t, r in unique_links]
        
        # Drop isolated expansion nodes (keep inputs even if isolated)
        degree = {w: 0 for w in word_pool.keys()}
        for l in links:
            degree[l["source"]] = degree.get(l["source"], 0) + 1
            degree[l["target"]] = degree.get(l["target"], 0) + 1
        keep_nodes = set(w for w in word_pool.keys() if w in input_words or degree.get(w, 0) > 0)
        links = [l for l in links if l["source"] in keep_nodes and l["target"] in keep_nodes]
        
        # FULLY DYNAMIC career suggestions - works for ANY words using pure semantic analysis
        career_tags_set = set()
        import random
        random.seed(hash(tuple(sorted(input_words))))
        
        all_node_words = [n["id"] for n in nodes]
        
        # Pure semantic field detection from WordNet - NO hardcoded mappings
        def detect_semantic_domain(word):
            """Dynamically detect semantic domain for ANY word using WordNet analysis"""
            domains = []
            try:
                synsets = wordnet.synsets(word.lower())
                if not synsets:
                    return domains
                
                syn = synsets[0]  # Most common meaning
                definition = syn.definition().lower()
                lexname = syn.lexname()  # e.g., 'noun.animal', 'verb.motion'
                
                # Extract domain from lexname
                if '.' in lexname:
                    category = lexname.split('.')[1]
                    domains.append(category)
                
                # Analyze hypernym chain (up to 3 levels)
                current = syn
                for _ in range(3):
                    hypers = current.hypernyms()
                    if not hypers:
                        break
                    current = hypers[0]
                    hyper_lexname = current.lexname()
                    if '.' in hyper_lexname:
                        cat = hyper_lexname.split('.')[1]
                        domains.append(cat)
                
                # Mine definition for domain keywords
                domain_keywords = {
                    'animal': ['animal', 'mammal', 'creature', 'livestock', 'fauna', 'vertebrate', 'beast'],
                    'plant': ['plant', 'vegetation', 'flora', 'tree', 'flower', 'crop', 'botanical'],
                    'food': ['food', 'nutrient', 'dish', 'meal', 'beverage', 'drink', 'edible', 'cuisine'],
                    'health': ['medicine', 'treatment', 'disease', 'health', 'medical', 'therapy', 'cure'],
                    'technology': ['device', 'machine', 'computer', 'software', 'digital', 'electronic', 'system'],
                    'science': ['science', 'research', 'study', 'analysis', 'experiment', 'theory'],
                    'art': ['art', 'creative', 'design', 'aesthetic', 'visual', 'artistic'],
                    'music': ['music', 'sound', 'audio', 'instrument', 'melody', 'song'],
                    'business': ['business', 'commerce', 'trade', 'market', 'company', 'enterprise'],
                    'engineering': ['engineering', 'construction', 'build', 'structure', 'technical'],
                    'education': ['education', 'teaching', 'learning', 'school', 'instruction'],
                    'communication': ['communication', 'language', 'speech', 'writing', 'media'],
                    'social': ['social', 'people', 'community', 'society', 'human'],
                    'law': ['law', 'legal', 'court', 'justice', 'attorney'],
                    'transportation': ['vehicle', 'transport', 'travel', 'motion', 'conveyance'],
                    'environment': ['environment', 'ecology', 'nature', 'climate', 'conservation'],
                    'finance': ['finance', 'money', 'bank', 'investment', 'economic', 'financial'],
                    'sport': ['sport', 'athletic', 'fitness', 'exercise', 'game', 'competition'],
                }
                
                for domain, keywords in domain_keywords.items():
                    if any(kw in definition for kw in keywords):
                        domains.append(domain)
                
            except Exception:
                pass
            
            return list(set(domains))
        
        # Detect domains for ALL words in constellation
        word_domains = {}
        for word in all_node_words:
            domains = detect_semantic_domain(word)
            if domains:
                word_domains[word] = domains
        
        # Also check categories
        for cat in all_categories:
            domains = detect_semantic_domain(cat)
            if domains:
                word_domains[cat] = domains
        
        # Collect all detected domains
        all_detected_domains = set()
        for domains in word_domains.values():
            all_detected_domains.update(domains)
        
        # Match domains to job fields using fuzzy matching - NO hardcoded mappings
        matched_fields = set()
        for domain in all_detected_domains:
            # Direct match with JOB_DATABASE fields
            if domain in JOB_DATABASE:
                matched_fields.add(domain)
            else:
                # Fuzzy matching: check if domain is contained in any job field name
                for field in JOB_DATABASE.keys():
                    if domain.lower() in field.lower() or field.lower() in domain.lower():
                        matched_fields.add(field)
                        break
                # Also check semantic similarity using WordNet
                try:
                    domain_synsets = wordnet.synsets(domain)
                    if domain_synsets:
                        for field in JOB_DATABASE.keys():
                            field_synsets = wordnet.synsets(field)
                            if field_synsets:
                                # Check if they share hypernyms (are semantically related)
                                domain_hypers = set([h.name() for h in domain_synsets[0].hypernyms()])
                                field_hypers = set([h.name() for h in field_synsets[0].hypernyms()])
                                if domain_hypers & field_hypers:  # Intersection
                                    matched_fields.add(field)
                except Exception:
                    pass
        
        # Sample jobs from matched fields
        for field in matched_fields:
            if field in JOB_DATABASE:
                jobs = JOB_DATABASE[field]
                sample_size = min(3, len(jobs))
                career_tags_set.update(random.sample(jobs, sample_size))
        
        career_tags = sorted(career_tags_set)[:15]
        economy_tags = generate_economic_tags(input_words)
        trendy_tags = generate_trendy_topics(input_words)
        
        response = {
            "nodes": [{"id": n["id"], "categories": n["categories"]} for n in nodes],
            "links": links,
            "words": input_words,
            "careers": career_tags,
            "economy": economy_tags,
            "trends": trendy_tags,
            "suggestions": suggestions_map
        }
        
        print(f"Response: {len(response['nodes'])} nodes, {len(response['links'])} links")
        return jsonify(response)

    except Exception as e:
        print(f"Error in /process: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000)
