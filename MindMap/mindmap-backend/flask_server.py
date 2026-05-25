from flask import Flask, request, jsonify
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import wordnet
from faker.providers.job.en_US import Provider as JobProvider
import random
import json
import os
import re
import traceback


#reads the economic sectors json file and returns it, empty dict if file is missing
def load_economic_sectors():
    file_path = 'economic_sectors.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

ECONOMIC_SECTORS_DB = load_economic_sectors()


#reads the trending topics json file, same deal as above
def load_trending_topics():
    file_path = 'trending_topics.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

TRENDING_TOPICS_DB = load_trending_topics()


#reads the domain keywords json file
def load_domain_keywords():
    file_path = 'domain_keywords.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

DOMAIN_KEYWORDS_DB = load_domain_keywords()


app = Flask(__name__)


#these headers allow the frontend to receive responses from other adresses 
#without this the browser would block the response due to CORS policy
@app.after_request
def add_cors_headers(response):
    #any website
    response.headers['Access-Control-Allow-Origin'] = '*'
    #allowed http methods
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    #for the json data include a content-type header for seeing its json
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


#wordnet
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

#extra wordnet files
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

#the list for carrere suggestions
JOB_DATABASE = list(set(JobProvider.jobs))

PROFANITY_KEYWORDS = {
    'shit', 'damn', 'hell', 'ass', 'bitch', 'crap', 'piss', 'dick', 'cock',
    'pussy', 'fuck', 'tit', 'wank', 'bollocks', 'arse', 'bastard', 'twat',
    'douchebag', 'asshole', 'jerk', 'fag', 'faggot', 'homo', 'dyke',
    'nigger', 'retard', 'slut', 'whore', 'skank', 'tramp', 'hoe',
    'douche', 'arsehole', 'offender', 'shitter', 'fucker',
    'xenophobic', 'zoophile', 'queer', 'queerish', 'queerly',
    'racist', 'racists', 'homosexual', 'homosexuals',
    'kike', 'wetback', 'spick', 'chink', 'gook', 'paki', 'raghead',
    'kill', 'murder', 'rape', 'torture', 'die', 'dead', 'killed',
    'raped', 'tortured', 'died',
    'nazi', 'naziism', 'hitler', 'hitlerism', 'fascism', 'fascist',
    'communism', 'communist', 'mormon', 'mormonism',
}


# checks if a word is in the banned list
def is_profanity(word):
    return word.lower().strip() in PROFANITY_KEYWORDS


#if a word is not english it tries to translate it to english so wordnet can understand it
def detect_and_translate(text):
    text_lower = text.lower().strip()

    if wordnet.synsets(text_lower):
        return text_lower

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


#strips leading articles like "a" and lowercases the string
def clean_label(label):
    return label.strip().lower().replace("a ", "").replace("an ", "").replace("the ", "")


#returns the wordnet category tags for a word when i hover over a line in the constellation
def get_wordnet_categories(word):
    synsets = wordnet.synsets(word)
    cats = []
    for syn in synsets:
        parts = syn.lexname().split(".")#split the categories
        if len(parts) > 1:
            cats.append(parts[1])#only the second part
        else:
            cats.append(parts[0])#gets whatever it gets
    return list(set(cats)) if synsets else []#lista goala sau lista cu categorii unice



#checks which economic sectors match the context of a word
def generate_economic_tags(words):
    tags_set = set()#unice

    for word in words:
        try:
            synsets = wordnet.synsets(word.lower())
            if not synsets:
                continue

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

            full_context = f"{word.lower()} {definition} {' '.join(hypernyms)}"
            clean_context = re.sub(r'[^\w\s]', '', full_context)#scot punctuatia
            padded_context = f" {clean_context} "#las spatiu 

            for sector, indicators in ECONOMIC_SECTORS_DB.items():
                if any(f" {indicator} " in padded_context for indicator in indicators):
                    tags_set.add(sector)

        except Exception:
            pass

    return sorted(list(tags_set))[:8]#maxim 8 tags


# same as economic tags but checks against trending topics, also adds synonyms to the context
def generate_trendy_topics(words):
    trends_set = set()

    for word in words:
        try:
            synsets = wordnet.synsets(word.lower())
            if not synsets:
                continue

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

            full_context = f"{word.lower()} {definition} {' '.join(hypernyms)} {' '.join(lemmas)}"
            clean_context = re.sub(r'[^\w\s]', '', full_context)
            padded_context = f" {clean_context} "

            for trend, indicators in TRENDING_TOPICS_DB.items():
                if any(f" {ind} " in padded_context for ind in indicators):
                    trends_set.add(trend)

        except Exception:
            pass

    return sorted(list(trends_set))[:8]


#similarity score between 0 and 1 for two words using wu-palmer
def _wup_sim(word_a, word_b):
    try:
        sa = wordnet.synsets(word_a)
        sb = wordnet.synsets(word_b)
        if not sa or not sb:
            return None
        return sa[0].wup_similarity(sb[0])
    except Exception:
        return None


#takes one word and finds related words from wordnet,then filters out anything too dissimilar
def expand_word_to_pool(word, max_expansions=10):
    word_lower = word.lower()
    expanded = set()

    try:
        synsets = wordnet.synsets(word_lower)
        if not synsets:
            return []

        primary = synsets[0]

        direct_synonyms = set()
        for lemma in primary.lemmas()[:5]:#doar primele 5 sinonime
            name = lemma.name().replace('_', ' ').lower()#fara _ de la wordnet
            if name != word_lower and len(name) > 2:
                expanded.add(name)
                direct_synonyms.add(name)

        for hyper in primary.hypernyms()[:2]:
            for lemma in hyper.lemmas()[:2]:
                name = lemma.name().replace('_', ' ').lower()
                if name != word_lower and len(name) > 2:
                    expanded.add(name)

        for hypo in primary.hyponyms()[:6]:
            if hypo.lemmas():
                name = hypo.lemmas()[0].name().replace('_', ' ').lower()
                if name != word_lower and len(name) > 2:
                    expanded.add(name)

        for mero in primary.part_meronyms()[:3]:
            if mero.lemmas():
                name = mero.lemmas()[0].name().replace('_', ' ').lower()
                if name != word_lower and len(name) > 2:
                    expanded.add(name)

        result = []
        for candidate in expanded:
            if candidate in direct_synonyms:#sinonimele directe raman direct, restul le filtram dupa scor
                result.append(candidate)
                continue
            score = _wup_sim(word_lower, candidate)
            if score is None or score >= 0.3:
                result.append(candidate)

        return sorted(result)[:max_expansions]#maxim 10 

    except Exception:
        return sorted(list(expanded))[:max_expansions]


#pt a compara dacă două cuvinte sunt din aceeași categorie in functia find_connection
def get_word_category(word):
    try:
        synsets = wordnet.synsets(word.lower())
        if not synsets:
            return None

        main_syn = synsets[0]
        lexname = main_syn.lexname()#noun.animal

        if '.' in lexname:
            return lexname.split('.')[1]#luam doar partea dupa punct care e categoria adica a doua dupa .
        return lexname
    except Exception:
        return None


# checks if two words are related enough to draw a link between them
def find_connection(word_a, word_b):
    a_lower = word_a.lower()
    b_lower = word_b.lower()

    try:
        synsets_a = wordnet.synsets(a_lower)
        synsets_b = wordnet.synsets(b_lower)

        if not synsets_a or not synsets_b:
            return False

        syn_a = synsets_a[0]
        syn_b = synsets_b[0]

        #if the similarity score is too low they are not related
        sim = syn_a.wup_similarity(syn_b)
        if sim is not None and sim < 0.2:
            return False

        cat_a = get_word_category(a_lower)
        cat_b = get_word_category(b_lower)
        broad_categories = {"artifact", "object", "whole", "part", "group"}

        #same wordnet category means they are related,but ignore broad categories
        if cat_a and cat_b and cat_a == cat_b and cat_a not in broad_categories:
            return True

        #one word appears in the other's definition
        def_a = syn_a.definition().lower()
        def_b = syn_b.definition().lower()
        if b_lower in def_a or b_lower.replace(' ', '_') in def_a:
            return True
        if a_lower in def_b or a_lower.replace(' ', '_') in def_b:
            return True

        #they share the same immediate parent category in wordnet
        if set(syn_a.hypernyms()) & set(syn_b.hypernyms()):
            return True

    except Exception:
        pass

    return False


# figures out which broad domains a word belongs to using its wordnet lexname and definition
def detect_semantic_domain(word):
    domains = []
    try:
        synsets = wordnet.synsets(word.lower())
        if not synsets:
            return domains

        syn = synsets[0]
        definition = syn.definition().lower()
        lexname = syn.lexname()

        if '.' in lexname:
            domains.append(lexname.split('.')[1])

        current = syn
        for _ in range(3):
            hypers = current.hypernyms()
            if not hypers:
                break
            current = hypers[0]
            hyper_lexname = current.lexname()
            if '.' in hyper_lexname:
                domains.append(hyper_lexname.split('.')[1])

        clean_def = re.sub(r'[^\w\s]', '', definition)
        padded_def = f" {clean_def} "
        for domain, keywords in DOMAIN_KEYWORDS_DB.items():
            if any(f" {kw} " in padded_def for kw in keywords):
                domains.append(domain)

    except Exception:
        pass

    return list(set(domains))


# finds job titles from faker that match the semantic domains of the input words
def generate_career_tags(words):
    random.seed(hash(tuple(sorted(words))))

    all_detected_domains = set()
    for word in words:
        all_detected_domains.update(detect_semantic_domain(word))

    matched_jobs = set()
    for domain in all_detected_domains:
        search_terms = {domain.lower()}
        for syn in wordnet.synsets(domain):
            for lemma in syn.lemmas():
                search_terms.add(lemma.name().lower().replace('_', ' '))
        for job in JOB_DATABASE:
            if any(term in job.lower().split() for term in search_terms):
                matched_jobs.add(job)

    return sorted(random.sample(list(matched_jobs), min(5, len(matched_jobs)))) if matched_jobs else []


#takes a comma separated list of words and returns the full data which 
#is nodes, links, career suggestions, economic tags and trendy topics
@app.route("/process", methods=["POST"])
def process_words():
    try:
        data = request.get_json()#jsonul din frontend
        if not data or "words" not in data:
            return jsonify({"error": "No words provided"}), 400

        input_words = [word.strip().lower() for word in data.get("words", "").split(",") if word.strip()]#curatare si split dupa virgula

        word_pool = {}
        seed_links = []

        for inp_word in input_words:
            translated = detect_and_translate(inp_word).lower()
            word_pool[translated] = "input"#translate them if not in english and add them

            expanded = expand_word_to_pool(translated, max_expansions=6)#find relative words
            for exp_word in expanded:
                if exp_word not in word_pool:
                    word_pool[exp_word] = "expanded"
                    seed_links.append((translated, exp_word))#link between input word and its expansions

        #for each node find words that could be added next,excluding ones already in the map
        existing_set = set(word_pool.keys())#to be able to suggest only words that are not already in the map
        suggestions_map = {}
        for w in list(existing_set):
            try:
                expanded = expand_word_to_pool(w, max_expansions=10)#related words
            except Exception:
                expanded = []
            suggestions_map[w] = [e for e in expanded if e not in existing_set][:3]#max 3 sug per word if they are not in the existing set

        #create the nodes with id, categories and type(input or expanded)
        nodes = []
        for word in word_pool.keys():#pt fiecare cuv din map facem un nod
            wordnet_cats = get_wordnet_categories(word)
            combined_cats = [c for c in wordnet_cats if c and c.lower() != word.lower()]#categorie!=cuvant
            nodes.append({
                "id": word,
                "categories": combined_cats[:3],
                "type": word_pool[word]#cat=> type: input sau expanded
            })

        #create the links between nodes
        links = []
        all_words = list(word_pool.keys())#lista cu toate cuvintele din map

        for src, tgt in seed_links:#src=cuvant input, tgt= cuvant expansiune
            links.append({"source": src, "target": tgt, "relation": "seed"})#seed link prin input-expansion

        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):#compare each pair of words to see if they link
                if find_connection(all_words[i], all_words[j]):
                    links.append({"source": all_words[i], "target": all_words[j], "relation": "related"})#related link prin find connection

        #remove duplicate links with set
        unique_links = {(l["source"], l["target"], l["relation"]) for l in links}
        links = [{"source": s, "target": t, "relation": r} for s, t, r in unique_links]

        #sterge noduri fara legaturi si care nu sunt input
        degree = {w: 0 for w in word_pool.keys()}#initializez gradul de legatura pt fiecare cuv
        for l in links:
            #gradul de legatura ii creste cu 1 daca e sursa sau tinta intr o legatura
            degree[l["source"]] = degree.get(l["source"], 0) + 1
            degree[l["target"]] = degree.get(l["target"], 0) + 1
        keep_nodes = set(w for w in word_pool.keys() if w in input_words or degree.get(w, 0) > 0)#only keep words that are input or have a link
        links = [l for l in links if l["source"] in keep_nodes and l["target"] in keep_nodes]#filter links to only keep those between kept nodes

        #response which is a dict 
        response = {
            "nodes": [{"id": n["id"], "categories": n["categories"]} for n in nodes],
            "links": links,
            "words": input_words,
            "careers": generate_career_tags(input_words),
            "economy": generate_economic_tags(input_words),
            "trends": generate_trendy_topics(input_words),
            "suggestions": suggestions_map
        }

        
        print(f"Response: {len(response['nodes'])} nodes, {len(response['links'])} links")#log to see what is being sent
        #transform the dict in json to send it to the frontend
        return jsonify(response)

    #to see exactly what error and on what line it happened and tells the frontend that the server has broken
    except Exception as e:
        print(f"Error in /process: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#just a ping to check the server is alive
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "Flask NLP"}), 200


#verifica cuv din frontend inainte sa fie date la functia de process 
@app.route('/check-profanity', methods=['POST'])
def check_profanity():
    try:
        data = request.get_json() or {}
        words_input = data.get('words', '').strip()#le scoatem spatiile inainte si dupa si returneaza "" in loc sa crape daca nu e words

        if not words_input:
            return jsonify({"clean": True, "rejected_words": []})

        words_list = [w.strip().lower() for w in words_input.split(',') if w.strip()]#curatare si split dupa virgula
        rejected = [w for w in words_list if is_profanity(w)]

        return jsonify({
            "clean": len(rejected) == 0,
            "rejected_words": rejected,
            "total_words": len(words_list),
            "message": f"Found {len(rejected)} inappropriate word(s)." if rejected else "All words are acceptable."
        })

    except Exception as e:
        print(f"Error in /check-profanity: {e}")
        return jsonify({"error": str(e), "clean": False}), 500


#definitia si partofspeech pentru un cuvand din map
@app.route('/define', methods=['POST'])
def define_word():
    try:
        data = request.get_json() or {}
        word = (data.get('word') or '').strip()
        if not word:
            return jsonify({'definition': None, 'partOfSpeech': None}), 400

        pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}#cuv intreg inloc de litera din wordnet

        def lookup(w):
            key = w.replace(' ', '_').replace('-', '_')#punem _ ca sa caute in wordnet
            syns = wordnet.synsets(key)
            if not syns:
                syns = wordnet.synsets(w.split()[0])#macar sa caute primul cuvant
            if syns:
                return syns[0].definition(), pos_map.get(syns[0].pos(), '')#def si part of speech
            return None, None

        definition, pos = lookup(word)
        if definition:
            return jsonify({'definition': definition, 'partOfSpeech': pos})
        return jsonify({'definition': None, 'partOfSpeech': None}), 404

    except Exception as e:
        print(f"Error in /define: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":#pornește serverul doar dacă fișierul e rulat direct, nu dacă e importat
    app.run(debug=False, threaded=True, host='127.0.0.1', port=5000)
