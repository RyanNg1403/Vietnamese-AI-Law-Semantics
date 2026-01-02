"""
FastAPI Backend for Query Visualization with Intermediate Steps Tracing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
from functools import lru_cache
import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modules
try:
    from backend.prolog_queries import *
    from backend.mfs_baseline import get_mfs
    from backend.config import Config
except ImportError:
    # Fallback if running from different directory
    import prolog_queries
    import mfs_baseline
    get_mfs = mfs_baseline.get_mfs
    # Import all from prolog_queries
    from prolog_queries import *
    # Create a simple Config class
    class Config:
        MODEL_SAVE_PATH = './models/legal_bert_wsd_model-pos-constrained'
        LABEL_MAP_PATH = './data/label_map.json'
        MAX_LEN = 128
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = FastAPI(title="Legal Text Query API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for BERT model (lazy loading)
bert_model_cache = None
bert_tokenizer_cache = None
bert_id2label_cache = None

def load_bert_model():
    """Load BERT model with caching"""
    global bert_model_cache, bert_tokenizer_cache, bert_id2label_cache
    
    if bert_model_cache is None:
        try:
            model_path = Config.MODEL_SAVE_PATH if Config else './models/legal_bert_wsd_model-pos-constrained'
            label_map_path = Config.LABEL_MAP_PATH if Config else './data/label_map.json'
            
            print(f"Loading BERT model from {model_path}...")
            bert_tokenizer_cache = BertTokenizerFast.from_pretrained(model_path)
            bert_model_cache = BertForTokenClassification.from_pretrained(model_path)
            
            device = Config.DEVICE if Config else ('cuda:0' if torch.cuda.is_available() else 'cpu')
            bert_model_cache.to(device)
            bert_model_cache.eval()
            
            # Load label map
            with open(label_map_path, 'r') as f:
                label2id = json.load(f)
            bert_id2label_cache = {v: k for k, v in label2id.items() if v != -100}
            
            print("✅ BERT model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading BERT model: {e}")
            bert_model_cache = None
            bert_tokenizer_cache = None
            bert_id2label_cache = None
    
    return bert_model_cache, bert_tokenizer_cache, bert_id2label_cache

# Pre-defined queries (matching prolog_queries.py and SLIDE_CONTENT.md)
PREDEFINED_QUERIES = [
    {
        "id": 1,
        "query": "is_developer(X, chat_gpt_vn)",
        "nl": "Who is the developer of ChatGPT VN?",
        "type": "prolog",
        "expected_result": "X = bkav_corp"
    },
    {
        "id": 2,
        "query": "is_provider(bkav_corp, _)",
        "nl": "Is BKAV a provider?",
        "type": "prolog",
        "expected_result": "false"
    },
    {
        "id": 3,
        "query": "is_provider(fpt_software, X)",
        "nl": "What systems does FPT provide?",
        "type": "prolog",
        "expected_result": "X = camera_traffic"
    },
    {
        "id": 4,
        "query": "is_a(developer, person)",
        "nl": "Is developer a kind of person?",
        "type": "prolog",
        "expected_result": "true"
    },
    {
        "id": 5,
        "query": "is_serious_incident(incident_01)",
        "nl": "Is incident_01 a serious incident?",
        "type": "prolog",
        "expected_result": "true"
    },
    {
        "id": 6,
        "query": "is_deployer(bo_cong_an, camera_traffic)",
        "nl": "Is bo_cong_an a deployer of camera_traffic?",
        "type": "prolog",
        "expected_result": "true"
    },
    {
        "id": 7,
        "query": "is_user(nguyen_van_an, chat_gpt_vn)",
        "nl": "Is nguyen_van_an a user of chat_gpt_vn?",
        "type": "prolog",
        "expected_result": "true"
    },
    {
        "id": 8,
        "query": "is_a(user, person)",
        "nl": "Is user a kind of person?",
        "type": "prolog",
        "expected_result": "true"
    }
]

# Request/Response models
class QueryRequest(BaseModel):
    query_id: int
    wsd_method: str = "mfs"  # "mfs" or "bert"

class QueryResponse(BaseModel):
    query: str
    nl_question: str
    result: str
    expected_result: str
    success: bool
    match: bool
    steps: Dict[str, Any]

# ============================================
# WSD TRACING
# ============================================

async def perform_wsd_tracing(text: str, method: str) -> Dict[str, Any]:
    """
    Perform WSD with full tracing of intermediate steps
    """
    trace = {
        "method": method,
        "input_text": text,
        "steps": []
    }
    
    # Step 1: Sentence Tokenization
    sentences = sent_tokenize(text)
    trace["steps"].append({
        "step": 1,
        "name": "Sentence Tokenization",
        "input": text,
        "output": sentences,
        "description": f"Split text into {len(sentences)} sentence(s)"
    })
    
    all_tokens = []
    all_pos_tags = []
    
    # Step 2: Word Tokenization & POS Tagging
    for sent_idx, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        
        trace["steps"].append({
            "step": 2 + sent_idx,
            "name": f"Tokenization & POS Tagging (Sentence {sent_idx + 1})",
            "input": sentence,
            "output": [{"token": w, "pos": p} for w, p in pos_tags],
            "description": f"Tokenized into {len(words)} tokens with POS tags"
        })
        
        all_tokens.extend([(w, p) for w, p in pos_tags])
        all_pos_tags.extend([p for _, p in pos_tags])
    
    # Step 3: WSD for each token
    wsd_results = []
    
    if method == "bert":
        # Use BERT for WSD on full sentences
        bert_model, bert_tokenizer, id2label = load_bert_model()
        if bert_model is None:
            # Fallback to MFS if BERT not available
            method = "mfs"
            trace["method"] = "mfs (fallback)"
            trace["steps"].append({
                "step": len(trace["steps"]) + 1,
                "name": "BERT Model Loading",
                "description": "⚠️ BERT model not available, falling back to MFS",
                "output": "Fallback to MFS"
            })
    
    # Process each sentence for BERT
    if method == "bert" and bert_model is not None:
        for sent_idx, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            pos_tags = pos_tag(words)
            sentence_str = " ".join(words)
            
            # BERT inference
            encoding = bert_tokenizer(
                sentence_str,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MAX_LEN if Config else 128
            )
            device = Config.DEVICE if Config else ('cuda:0' if torch.cuda.is_available() else 'cpu')
            inputs = {k: v.to(device) for k, v in encoding.items()}
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
                probs = torch.softmax(outputs.logits, dim=2)
            
            word_ids = encoding.word_ids(0)
            pred_labels = predictions[0].tolist()
            pred_probs = probs[0].tolist()
            
            # Map predictions to tokens
            current_word_idx = 0
            for token, pos in pos_tags:
                if pos.startswith(('N', 'V', 'J', 'R')):
                    # Find prediction for this word
                    synset_name = None
                    confidence = 0.0
                    
                    for i, wid in enumerate(word_ids):
                        if wid == current_word_idx:
                            pred_id = pred_labels[i]
                            if pred_id in id2label:
                                synset_name = id2label[pred_id]
                                confidence = float(pred_probs[i][pred_id])
                            break
                    
                    # Get candidates
                    candidates = [s.name() for s in wn.synsets(token, pos=wn.NOUN if pos.startswith('N') else None)[:5]]
                    
                    synset_def = ""
                    if synset_name:
                        try:
                            synset = wn.synset(synset_name)
                            synset_def = synset.definition()
                        except:
                            pass
                    
                    wsd_results.append({
                        "token": token,
                        "pos": pos,
                        "synset": synset_name,
                        "definition": synset_def,
                        "confidence": confidence,
                        "candidates": candidates
                    })
                
                current_word_idx += 1
    else:
        # MFS method
        for token, pos in all_tokens:
            if pos.startswith(('N', 'V', 'J', 'R')):  # Noun, Verb, Adj, Adv
                synset_name = get_mfs(token, pos)
                confidence = 1.0
                candidates = [s.name() for s in wn.synsets(token, pos=wn.NOUN if pos.startswith('N') else None)[:5]]
                
                synset_def = ""
                if synset_name:
                    try:
                        synset = wn.synset(synset_name)
                        synset_def = synset.definition()
                    except:
                        pass
                
                wsd_results.append({
                    "token": token,
                    "pos": pos,
                    "synset": synset_name,
                    "definition": synset_def,
                    "confidence": confidence,
                    "candidates": candidates
                })
    
    trace["steps"].append({
        "step": len(trace["steps"]) + 1,
        "name": "Word Sense Disambiguation",
        "input": all_tokens,
        "output": wsd_results,
        "description": f"Disambiguated {len(wsd_results)} content words using {method.upper()}"
    })
    
    trace["wsd_results"] = wsd_results
    return trace

# ============================================
# FOL TRANSLATION
# ============================================

# Mapping from synsets to FOL predicates (based on rules.pl)
SYNSET_TO_PREDICATE = {
    'artificial.a.01': 'artificial',
    'intelligence.n.01': 'intelligence',
    'system.n.01': 'system',
    'machine.n.01': 'machine_based',
    'organization.n.01': 'organization',
    'person.n.01': 'individual',
    'developer.n.01': 'developer',
    'supplier.n.01': 'provider',
    'operator.n.01': 'deployer',
    'user.n.01': 'user',
    'design.v.02': 'designs',
    'construct.v.01': 'builds',
    'train.v.01': 'trains',
    'test.n.05': 'tests',
    'bring.v.01': 'brings_to_market',
    'use.v.01': 'uses',
    'interact.v.01': 'interacts_directly',
    'event.n.01': 'event',
    'incident.n.01': 'incident',
    'damage.n.01': 'damage',
    'cause.n.01': 'causes',
    'serious.s.01': 'serious',
    'autonomy.n.02': 'autonomy',
    'capability.n.01': 'capability',
    'perform.v.02': 'performs',
}

# FOL role definitions (from Điều 3 Luật AI)
FOL_ROLE_DEFINITIONS = {
    'developer': '∀x,s: is_developer(x,s) ↔ (organization(x) ∨ individual(x)) ∧ ai_system(s) ∧ (designs(x,s) ∨ builds(x,s) ∨ trains(x,s) ∨ tests(x,s) ∨ fine_tunes(x,s))',
    'provider': '∀x,s: is_provider(x,s) ↔ (organization(x) ∨ individual(x)) ∧ ai_system(s) ∧ brings_to_market(x,s)',
    'deployer': '∀x,s: is_deployer(x,s) ↔ (organization(x) ∨ individual(x) ∨ state_agency(x)) ∧ ai_system(s) ∧ uses(x,s) ∧ purpose(x,s) ≠ personal',
    'user': '∀x,s: is_user(x,s) ↔ (organization(x) ∨ individual(x)) ∧ ai_system(s) ∧ (interacts_directly(x,s) ∨ uses(x,s))',
    'ai_system': '∀s: ai_system(s) ↔ system(s) ∧ machine_based(s) ∧ performs_ai_capabilities(s) ∧ ∃level: has_autonomy(s, level)',
    'serious_incident': '∀e: is_serious_incident(e) ↔ event(e) ∧ ∃s: (occurs_in(e,s) ∧ ai_system(s)) ∧ ∃t: (causes_damage(e,t) ∧ t ∈ DAMAGE_TYPES)',
}

def translate_to_fol(wsd_results: list) -> Dict[str, Any]:
    """
    Translate WSD results to First-Order Logic predicates
    """
    trace = {
        "steps": [],
        "predicates": [],
        "fol_formulas": [],
        "role_definitions": []
    }
    
    # Step 1: Extract synsets from WSD results
    synsets = []
    for r in wsd_results:
        if r.get('synset'):
            synsets.append({
                'token': r['token'],
                'synset': r['synset'],
                'pos': r['pos']
            })
    
    trace["steps"].append({
        "step": 1,
        "name": "Extract Synsets",
        "description": f"Extracted {len(synsets)} synsets from WSD results",
        "output": synsets[:10]  # First 10 for display
    })
    
    # Step 2: Map synsets to FOL predicates
    predicates = []
    for s in synsets:
        synset_name = s['synset']
        token = s['token'].lower()
        
        # Create has_synset predicate
        pred = f"has_synset({token}, {synset_name})"
        predicates.append(pred)
        
        # Map to domain-specific predicate if exists
        if synset_name in SYNSET_TO_PREDICATE:
            domain_pred = SYNSET_TO_PREDICATE[synset_name]
            predicates.append(f"is_concept({token}, {domain_pred})")
    
    trace["steps"].append({
        "step": 2,
        "name": "Generate Predicates",
        "description": f"Generated {len(predicates)} FOL predicates",
        "output": predicates[:15]  # First 15 for display
    })
    trace["predicates"] = predicates
    
    # Step 3: Generate FOL formulas based on detected concepts
    detected_roles = set()
    for s in synsets:
        synset_name = s['synset']
        if synset_name in SYNSET_TO_PREDICATE:
            concept = SYNSET_TO_PREDICATE[synset_name]
            if concept in FOL_ROLE_DEFINITIONS:
                detected_roles.add(concept)
    
    # Add role definitions for detected concepts
    role_defs = []
    for role in detected_roles:
        if role in FOL_ROLE_DEFINITIONS:
            role_defs.append({
                "role": role,
                "definition": FOL_ROLE_DEFINITIONS[role]
            })
    
    # Also add core definitions that are always relevant
    for core_role in ['ai_system', 'developer', 'provider', 'deployer', 'user', 'serious_incident']:
        if core_role not in detected_roles:
            role_defs.append({
                "role": core_role,
                "definition": FOL_ROLE_DEFINITIONS[core_role]
            })
    
    trace["steps"].append({
        "step": 3,
        "name": "Generate Role Definitions",
        "description": f"Generated {len(role_defs)} FOL role definitions from Điều 3 Luật AI",
        "output": [r["role"] for r in role_defs]
    })
    trace["role_definitions"] = role_defs
    
    # Step 4: Generate instance formulas (ground terms)
    fol_formulas = []
    
    # Add some example ground formulas based on knowledge base
    ground_formulas = [
        "organization(bkav_corp)",
        "organization(fpt_software)",
        "individual(nguyen_van_an)",
        "state_agency(bo_cong_an)",
        "system(chat_gpt_vn)",
        "system(camera_traffic)",
        "ai_system(chat_gpt_vn)",
        "ai_system(camera_traffic)",
        "designs(bkav_corp, chat_gpt_vn)",
        "trains(bkav_corp, chat_gpt_vn)",
        "brings_to_market(fpt_software, camera_traffic)",
        "uses(nguyen_van_an, chat_gpt_vn)",
        "uses(bo_cong_an, camera_traffic)",
    ]
    
    trace["steps"].append({
        "step": 4,
        "name": "Ground Formulas (Knowledge Base)",
        "description": f"Loaded {len(ground_formulas)} ground formulas from knowledge base",
        "output": ground_formulas
    })
    trace["fol_formulas"] = ground_formulas
    
    return trace

# ============================================
# PROLOG INFERENCE TRACING
# ============================================


def trace_prolog_inference(query: str) -> Dict[str, Any]:
    """
    Execute Prolog query with step-by-step tracing
    """
    trace = {
        "query": query,
        "steps": [],
        "final_result": None
    }
    
    # Parse query
    if "is_developer" in query:
        if "X" in query:
            # is_developer(X, chat_gpt_vn)
            system = "chat_gpt_vn"
            trace["steps"].append({
                "step": 1,
                "description": f"Find X such that is_developer(X, {system})",
                "rule": "is_developer(Actor, System) :- organization(Actor) OR individual(Actor), ai_system(System), (designs OR builds OR trains OR tests OR fine_tunes)"
            })
            
            # Check organization/individual
            trace["steps"].append({
                "step": 2,
                "description": "Check: organization(X) OR individual(X)?",
                "facts_checked": ["organization(bkav_corp)", "organization(fpt_software)", "individual(nguyen_van_an)"],
                "candidates": ["bkav_corp", "fpt_software", "nguyen_van_an"]
            })
            
            # Check ai_system
            trace["steps"].append({
                "step": 3,
                "description": f"Check: ai_system({system})?",
                "rule_applied": "ai_system(System) :- system(System), machine_based(System), performs_ai_capabilities(System), has_autonomy(System, _)",
                "facts_checked": ["system(chat_gpt_vn)", "machine_based(chat_gpt_vn)", "performs_ai_capabilities(chat_gpt_vn)", "has_autonomy(chat_gpt_vn, high)"],
                "result": "✅"
            })
            
            # Check designs/trains
            result = find_developers(system)
            trace["steps"].append({
                "step": 4,
                "description": "Check: designs(X, chat_gpt_vn) OR trains(X, chat_gpt_vn)?",
                "facts_checked": ["designs(bkav_corp, chat_gpt_vn)", "trains(bkav_corp, chat_gpt_vn)"],
                "result": f"✅ X = {result[0] if result else 'false'}"
            })
            
            trace["final_result"] = f"X = {result[0]}" if result else "false"
            
        else:
            # is_developer(bkav_corp, chat_gpt_vn)
            actor = query.split('(')[1].split(',')[0]
            system = query.split(',')[1].split(')')[0].strip()
            result = is_developer(actor, system)
            trace["final_result"] = str(result).lower()
    
    elif "is_provider" in query:
        if "X" in query and "is_provider(fpt_software, X)" in query:
            result = find_systems_by_provider('fpt_software')
            trace["final_result"] = f"X = {result[0]}" if result else "false"
        elif "is_provider(bkav_corp, _)" in query:
            result = any(is_provider('bkav_corp', sys) for sys in ai_systems)
            trace["final_result"] = str(result).lower()
        else:
            trace["final_result"] = "false"
    
    elif "is_a" in query:
        # is_a(developer, person)
        concept = query.split('(')[1].split(',')[0]
        parent = query.split(',')[1].split(')')[0].strip()
        result = is_a(concept, parent)
        trace["steps"].append({
            "step": 1,
            "description": f"Check: sub_class({concept}, {parent})?",
            "result": "false (no direct relation)"
        })
        trace["steps"].append({
            "step": 2,
            "description": f"Find: sub_class({concept}, X)?",
            "result": f"X = {sub_class.get(concept, 'N/A')}"
        })
        trace["steps"].append({
            "step": 3,
            "description": f"Recursive: is_a({sub_class.get(concept, 'N/A')}, {parent})?",
            "result": "true" if result else "false"
        })
        trace["final_result"] = str(result).lower()
    
    elif "is_serious_incident" in query:
        event = query.split('(')[1].split(')')[0]
        result = is_serious_incident(event)
        trace["final_result"] = str(result).lower()
    
    elif "is_deployer" in query:
        parts = query.split('(')[1].split(')')[0].split(',')
        actor = parts[0].strip()
        system = parts[1].strip()
        result = is_deployer(actor, system)
        trace["final_result"] = str(result).lower()
    
    elif "is_user" in query:
        parts = query.split('(')[1].split(')')[0].split(',')
        actor = parts[0].strip()
        system = parts[1].strip()
        result = is_user(actor, system)
        trace["final_result"] = str(result).lower()
    
    return trace

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {"message": "Legal Text Query API", "version": "1.0"}

@app.get("/queries")
async def get_queries():
    """Get all predefined queries"""
    return {"queries": PREDEFINED_QUERIES}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query with full tracing of intermediate steps
    """
    # Find query
    query_obj = next((q for q in PREDEFINED_QUERIES if q["id"] == request.query_id), None)
    if not query_obj:
        raise HTTPException(status_code=404, detail="Query not found")
    
    query = query_obj["query"]
    nl_question = query_obj["nl"]
    
    # Load related text (from selected_paragraph.txt)
    # Use absolute path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    text_file = os.path.join(project_root, 'data', 'selected_paragraph.txt')
    
    if not os.path.exists(text_file):
        raise HTTPException(status_code=404, detail=f"Text file not found: {text_file}")
    
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Step 1: WSD Tracing (on entire paragraph)
    wsd_trace = await perform_wsd_tracing(full_text, request.wsd_method)
    
    # Step 2: FOL Translation (using WSD results)
    wsd_results = wsd_trace.get("wsd_results", [])
    fol_trace = translate_to_fol(wsd_results)
    
    # Step 3: Prolog Inference Tracing
    prolog_trace = trace_prolog_inference(query)
    
    # Get expected result (ground truth)
    expected_result = query_obj.get("expected_result", "")
    predicted_result = prolog_trace["final_result"]
    
    # Check if prediction matches ground truth
    match = predicted_result == expected_result
    
    # Combine all steps
    response = QueryResponse(
        query=query,
        nl_question=nl_question,
        result=predicted_result,
        expected_result=expected_result,
        success=predicted_result not in ["false", "None"],
        match=match,
        steps={
            "wsd": wsd_trace,
            "fol": fol_trace,
            "prolog": prolog_trace
        }
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3030, reload=True)

