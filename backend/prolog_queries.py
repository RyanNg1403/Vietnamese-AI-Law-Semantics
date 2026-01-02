# prolog_queries.py - Python-based Knowledge Query System
"""
Simulate Prolog queries using Python.
Based on knowledge_base.pl and rules.pl
"""

# ============================================
# KNOWLEDGE BASE (from knowledge_base.pl)
# ============================================

# Basic synset mappings
has_synset = {
    'artificial': 'artificial.a.01',
    'autonomy': 'autonomy.n.02',
    'be': 'be.v.02',
    'bring': 'bring.v.01',
    'build': 'construct.v.01',
    'capability': 'capability.n.01',
    'cause': 'cause.n.01',
    'damage': 'damage.n.01',
    'design': 'design.v.02',
    'developer': 'developer.n.01',
    'deployer': 'operator.n.01',
    'electronic': 'electronic.a.01',
    'event': 'event.n.01',
    'fine_tune': 'fine-tune.v.02',
    'human': 'human.a.01',
    'implementation': 'implementation.n.02',
    'incident': 'incident.n.01',
    'individual': 'person.n.01',
    'intelligence': 'intelligence.n.01',
    'interact': 'interact.v.01',
    'level': 'degree.n.01',
    'life': 'life.n.01',
    'machine_based': 'machine.n.01',
    'market': 'market.n.01',
    'occur': 'happen.v.01',
    'operation': 'operation.n.01',
    'organization': 'organization.n.01',
    'perform': 'perform.v.02',
    'provider': 'supplier.n.01',
    'serious': 'serious.s.01',
    'significant': 'significant.a.01',
    'system': 'system.n.01',
    'test': 'test.n.05',
    'train': 'train.v.01',
    'user': 'user.n.01',
    'use': 'use.v.01',
    'affect': 'affect.v.01',
    'health': 'health.n.01',
    'property': 'property.n.01',
    'environment': 'environment.n.01',
}

is_concept = set(has_synset.keys())

# WordNet Hypernyms (sub_class relationships)
sub_class = {
    'autonomy': 'independence',
    'bring': 'transport',
    'build': 'make',
    'capability': 'ability',
    'cause': 'origin',
    'damage': 'change',
    'design': 'intend',
    'developer': 'creator',
    'deployer': 'person',
    'event': 'psychological_feature',
    'fine_tune': 'tune',
    'implementation': 'act',
    'incident': 'happening',
    'individual': 'organism',
    'intelligence': 'ability',
    'interact': 'act',
    'level': 'property',
    'life': 'being',
    'machine_based': 'device',
    'market': 'activity',
    'operation': 'action',
    'organization': 'social_group',
    'perform': 'act',
    'provider': 'businessperson',
    'system': 'instrumentality',
    'test': 'attempt',
    'train': 'teach',
    'user': 'person',
    'creator': 'person',  # Extended chain
    'businessperson': 'person',
}

# ============================================
# MOCK DATA (from rules.pl)
# ============================================

organizations = {'bkav_corp', 'fpt_software'}
individuals = {'nguyen_van_an', 'le_thi_b'}
state_agencies = {'bo_cong_an'}

systems = {'chat_gpt_vn', 'camera_traffic', 'excel_macro'}
ai_systems = {'chat_gpt_vn', 'camera_traffic'}

# Relations
designs = {('bkav_corp', 'chat_gpt_vn')}
builds = set()
trains = {('bkav_corp', 'chat_gpt_vn')}
tests = set()
fine_tunes = set()
brings_to_market = {('fpt_software', 'camera_traffic')}
uses = {('nguyen_van_an', 'chat_gpt_vn'), ('bo_cong_an', 'camera_traffic')}
purposes = {
    ('nguyen_van_an', 'chat_gpt_vn'): 'personal',
    ('bo_cong_an', 'camera_traffic'): 'professional',
}
interacts_directly = {('nguyen_van_an', 'chat_gpt_vn')}

events = {'incident_01'}
occurs_in = {('incident_01', 'camera_traffic')}
causes_damage = {('incident_01', 'reputation')}

DAMAGE_TYPES = {'human_life', 'health', 'property', 'reputation', 'national_security', 'environment'}

# ============================================
# INFERENCE RULES
# ============================================

def is_developer(actor, system):
    """Developer = (org OR individual) + ai_system + (designs OR builds OR trains OR tests OR fine_tunes)"""
    if actor not in organizations and actor not in individuals:
        return False
    if system not in ai_systems:
        return False
    if (actor, system) in designs:
        return True
    if (actor, system) in builds:
        return True
    if (actor, system) in trains:
        return True
    if (actor, system) in tests:
        return True
    if (actor, system) in fine_tunes:
        return True
    return False

def is_provider(actor, system):
    """Provider = (org OR individual) + ai_system + brings_to_market"""
    if actor not in organizations and actor not in individuals:
        return False
    if system not in ai_systems:
        return False
    return (actor, system) in brings_to_market

def is_deployer(actor, system):
    """Deployer = (org OR individual OR state_agency) + ai_system + uses + professional purpose"""
    if actor not in organizations and actor not in individuals and actor not in state_agencies:
        return False
    if system not in ai_systems:
        return False
    if (actor, system) not in uses:
        return False
    purpose = purposes.get((actor, system))
    return purpose != 'personal'

def is_user(actor, system):
    """User = (org OR individual) + ai_system + (interacts_directly OR uses)"""
    if actor not in organizations and actor not in individuals:
        return False
    if system not in ai_systems:
        return False
    return (actor, system) in interacts_directly or (actor, system) in uses

def is_serious_incident(event):
    """Serious incident = event + occurs_in AI system + causes significant damage"""
    if event not in events:
        return False
    for (e, sys) in occurs_in:
        if e == event and sys in ai_systems:
            for (ev, dtype) in causes_damage:
                if ev == event and dtype in DAMAGE_TYPES:
                    return True
    return False

def is_a(concept, parent):
    """Transitive is-a relationship via WordNet hypernyms"""
    if concept == parent:
        return True
    if concept in sub_class:
        direct_parent = sub_class[concept]
        if direct_parent == parent:
            return True
        return is_a(direct_parent, parent)
    return False

def find_developers(system):
    """Find all developers for a system"""
    results = []
    for actor in organizations | individuals:
        if is_developer(actor, system):
            results.append(actor)
    return results

def find_providers(system):
    """Find all providers for a system"""
    results = []
    for actor in organizations | individuals:
        if is_provider(actor, system):
            results.append(actor)
    return results

def find_systems_by_provider(actor):
    """Find systems provided by an actor"""
    results = []
    for sys in ai_systems:
        if is_provider(actor, sys):
            results.append(sys)
    return results

# ============================================
# RUN QUERIES
# ============================================

def run_all_queries():
    print("=" * 70)
    print("PROLOG-LIKE KNOWLEDGE QUERY RESULTS")
    print("=" * 70)
    
    queries = []
    
    # Query 1: Who is developer of chat_gpt_vn?
    print("\n📌 Query 1: is_developer(X, chat_gpt_vn)")
    result = find_developers('chat_gpt_vn')
    print(f"   Result: X = {result}")
    queries.append(("is_developer(X, chat_gpt_vn)", f"X = {result[0]}" if result else "false"))
    
    # Query 2: Is BKAV a provider?
    print("\n📌 Query 2: is_provider(bkav_corp, _)")
    result = any(is_provider('bkav_corp', sys) for sys in ai_systems)
    print(f"   Result: {result}")
    queries.append(("is_provider(bkav_corp, _)", str(result).lower()))
    
    # Query 3: What systems does FPT provide?
    print("\n📌 Query 3: is_provider(fpt_software, X)")
    result = find_systems_by_provider('fpt_software')
    print(f"   Result: X = {result}")
    queries.append(("is_provider(fpt_software, X)", f"X = {result[0]}" if result else "false"))
    
    # Query 4: Is developer a kind of person? (WordNet inference)
    print("\n📌 Query 4: is_a(developer, person)")
    result = is_a('developer', 'person')
    print(f"   Result: {result}")
    print(f"   Explanation: developer → creator → person (hypernym chain)")
    queries.append(("is_a(developer, person)", str(result).lower()))
    
    # Query 5: Is incident_01 a serious incident?
    print("\n📌 Query 5: is_serious_incident(incident_01)")
    result = is_serious_incident('incident_01')
    print(f"   Result: {result}")
    print(f"   Explanation: incident_01 causes 'reputation' damage ∈ DAMAGE_TYPES")
    queries.append(("is_serious_incident(incident_01)", str(result).lower()))
    
    # Query 6: Is bo_cong_an a deployer of camera_traffic?
    print("\n📌 Query 6: is_deployer(bo_cong_an, camera_traffic)")
    result = is_deployer('bo_cong_an', 'camera_traffic')
    print(f"   Result: {result}")
    print(f"   Explanation: bo_cong_an uses camera_traffic for 'professional' purpose")
    queries.append(("is_deployer(bo_cong_an, camera_traffic)", str(result).lower()))
    
    # Query 7: Is nguyen_van_an a user of chat_gpt_vn?
    print("\n📌 Query 7: is_user(nguyen_van_an, chat_gpt_vn)")
    result = is_user('nguyen_van_an', 'chat_gpt_vn')
    print(f"   Result: {result}")
    queries.append(("is_user(nguyen_van_an, chat_gpt_vn)", str(result).lower()))
    
    # Query 8: Is user a kind of person?
    print("\n📌 Query 8: is_a(user, person)")
    result = is_a('user', 'person')
    print(f"   Result: {result}")
    queries.append(("is_a(user, person)", str(result).lower()))
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Query':<45} {'Result':<20}")
    print("-" * 65)
    for q, r in queries:
        print(f"{q:<45} {r:<20}")
    
    # Save results
    import csv
    with open('./data/prolog_query_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Query', 'Result'])
        writer.writerows(queries)
    print(f"\n💾 Results saved to './data/prolog_query_results.csv'")
    
    return queries

if __name__ == "__main__":
    run_all_queries()

