# mfs_baseline.py - Most Frequent Sense Baseline
"""
Baseline MFS: Luôn chọn nghĩa phổ biến nhất từ WordNet cho mỗi từ.
Dùng làm baseline để so sánh với BERT model.
"""
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Download required data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

def get_mfs(word, pos_tag):
    """
    Get Most Frequent Sense from WordNet.
    WordNet synsets are ordered by frequency (first = most frequent).
    """
    # Map POS tags to WordNet POS
    pos_map = {
        'N': wn.NOUN,
        'V': wn.VERB,
        'J': wn.ADJ,
        'R': wn.ADV
    }
    
    wn_pos = None
    for key, val in pos_map.items():
        if pos_tag.startswith(key):
            wn_pos = val
            break
    
    # Get synsets (ordered by frequency)
    if wn_pos:
        synsets = wn.synsets(word, pos=wn_pos)
    else:
        synsets = wn.synsets(word)
    
    # Return first synset (most frequent) or None
    if synsets:
        return synsets[0].name()
    return None

def evaluate_mfs_baseline(test_file='./data/gold_standard_completed.csv'):
    """
    Evaluate MFS baseline on test set.
    """
    print("=" * 60)
    print("MOST FREQUENT SENSE (MFS) BASELINE EVALUATION")
    print("=" * 60)
    
    # Load test data
    df = pd.read_csv(test_file)
    df = df.dropna(subset=['Selected_Synset'])
    
    y_true = []
    y_pred = []
    results = []
    
    for idx, row in df.iterrows():
        token = row['Token']
        pos = row['POS']
        gold_synset = row['Selected_Synset']
        
        # Get MFS prediction
        pred_synset = get_mfs(token, pos)
        if pred_synset is None:
            pred_synset = "O"  # No synset found
        
        y_true.append(gold_synset)
        y_pred.append(pred_synset)
        
        results.append({
            'Token': token,
            'POS': pos,
            'Gold': gold_synset,
            'MFS_Pred': pred_synset,
            'Match': gold_synset == pred_synset
        })
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print results
    print(f"\n📊 Results on {len(y_true)} tokens:")
    print("-" * 40)
    print(f"Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro):    {recall:.4f}")
    print(f"F1-Macro:          {f1_macro:.4f}")
    print(f"F1-Weighted:       {f1_weighted:.4f}")
    
    # Count correct predictions
    correct = sum(1 for r in results if r['Match'])
    print(f"\nCorrect: {correct}/{len(results)}")
    
    # Show some examples
    print("\n📝 Sample Predictions:")
    print("-" * 60)
    for r in results[:10]:
        match = "✅" if r['Match'] else "❌"
        print(f"{match} {r['Token']:15} | Gold: {r['Gold']:20} | MFS: {r['MFS_Pred']}")
    
    # Save results
    df_result = pd.DataFrame(results)
    output_path = './data/mfs_baseline_result.csv'
    df_result.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to '{output_path}'")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

if __name__ == "__main__":
    evaluate_mfs_baseline()

