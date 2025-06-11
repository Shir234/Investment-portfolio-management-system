# PCA_analysis.py
import re
from collections import Counter
import statistics

def analyze_pca_results(log_text):
    """
    Analyze PCA results from log text to extract statistics about 
    PCA percentages and component counts.
    
    Parameters:
    log_text (str): The complete log text containing PCA results
    
    Returns:
    dict: Statistics about PCA usage
    """
    
    # Pattern to match "Best feature set: PCA X% with Y components"
    pattern = r"Best feature set: PCA (\d+)% with (\d+) components"
    
    # Find all matches
    matches = re.findall(pattern, log_text)
    
    if not matches:
        return {"error": "No PCA results found in log"}
    
    # Extract percentages and component counts
    percentages = [int(match[0]) for match in matches]
    components = [int(match[1]) for match in matches]
    
    # Calculate statistics
    percentage_counts = Counter(percentages)
    component_counts = Counter(components)
    
    results = {
        "total_tickers": len(matches),
        "percentage_stats": {
            "most_common": percentage_counts.most_common(1)[0],
            "least_common": percentage_counts.most_common()[-1],
            "distribution": dict(percentage_counts),
            "usage_rates": {pct: f"{count/len(percentages)*100:.1f}%" 
                           for pct, count in percentage_counts.items()}
        },
        "component_stats": {
            "average": round(statistics.mean(components), 1),
            "median": statistics.median(components),
            "min": min(components),
            "max": max(components),
            "most_common": component_counts.most_common(1)[0],
            "distribution": dict(component_counts)
        },
        "detailed_results": list(zip(percentages, components))
    }
    
    return results

def print_analysis_summary(results):
    """Print a formatted summary of the PCA analysis"""
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print("=== PCA ANALYSIS SUMMARY ===")
    print(f"Total tickers analyzed: {results['total_tickers']}")
    
    print("\n--- PCA PERCENTAGE USAGE ---")
    pct_stats = results['percentage_stats']
    print(f"Most used: {pct_stats['most_common'][0]}% ({pct_stats['most_common'][1]} times)")
    print(f"Least used: {pct_stats['least_common'][0]}% ({pct_stats['least_common'][1]} times)")
    print("Usage distribution:")
    for pct in sorted(pct_stats['distribution'].keys()):
        count = pct_stats['distribution'][pct]
        rate = pct_stats['usage_rates'][pct]
        print(f"  PCA {pct}%: {count} times ({rate})")
    
    print("\n--- COMPONENT COUNT STATISTICS ---")
    comp_stats = results['component_stats']
    print(f"Average components: {comp_stats['average']}")
    print(f"Median components: {comp_stats['median']}")
    print(f"Range: {comp_stats['min']} - {comp_stats['max']} components")
    print(f"Most common count: {comp_stats['most_common'][0]} components ({comp_stats['most_common'][1]} times)")
    
    return results

# Example usage:
# Replace 'your_log_text' with your actual log content
if __name__ == "__main__":
    # Example log text (replace with your actual log)
    sample_log = """
    Best feature set: PCA 97% with 15 components
    Best feature set: PCA 95% with 12 components
    Best feature set: PCA 97% with 18 components
    Best feature set: PCA 90% with 8 components
    """
    
    # To use with your actual log file:
    with open('stock_pipeline_20250527.txt', 'r') as f:
        log_content = f.read()
    results = analyze_pca_results(log_content)
    
    #results = analyze_pca_results(sample_log)
    print_analysis_summary(results)
    
    # For your engineering report, you can extract specific values:
    print(f"\nFor final report:")
    print(f"- Most used PCA threshold: {results['percentage_stats']['most_common'][0]}%")
    print(f"- Average number of components: {results['component_stats']['average']}")