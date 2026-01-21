def compression_ratio(input_text, output_text):
    """Compute ratio of summary length to original length"""
    return len(output_text) / len(input_text)

def token_count(text):
    """Count number of words (tokens) in text"""
    return len(text.split())

def summary_stats(input_text, summary_text):
    """Return basic evaluation metrics for summaries"""
    return {
        "input_tokens": token_count(input_text),
        "summary_tokens": token_count(summary_text),
        "compression_ratio": compression_ratio(input_text, summary_text)
    }
