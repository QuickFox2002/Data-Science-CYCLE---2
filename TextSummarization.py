# task2_text_summarization.py
from datasets import load_dataset
from transformers import pipeline
from rouge_score import rouge_scorer

def main():
    # Load a small portion of the CNN/DailyMail dataset for demonstration
    print("Loading dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:2]")  # just 2 samples

    # Load the pretrained summarization pipeline (BART large CNN)
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i, sample in enumerate(dataset):
        article = sample["article"]
        reference = sample["highlights"]

        # Generate summary
        summary = summarizer(article, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

        print(f"\n=== Article {i+1} ===")
        print("Original Article (first 300 chars):")
        print(article[:300] + "...\n")

        print("Generated Summary:")
        print(summary + "\n")

        print("Reference Summary:")
        print(reference + "\n")

        # Calculate ROUGE scores
        scores = scorer.score(reference, summary)
        print("ROUGE Scores:")
        for key, score in scores.items():
            print(f"  {key}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")

if __name__ == "__main__":
    main()
