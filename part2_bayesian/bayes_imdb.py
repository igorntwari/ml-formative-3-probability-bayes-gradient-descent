import csv
import re

POSITIVE_KEYWORDS = ["excellent", "wonderful", "superb", "amazing"]
NEGATIVE_KEYWORDS = ["worst", "awful", "terrible", "waste"]


def load_reviews(path, text_col="review", label_col="sentiment"):
    """Return a list of (set_of_words, label) tuples."""
    csv.field_size_limit(10_000_000)
    reviews = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews.append((tokenize(row[text_col]), row[label_col].strip().lower()))
    return reviews


def tokenize(text):
    """Lowercase, strip HTML tags, return the set of distinct words."""
    text = re.sub(r"<[^>]+>", " ", text.lower())
    return set(re.findall(r"[a-z']+", text))


def count_reviews(reviews, keyword):
    """Return (reviews containing keyword, positive reviews containing keyword)."""
    in_all = sum(1 for words, _ in reviews if keyword in words)
    in_pos = sum(1 for words, label in reviews if keyword in words and label == "positive")
    return in_all, in_pos


def bayes_posterior(prior, likelihood, marginal):
    """P(Positive | keyword) = P(keyword | Positive) * P(Positive) / P(keyword)"""
    return likelihood * prior / marginal if marginal > 0 else 0.0


def analyze_keyword(reviews, keyword, n_total, n_pos):
    in_all, in_pos = count_reviews(reviews, keyword)
    prior = n_pos / n_total                  # P(Positive)
    likelihood = in_pos / n_pos              # P(keyword | Positive)
    marginal = in_all / n_total              # P(keyword)
    posterior = bayes_posterior(prior, likelihood, marginal)
    return {"keyword": keyword, "prior": prior, "likelihood": likelihood,
            "marginal": marginal, "posterior": posterior}


def print_table(results):
    print("| Keyword | P(Positive) | P(kw\\|Positive) | P(kw) | P(Positive\\|kw) |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['keyword']} | {r['prior']:.4f} | {r['likelihood']:.4f} "
              f"| {r['marginal']:.4f} | {r['posterior']:.4f} |")

def direct_posterior(reviews, keyword):
    """Check: positive reviews containing keyword / all reviews containing keyword."""
    in_all, in_pos = count_reviews(reviews, keyword)
    return in_pos / in_all if in_all else 0.0

if __name__ == "__main__":
    reviews = load_reviews("IMDB Dataset.csv")
    n_total = len(reviews)
    n_pos = sum(1 for _, label in reviews if label == "positive")

    keywords = POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS
    results = [analyze_keyword(reviews, k, n_total, n_pos) for k in keywords]
    print_table(results)