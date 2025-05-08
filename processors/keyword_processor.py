import re
from collections import Counter
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


class KeywordProcessor:
    """Handles keyword extraction and text preprocessing for BM25"""

    def __init__(self):
        # Basic English stopwords only - don't overfilter
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.min_word_length = 3
        self.max_keywords = 20

    def extract_keywords(self, text: str, use_stemming: bool = False) -> List[str]:
        """
        Extract important keywords from text - simple and robust implementation

        Args:
            text: The input text to extract keywords from
            use_stemming: Whether to apply stemming to keywords (default: False)

        Returns:
            List of extracted keywords
        """
        if not text or len(text.strip()) == 0:
            return []

        try:
            # Basic tokenization - just split on whitespace for robustness
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            # Simple filtering
            filtered_words = []
            for word in words:
                if (len(word) >= self.min_word_length and
                        word not in self.stop_words):
                    filtered_words.append(word)

            # If we still have no keywords, try with fewer filters
            if not filtered_words:
                filtered_words = [word for word in words if len(
                    word) >= self.min_word_length]

            # Count word frequency
            word_counts = Counter(filtered_words)
            keywords = [w for w, _ in word_counts.most_common(
                self.max_keywords)]

            # Apply stemming if requested
            if use_stemming and keywords:
                return [self.stemmer.stem(kw) for kw in keywords]

            return keywords

        except Exception as e:
            # Fallback in case of any error
            print(f"Error extracting keywords: {str(e)}")
            # Last resort - just extract any words with regex
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return words[:self.max_keywords]

    def preprocess_text(self, text: str, use_stemming: bool = False) -> str:
        """
        Preprocess text for BM25 indexing

        Args:
            text: The input text to preprocess
            use_stemming: Whether to apply stemming (default: False)

        Returns:
            Preprocessed text
        """
        try:
            # Simple whitespace tokenization for robustness
            tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

            # Filter out stopwords
            filtered_tokens = [t for t in tokens if t not in self.stop_words]

            # If we filtered everything, just return the original tokens
            if not filtered_tokens:
                filtered_tokens = tokens

            # Apply stemming if requested
            if use_stemming and filtered_tokens:
                processed = [self.stemmer.stem(t) for t in filtered_tokens]
            else:
                processed = filtered_tokens

            return ' '.join(processed)

        except Exception as e:
            print(f"Error preprocessing text: {str(e)}")
            return text.lower()

    def debug_extract_keywords(self, text: str) -> dict:
        """
        Debug version that returns intermediate steps to diagnose issues

        Args:
            text: Input text

        Returns:
            Dictionary with diagnostic information
        """
        results = {"original_text": text}

        # Step 1: Basic tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        results["all_words"] = words
        results["word_count"] = len(words)

        # Step 2: Filter by length
        length_filtered = [w for w in words if len(w) >= self.min_word_length]
        results["length_filtered"] = length_filtered
        results["length_filtered_count"] = len(length_filtered)

        # Step 3: Filter stopwords
        stop_filtered = [
            w for w in length_filtered if w not in self.stop_words]
        results["stopword_filtered"] = stop_filtered
        results["stopword_filtered_count"] = len(stop_filtered)

        # Step 4: Count frequencies
        word_counts = Counter(stop_filtered)
        results["word_frequencies"] = dict(word_counts)

        # Final keywords
        keywords = [w for w, _ in word_counts.most_common(self.max_keywords)]
        results["final_keywords"] = keywords
        results["final_count"] = len(keywords)

        return results
