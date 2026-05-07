from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

_vectorizer = None
_tfidf_matrix = None
_jobs = []
_job_texts_lower = []


def clean_description(desc):
    if not desc:
        return desc
    # Strip company intro up to "About the role" / "About this role" heading
    m = re.search(r'About (?:the|this) role\n', desc, re.IGNORECASE)
    if m:
        return desc[m.end():].strip()
    # Fallback: strip leading "About [Company]\n[content]\n\n" block
    m = re.match(r'About \S[^\n]*\n.+?\n\n', desc, re.DOTALL)
    if m:
        return desc[m.end():].strip()
    return desc


def build_candidate_text(candidate):
    parts = [
        candidate.get('headline', ''),
        candidate.get('summary', ''),
        ' '.join(candidate.get('skills', [])),
    ]

    for emp in candidate.get('current_employers', []):
        parts.append(emp.get('employee_title', ''))
        parts.append(emp.get('employee_description', ''))

    # Only first 3 past employers to keep signal tight
    for emp in candidate.get('past_employers', [])[:3]:
        parts.append(emp.get('employee_title', ''))

    for edu in candidate.get('education_background', []):
        parts.append(edu.get('degree_name', ''))
        parts.append(edu.get('field_of_study', ''))
        parts.append(edu.get('institute_name', ''))

    return ' '.join(p for p in parts if p)


def build_job_text(job):
    parts = [
        job.get('title', ''),
        job.get('company', ''),
        clean_description(job.get('description', ''))[:500],
        job.get('location', ''),
        job.get('yc_batch', '') or '',
    ]
    return ' '.join(p for p in parts if p)


def build_tfidf_index(jobs):
    global _vectorizer, _tfidf_matrix, _jobs, _job_texts_lower
    _jobs = jobs
    job_texts = [build_job_text(j) for j in jobs]
    _job_texts_lower = [t.lower() for t in job_texts]
    _vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
    )
    _tfidf_matrix = _vectorizer.fit_transform(job_texts)


def rank_jobs(candidate_text, preferences):
    # Augment query with boost + require terms so TF-IDF retrieval surfaces preference-aligned jobs
    boost_terms = ' '.join(preferences.get('boost', []))
    require_flat = ' '.join(t for group in preferences.get(
        'require', []) for t in (group if isinstance(group, list) else [group]))
    augmented_query = f"{candidate_text} {boost_terms} {require_flat}".strip()
    candidate_vec = _vectorizer.transform([augmented_query])

    similarities = cosine_similarity(candidate_vec, _tfidf_matrix).flatten()

    # Reduce the full job set to top 100 by TF-IDF score before preference scoring
    top_100_indices = np.argsort(similarities)[::-1][:100]

    scored = []
    for idx in top_100_indices:
        job = _jobs[idx]
        job_text = _job_texts_lower[idx]
        score = float(similarities[idx])

        for term in preferences.get('boost', []):
            if term.lower() in job_text:
                score += 0.1

        for term in preferences.get('penalize', []):
            if term.lower() in job_text:
                score -= 0.15

        # Hard requirements — -999 if no term in the group matches
        for group in preferences.get('require', []):
            terms = group if isinstance(group, list) else [group]
            if not any(t.lower() in job_text for t in terms):
                score -= 999

        scored.append((job, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for job, score in scored[:3]:
        job_copy = dict(job)
        job_copy['score'] = round(score, 3)
        # Clean description here so the API response and frontend both get boilerplate-free text
        job_copy['description'] = clean_description(job.get('description', ''))
        results.append(job_copy)

    return results