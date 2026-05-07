import json
import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def parse_feedback(feedback_text, current_preferences):
    client = _get_client()

    prompt = f"""You are a job preference parser. Given a candidate's feedback about job recommendations, extract structured preferences to re-rank jobs.

Current preferences (MERGE with these, do not discard them unless the new feedback contradicts them):
{json.dumps(current_preferences, indent=2)}

New feedback: "{feedback_text}"

For boost and penalize, include synonyms, abbreviations, and geographic neighbors — not just the literal words:
- 'machine learning' → ['machine learning', 'ml', 'deep learning', 'ai', 'llm']
- 'backend' → ['backend', 'back-end', 'back end', 'server-side', 'infrastructure', 'systems', 'api', 'distributed systems']
- 'early stage' → ['early stage', 'seed', 'series a', 'pre-seed', 'yc', 'early-stage', 'startup', 'founding']
- 'enterprise' → ['enterprise', 'fortune 500', 'fortune500', 'large company', 'corporate', 'big tech']
- 'bay area' → ['bay area', 'san francisco', 'sf', 'oakland', 'berkeley', 'san jose', 'south bay', 'east bay', 'peninsula', 'soma']
- 'los angeles' / 'la' → ['los angeles', 'la', 'santa monica', 'culver city', 'west hollywood', 'venice', 'playa vista', 'el segundo']
- 'new york' → ['new york', 'nyc', 'ny', 'manhattan', 'brooklyn', 'new york city']
- 'no internships' → penalize: ['intern', 'internship']

For require, use synonym groups (lists of equivalent terms) — a job passes if it matches ANY term in the group:
- 'remote' → [['remote', 'remote ok', 'distributed', 'work from home', 'wfh']]
- 'new york' → [['new york', 'nyc', 'ny', 'manhattan', 'brooklyn', 'new york city']]
- 'must be full-time' → [['full-time', 'full time']]

Return ONLY valid JSON, no markdown fences:
{{
  "boost": ["term1", "term2", "term3"],
  "penalize": ["term1", "term2"],
  "require": [["term1a", "term1b"], ["term2a"]],
  "explanation": "One sentence explaining what changed and why"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        text = text.rsplit('```', 1)[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {
            **current_preferences,
            'explanation': 'Could not parse feedback response, preferences unchanged'
        }
