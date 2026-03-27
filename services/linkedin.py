"""
LinkedIn post generator — turns a research digest into a viral
LinkedIn-bro thought-leadership post via Ollama.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import httpx

from shared.types import DigestResult, ScoredPaperRecord

_OLLAMA_TIMEOUT = 120.0


async def _ollama_generate(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    system: str,
    prompt: str,
) -> str:
    url = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }
    resp = await client.post(url, json=payload, timeout=_OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


def _default_persona() -> str:
    return (
        "You are a LinkedIn thought leader and influencer. "
        "You turn cutting-edge research into business wisdom. "
        "You are humble yet confident, always learning, and everything "
        "you read somehow makes you a better leader and entrepreneur."
    )


_SYSTEM_PROMPT = """\
You write viral LinkedIn posts. You turn scientific research into \
inspiring business and leadership lessons. You write in plain text \
only (no markdown, no bold, no italics, no bullet points). \
You are concise: every post is under 1300 characters.\
"""

_GOOD_EXAMPLE = """\
I used to think great teams were built on talent.

I was satisfyingly wrong.

This week I discovered a study on how friendship networks actually form. \
Turns out, who you already know matters 400x more than who you are.

Your personality? Almost irrelevant.
Your network structure? Everything.

This hit me hard as a leader.

We spend so much time screening for "culture fit" in hiring. \
But the research says the real magic is in how you connect people \
AFTER they join.

One more finding blew my mind: ants and random forests use \
the exact same strategy to make smart group decisions. \
Decorrelation. Independence. Diverse viewpoints.

The best teams are not full of people who agree. \
They are full of people who think differently and still coordinate.

Nature figured out ensemble intelligence millions of years ago. \
We are just catching up.

What is one thing you do to create real cognitive diversity on your team?

#Leadership #TeamBuilding #Research #NeuralNetworks #CollectiveIntelligence\
"""

_BAD_EXAMPLE = """\
**Is your team's communication as efficient as it could be?**

I was blown away by research that shows how EEG-based machine learning can \
enhance brain network connectivity during complex tasks like driving. This \
means we might soon see smarter interaction design for collective behavior \
in shared spaces. #InnovationInBrainTech

Understanding friendship formation is crucial for building strong teams. \
I discovered an explainable ML model that revealed local network structure \
significantly drives social ties, with individual traits playing a minor \
role. #SocialNeuroscience

#NeuralMechanisms #SocialInteraction #TeamPerformance\
"""

_USER_PROMPT_TEMPLATE = """\
Write a LinkedIn post based on these research papers.

PAPERS:
{paper_block}

STRUCTURE (follow this pattern closely):

Line 1: A bold, contrarian statement that challenges common belief. Short.

Line 2: An even shorter follow-up that creates tension or surprise.

Lines 3-6: Describe the most interesting finding from ONE paper. \
Use plain language. Translate the science into a lesson about \
business, teams, leadership, or personal growth. Sound amazed. \
Say "I discovered" or "this week I learned" instead of "I read a paper."

Lines 7-10: Bring in a second finding from another paper. \
Connect it to the first one thematically. Build a narrative arc. \
Make it feel like a journey of insight, not a list.

Optional lines 11-13: If a third paper fits the narrative, weave it in \
briefly. Otherwise skip it. Never force a paper in.

Final 2 lines: A punchy question to invite comments. Then 3-5 hashtags \
on their own line.

GOOD EXAMPLE (study this style, rhythm, and tone):

{good_example}

BAD EXAMPLE (avoid this style — it is too academic, uses markdown bold, \
puts hashtags inline, and reads like a paper summary not a story):

{bad_example}

RULES:
- Plain text only. No asterisks, no bold, no markdown, no formatting.
- One sentence per line. Blank line between each sentence.
- Under 1300 characters total.
- Sound like a founder sharing a personal revelation, not a professor.
- Emojis: use 0-2 maximum. Only at natural emphasis points.
- All hashtags go on the very last line, together.
- Weave the papers into ONE narrative. This is a story, not a list.

Now write the post. Plain text, under 1300 characters:\
"""


def _select_top_papers(
    scored_papers: list[ScoredPaperRecord],
    max_papers: int,
) -> list[ScoredPaperRecord]:
    included = [sp for sp in scored_papers if sp.included_in_digest and sp.summary]
    included.sort(key=lambda sp: sp.final_score, reverse=True)
    return included[:max_papers]


def _build_paper_block(papers: list[ScoredPaperRecord]) -> str:
    lines = []
    for i, sp in enumerate(papers, 1):
        p = sp.paper
        authors_short = ", ".join(a.name for a in p.authors[:3])
        lines.append(f"Paper {i}: \"{p.title}\" by {authors_short}")
        if sp.summary:
            lines.append(f"  Finding: {sp.summary}")
        lines.append("")
    return "\n".join(lines)


def _build_prompt(
    papers: list[ScoredPaperRecord],
    window_start: date,
    window_end: date,
) -> str:
    paper_block = _build_paper_block(papers)
    return _USER_PROMPT_TEMPLATE.format(
        paper_block=paper_block,
        good_example=_GOOD_EXAMPLE,
        bad_example=_BAD_EXAMPLE,
    )


async def build_linkedin_post(
    digest_result: DigestResult,
    research_profile: str,
    ollama_model: str,
    ollama_host: str,
    linkedin_config: dict,
) -> str:
    """Generate a LinkedIn post from the digest and save it to a file.

    Returns the post text.
    """
    max_papers = linkedin_config.get("max_papers", 3)
    persona = linkedin_config.get("persona", "").strip() or _default_persona()

    top_papers = _select_top_papers(digest_result.scored_papers, max_papers)
    if not top_papers:
        return ""

    system = f"{persona}\n\n{_SYSTEM_PROMPT}"
    prompt = _build_prompt(
        top_papers,
        digest_result.window_start,
        digest_result.window_end,
    )

    async with httpx.AsyncClient() as client:
        post_text = await _ollama_generate(
            client, ollama_host, ollama_model, system, prompt,
        )

    post_text = post_text.strip()

    digests_dir = Path("digests")
    digests_dir.mkdir(exist_ok=True)
    filename = f"linkedin_{digest_result.window_end.strftime('%Y-%m-%d')}.txt"
    filepath = digests_dir / filename
    filepath.write_text(post_text, encoding="utf-8")

    return post_text
