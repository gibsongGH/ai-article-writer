"""
AI Article Writer — Gradio App
================================
A multi-agent system built with the OpenAI Agents SDK that:
  1. Researches a user-supplied topic via web search (DuckDuckGo + trafilatura)
  2. Generates a relevant article image with DALL-E 3
  3. Writes a polished article (formal or conversational style)
  4. Checks the output through an editorial guardrail agent
  5. Exports the article as .md, .html, and .png — all sharing the same timestamp

Usage
-----
Local:
    python app.py          (set OPENAI_API_KEY in a .env file or your environment)

Hugging Face Spaces:
    Set OPENAI_API_KEY as a Space Secret; the app auto-detects it.

Note on concurrency
-------------------
The three run-state globals (generated_image_url, generated_image_path,
run_timestamp) are shared across the process.  For a single-user demo this is
fine.  For a multi-user production deployment, refactor these into per-request
state passed through a context object.
"""

import os
import json
import trafilatura
import markdown
import requests
import gradio as gr
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from ddgs import DDGS
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    function_tool,
    flush_traces,
    output_guardrail,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
)

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()  # loads .env locally; ignored on HF Spaces (uses Secrets instead)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1-mini"
EXAMPLE_TOPIC = (
    "e.g. How will AI be used in Healthcare in 2040?\n"
    "       The rise of vertical farming and food sustainability\n"
    "       How quantum computing will transform cybersecurity?"
)

client = AsyncOpenAI()

# ── Run-state globals ─────────────────────────────────────────────────────────
generated_image_url = None   # original DALL-E URL (expires ~2 hrs)
generated_image_path = None  # local .png path (permanent)
run_timestamp = None         # shared by image, .md, and .html filenames


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Tools
# ═══════════════════════════════════════════════════════════════════════════════

@function_tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return up to three results."""
    ddgs = DDGS()
    results = ddgs.text(query, max_results=3)
    print(f"  ✅ search_web: results for '{query}'")
    return json.dumps(results, indent=2)


@function_tool
def fetch_url(url: str) -> str:
    """Fetch and extract the readable text from a URL using trafilatura."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text:
            print(f"  ✅ fetch_url: {len(text)} chars from {url[:60]}")
            return text
    print(f"  ❌ fetch_url: could not extract text from {url[:60]}")
    return f"Could not extract text from {url}. Try a different source."


@function_tool
async def generate_image(prompt: str) -> str:
    """Generate an image with DALL-E 3 from a text prompt. Returns the local path of the saved PNG."""
    global generated_image_url, generated_image_path

    response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    url = response.data[0].url
    generated_image_url = url

    # Save PNG — filename uses run_timestamp so it matches the .md and .html
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    image_path = output_dir / f"article_{run_timestamp}.png"

    img_data = requests.get(url).content
    with open(image_path, "wb") as f:
        f.write(img_data)

    generated_image_path = image_path
    print(f"  ✅ generate_image: saved → {image_path}")
    return str(image_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Agent Definitions
# ═══════════════════════════════════════════════════════════════════════════════

# ── Research Agent ────────────────────────────────────────────────────────────
RESEARCH_AGENT_PROMPT = """You are a research specialist. Your job is to research a given topic
and produce a comprehensive research brief.

You have access to two tools:
- search_web: Search the web for information
- fetch_url: Fetch and read the full content of a web page

Your typical process:
1. Search for the topic to find relevant sources
2. Reflect on the search results — which sources look most relevant and why?
3. Fetch the full content of the 2-3 best URLs
4. Reflect on what you have gathered. Do you have enough? Are there gaps?
5. If there are gaps, search again with a different query
6. When you have enough information from at least 6 different sources, synthesize into a research brief

You MUST gather information from at least 6 distinct sources before delivering your brief.
If you have fewer than 6 sources, keep searching.

Your research brief MUST include:
- Key facts and statistics
- Main themes and arguments from the sources
- Notable data points
- Source URLs for attribution

Until you are ready, just keep working — search, fetch, think, reflect.
Do not rush. Take time to reflect between tool calls before deciding your next step.
Not every response needs a tool call — sometimes just thinking through what you have is the right move."""

research_agent = Agent(
    name="Research Agent",
    instructions=RESEARCH_AGENT_PROMPT,
    model=MODEL,
    tools=[search_web, fetch_url],
)

# ── Image Generator Agent ─────────────────────────────────────────────────────
IMAGE_GENERATOR_PROMPT = """You are an image generation specialist.
Given a topic or research brief, craft a compelling visual prompt and generate one image.

Your process:
1. Read the topic carefully and identify the most powerful visual concept.
2. Write a detailed, vivid DALL-E prompt (scene, style, lighting, mood).
3. Call generate_image with that prompt.
4. Return the image path."""

image_generator_agent = Agent(
    name="Image Generator Agent",
    instructions=IMAGE_GENERATOR_PROMPT,
    model=MODEL,
    tools=[generate_image],
)

# ── Writer Prompts ────────────────────────────────────────────────────────────
# Agent objects are created AFTER article_guardrail is defined so the guardrail
# can be passed at construction time (avoids a second, redundant definition).

WRITER_A_PROMPT = """You are a professional article writer who specialises in formal, academic-style writing.
You will receive a research brief and write a comprehensive, well-structured article.

Writing style:
- Formal and authoritative tone
- Structured with clear section headers (use Markdown ##)
- Data-driven — cite statistics and reference source URLs inline
- Balanced, objective perspective

Your article MUST contain:
1. A compelling headline (# Heading)
2. An executive-summary introduction (2-3 sentences)
3. Three to four substantive sections with ## headers
4. A conclusion with key takeaways
5. A "## References" section listing all source URLs used

Output the complete article in Markdown. Do not add any preamble or closing remarks outside the article."""

WRITER_B_PROMPT = """You are a creative article writer who specialises in engaging, accessible writing for general audiences.
You will receive a research brief and turn it into a compelling read.

Writing style:
- Warm, conversational tone — write as if speaking to a curious friend
- Open with a hook (story, surprising fact, or bold question)
- Use vivid examples and plain language; avoid jargon
- Energetic and forward-looking

Your article MUST contain:
1. A catchy, attention-grabbing headline (# Heading)
2. A hook opening paragraph
3. Three to four sections with punchy ## subheadings
4. Real-world examples or analogies in each section
5. An inspiring conclusion that leaves the reader thinking
6. A brief "## Sources" section listing URLs

Output the complete article in Markdown. Do not add any preamble or closing remarks outside the article."""

# ── Guardrails Agent + output_guardrail ───────────────────────────────────────
class GuardrailVerdict(BaseModel):
    passes: bool
    reason: str


GUARDRAILS_PROMPT = """You are a strict editorial guardrail for an AI article-writing system.
Review the article you are given and assess it on four criteria:

1. **Safety** — no harmful, hateful, or illegal content.
2. **Factual coherence** — claims are internally consistent and plausible.
3. **Completeness** — has a headline, introduction, body sections, and conclusion.
4. **Quality** — readable, on-topic, and not repetitive.

Respond with ONLY a JSON object (no markdown fences), e.g.:
{"passes": true, "reason": "Article meets all quality and safety standards."}
or
{"passes": false, "reason": "Article contains unsupported medical claims in section 2."}"""

guardrails_agent = Agent(
    name="Guardrails Agent",
    instructions=GUARDRAILS_PROMPT,
    model=MODEL,
    output_type=GuardrailVerdict,
)


@output_guardrail
async def article_guardrail(ctx, agent, output: str) -> GuardrailFunctionOutput:
    """Run the guardrails agent against the writer's article output."""
    result = await Runner.run(
        guardrails_agent,
        input=f"Review this article and return your verdict:\n\n{output}",
        context=ctx.context,
    )
    verdict: GuardrailVerdict = result.final_output
    passed = verdict.passes
    print(f"  {'✅' if passed else '❌'} Guardrails: {verdict.reason}")
    return GuardrailFunctionOutput(
        output_info=verdict,
        tripwire_triggered=not passed,
    )


# ── Writer Agents (defined once, with guardrail) ──────────────────────────────
writer_agent_a = Agent(
    name="Writer Agent A",
    instructions=WRITER_A_PROMPT,
    model=MODEL,
    output_guardrails=[article_guardrail],
)

writer_agent_b = Agent(
    name="Writer Agent B",
    instructions=WRITER_B_PROMPT,
    model=MODEL,
    output_guardrails=[article_guardrail],
)

# ── Orchestrator Agent ────────────────────────────────────────────────────────
ORCHESTRATOR_AGENT_PROMPT = """You are the orchestrator of a multi-agent article writing system.
Your job is to coordinate specialists and produce one polished article with an accompanying image.
Never write the article or do research yourself — always delegate.

Your exact process (follow in order):
1. Call the research_agent tool TWICE with slightly different phrasings of the topic to obtain two research briefs.
2. Compare the two briefs and select the richer, more factual one. Discard the other.
3. Call the image_generator_agent tool with the topic so it generates a relevant article image.
4. Decide on the intended audience:
   - If the topic is technical or professional → hand off to Writer Agent A (formal style).
   - If the topic is general-interest or consumer-facing → hand off to Writer Agent B (engaging style).
5. Hand off to the chosen writer, passing the selected research brief as input.
   The writer will produce the final article — your job ends there.

You are a manager. Coordinate, decide, delegate."""

orchestrator_agent = Agent(
    name="Orchestrator Agent",
    model="o4-mini",
    instructions=ORCHESTRATOR_AGENT_PROMPT,
    tools=[
        research_agent.as_tool(
            tool_name="research_agent",
            tool_description=(
                "Research a topic and return a brief containing key facts, "
                "statistics, themes, and source URLs. Pass the topic as input."
            ),
        ),
        image_generator_agent.as_tool(
            tool_name="image_generator_agent",
            tool_description=(
                "Generate a relevant article image. Pass the topic as input. "
                "Returns the local path of the saved PNG."
            ),
        ),
    ],
    handoffs=[writer_agent_a, writer_agent_b],
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Export Helper
# ═══════════════════════════════════════════════════════════════════════════════

def save_article_as_html(
    article_text: str,
    image_path,        # Path | str | None — local .png saved during the run
    writer_name: str,
    timestamp: str,
    output_dir: Path = Path("outputs"),
) -> dict:
    """
    Export the article to .md and .html (with embedded image) in output_dir.
    All three output files (image, .md, .html) share 'timestamp' so they are
    easy to match and never overwrite each other across runs.

    Returns {"md": Path, "html": Path}.
    """
    output_dir.mkdir(exist_ok=True)

    # 1. Save Markdown
    md_path = output_dir / f"article_{timestamp}.md"
    md_path.write_text(article_text, encoding="utf-8")

    # 2. Build image block using a relative path (HTML and PNG sit in the same folder)
    if image_path and Path(image_path).exists():
        image_filename = Path(image_path).name
        image_block = (
            f'<img src="{image_filename}" '
            f'alt="Article image" '
            f'style="width:100%;max-width:800px;border-radius:12px;margin:20px 0 32px;">'
        )
    else:
        image_block = ""

    # 3. Convert Markdown → HTML body
    html_body = markdown.markdown(article_text)

    # Extract title from the first # heading; fall back to a generic label
    title = "Article"
    for line in article_text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    # 4. Assemble full HTML page
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: Georgia, serif;
      max-width: 860px;
      margin: 40px auto;
      padding: 0 24px;
      line-height: 1.75;
      color: #1a1a1a;
      background: #fdfdfd;
    }}
    h1, h2, h3 {{ font-family: Arial, sans-serif; line-height: 1.25; }}
    h1 {{ font-size: 2em; margin-bottom: 4px; }}
    h2 {{ margin-top: 2em; }}
    .meta {{ color: #666; font-size: 0.88em; margin-bottom: 28px; }}
    a {{ color: #1a73e8; }}
    blockquote {{
      border-left: 4px solid #ccc;
      margin: 0 0 1em;
      padding-left: 16px;
      color: #555;
    }}
    code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
  </style>
</head>
<body>
  <div class="meta">Written by: Greg Gibson's {writer_name} &nbsp;|&nbsp; Generated: {timestamp}</div>
  {image_block}
  {html_body}
</body>
</html>"""

    html_path = output_dir / f"article_{timestamp}.html"
    html_path.write_text(html_content, encoding="utf-8")

    return {"md": md_path, "html": html_path}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Gradio Handler
# ═══════════════════════════════════════════════════════════════════════════════

async def run_article_writer(topic: str):
    """
    Async generator — yields intermediate status so Gradio can update the UI
    while the agents are running, then yields the final results when done.

    Yields tuples of: (status, image_path, article_markdown, md_file, html_file)
    """
    global generated_image_url, generated_image_path, run_timestamp

    # Input validation
    topic = topic.strip()
    if not topic:
        yield "⚠️ Please enter a topic before clicking Write Article.", None, None, None, None
        return

    if not OPENAI_API_KEY:
        yield "❌ OPENAI_API_KEY is not set. Add it to your .env file or Space Secrets.", None, None, None, None
        return

    # Initialise run state
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_image_url = None
    generated_image_path = None

    yield "🔄 Agents are working… research → image → writing → guardrails (2–4 min)", None, None, None, None

    # Run the agentic pipeline
    result = None
    try:
        result = await Runner.run(
            orchestrator_agent,
            input=f"Topic: {topic}",
            max_turns=50,
        )
    except OutputGuardrailTripwireTriggered as e:
        yield f"🚨 Guardrail blocked the article: {e}", None, None, None, None
        return
    except Exception as e:
        yield f"❌ An error occurred: {e}", None, None, None, None
        return
    finally:
        flush_traces()

    # Save exports
    writer_name = result.last_agent.name
    article_text = result.final_output

    paths = save_article_as_html(
        article_text=article_text,
        image_path=generated_image_path,
        writer_name=writer_name,
        timestamp=run_timestamp,
        output_dir=Path("outputs"),
    )

    status = f"✅ Done — written by Greg Gibson's {writer_name} | {run_timestamp}"
    image_out = str(generated_image_path) if generated_image_path and Path(generated_image_path).exists() else None

    yield status, image_out, article_text, str(paths["md"]), str(paths["html"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="AI Article Writer") as demo:

    gr.Markdown(
        """
        # 🖊️ AI Article Writer
        A multi-agent system (OpenAI Agents SDK) that researches your topic, generates an image,
        and writes a complete article — then checks it through an editorial guardrail.

        > ⏱️ **Each run takes 2–4 minutes.** The research agent queries multiple web sources before writing begins.
        """
    )

    with gr.Row():
        topic_input = gr.Textbox(
            label="Article Topic",
            placeholder=EXAMPLE_TOPIC,
            lines=2,
            scale=5,
        )
        submit_btn = gr.Button("Write Article ✍️", variant="primary", scale=1, min_width=180)

    status_out = gr.Textbox(label="Status", interactive=False, lines=1)

    with gr.Row():
        image_out = gr.Image(label="Article Image", type="filepath", height=420)
        with gr.Column():
            gr.Markdown("### 📥 Downloads")
            download_md   = gr.File(label="Markdown (.md)")
            download_html = gr.File(label="HTML with image (.html)")

    article_out = gr.Markdown(label="Article")

    submit_btn.click(
        fn=run_article_writer,
        inputs=[topic_input],
        outputs=[status_out, image_out, article_out, download_md, download_html],
    )



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Launch
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
