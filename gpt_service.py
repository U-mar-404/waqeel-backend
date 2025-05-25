# gpt_service.py

import openai
from config import OPENAI_API_KEY

# set your API key
openai.api_key = OPENAI_API_KEY

def generate_answer_with_gpt(query, top_sections, model="gpt-4o-mini"):
    """
    Builds a context block from the top_sections (both PPC & CrPC),
    then asks GPT to answer using *only* that context.
    """
    # 1) assemble CONTEXT string
    context = ""
    for sec in top_sections:
        # sec["Section"] is the number, sec["Section_name"] is the title
        context += f"Section {sec['Section']} – {sec.get('Section_name','').strip()}\n"
        context += sec.get('Description','').strip() + "\n\n"

    # 2) craft the prompt
    prompt = (
    "You are a professional legal assistant trained on both the Pakistan Penal Code (PPC) and the Code of Criminal Procedure (CrPC). "
    "Your task is to answer the user's question using relevant legal sections from PPC or CrPC if possible. If no relevant section applies, you must search the web for the correct legal explanation.\n\n"

    "🧠 Logic:\n"
    "- First, check if any section numbers in `context` are relevant to the query.\n"
    "- If they are relevant, fetch their full text (if needed) and respond using PPC/CrPC sections.\n"
    "- If no section in the context applies, or context is empty, search the web and respond with an accurate, sourced explanation.\n\n"

    "📝 Format the response using **Markdown**:\n\n"

    "✅ **GPT Answer:**\n"
    "Start with an empathetic sentence about the situation.\n\n"

    "🔎 **Applicable Legal Sections (PPC/CrPC):**\n"
    "- List each relevant section using this format:\n\n"
    "🔴 **Section {number} – {short title}**\n"
    "**Description:**\n"
    "{Plain-language explanation}\n\n"
    "**Punishment/Procedure:**\n"
    "- Point 1\n"
    "- Point 2\n\n"

    "📌 **What Should You Do Now?**\n"
    "- Step-by-step practical advice in bullet points\n\n"

    "⚖️ **Summary:**\n"
    "- Brief recap of applicable sections and what they imply.\n\n"

    "🌐 **Web-Based Legal Guidance (if PPC/CrPC not applicable):**\n"
    "If no PPC/CrPC content is relevant, use this section instead.\n"
    "- Provide a clear, simple answer based on online legal information.\n"
    "- Do not guess. Use real legal procedure or terminology.\n"

    "⚠️ If neither PPC nor CrPC sections apply and no web search is performed, respond exactly:\n"
    "> I’m sorry, the provided legal content (PPC/CrPC) does not cover this situation. For civil or specialized matters, please consult a qualified lawyer or check relevant civil laws.\n\n"

    f"Question: {query}\n"
    f"Content:\n{context}"
)

    # 3) call OpenAI
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # 4) return the assistant’s reply
    return response.choices[0].message.content.strip()
