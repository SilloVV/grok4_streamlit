import streamlit as st
from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.search import SearchParameters, web_source
from dotenv import load_dotenv
import os

load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")

if "messages_admin" not in st.session_state:
    st.session_state.messages_admin = []
if "messages_test" not in st.session_state:
    st.session_state.messages_test = []
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

st.title("Assistant Juridique Français - Grok-4")

# Create tabs
tab1, tab2 = st.tabs(["🔒 Admin", "🧪 Test"])

def render_sidebar():
    """Render the sidebar with search parameters"""
    with st.sidebar:
        st.header("Paramètres de recherche")
    search_mode = st.selectbox(
        "Mode de recherche", 
        ["auto", "on", "off"], 
        index=0,
        help="auto: le modèle décide automatiquement, on: recherche activée, off: recherche désactivée"
    )
    return_citations = st.checkbox("Afficher les citations", value=True)
    
    st.header("✅ Avantages Grok-4")
    with st.expander("Points forts du modèle"):
        st.markdown("""
        **Intelligence**: Scores élevés sur les benchmarks
        - MMLU (connaissances générales)
        - GPQA (questions scientifiques avancées)
        
        **Spécialisations**: 
        - Raisonnement complexe
        - Analyse juridique approfondie
        - Recherche web intégrée
        - Sélection de domaines de recherche
        - Limitation du nombre de sources (max 6)
        """)
    
    st.header("⚠️ Limitations Grok-4")
    with st.expander("Informations importantes"):
        st.markdown("""
        **Contexte**: 260k tokens maximum
        
        **Vitesse moyenne**: ~16s avant le premier token
        
        **Coût recherche**: 0.025$ par source (pas par recherche)
        
        ⚡ Pour des réponses plus rapides, utilisez le mode "off" pour désactiver la recherche.
        """)
    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            col1, col2 = st.columns(2)
            
            with col1:
                if "advantages" in message and message["advantages"]:
                    with st.container(border=True):
                        st.markdown("**✅ Avantages**")
                        for advantage in message["advantages"]:
                            st.markdown(f"• {advantage}")
            
            with col2:
                if "disadvantages" in message and message["disadvantages"]:
                    with st.container(border=True):
                        st.markdown("**❌ Inconvénients**")
                        for disadvantage in message["disadvantages"]:
                            st.markdown(f"• {disadvantage}")
            
        
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for i, citation in enumerate(message["citations"], 1):
                    if isinstance(citation, dict):
                        st.write(f"**{i}.** [{citation.get('title', 'Source')}]({citation.get('url', '#')})")
                        if citation.get('snippet'):
                            st.write(f"_{citation['snippet']}_")
                    else:
                        st.write(f"**{i}.** {citation}")

if prompt := st.chat_input("Que voulez-vous demander à Grok-4?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            client = Client(api_key=GROK_API_KEY, timeout=3600)
            
            chat_params = {"model": "grok-4"}
            if search_mode != "off":
                chat_params["search_parameters"] = SearchParameters(
                    mode=search_mode,
                    return_citations=return_citations,
                    sources=[web_source(allowed_websites=["legifrance.gouv.fr", "juricaf.org", "conseil-etat.fr", "service-public.fr"])],
                    max_search_results=6
                )
            
            chat = client.chat.create(**chat_params)
            
            legal_prompt = """Vous êtes un assistant juridique spécialisé dans le droit français. Vous devez :

1. **Politesse** : Répondre aux salutations et formules de politesse de manière courtoise et professionnelle. Pour les simples salutations, répondez naturellement sans avertissement juridique.

2. **Expertise** : Fournir des informations précises sur le droit français (civil, pénal, commercial, administratif, du travail, etc.)

3. **Sources** : Toujours citer vos sources (Code civil, Code pénal, jurisprudence, etc.) et utiliser la recherche web pour les informations récentes

4. **Prudence** : Pour les questions juridiques, rappeler que vos réponses sont à titre informatif uniquement et qu'il est recommandé de consulter un avocat pour des conseils personnalisés

5. **Structure** : Organiser vos réponses juridiques de manière claire avec :
   - Le principe juridique général
   - Les textes de loi applicables
   - La jurisprudence pertinente si applicable
   - Les exceptions ou cas particuliers
   - Les démarches pratiques si nécessaire
   - Quand pertinent, lister les avantages et inconvénients d'une situation juridique

6. **Actualité** : Mentionner si des réformes récentes peuvent affecter la réponse

7. **Langue** : Répondre exclusivement en français avec la terminologie juridique appropriée pour les questions juridiques


"""
            chat.append(system(legal_prompt))
            
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat.append(user(msg["content"]))
            
            chat.append(user(prompt))
            
            message_placeholder = st.empty()
            reasoning_placeholder = st.empty()
            citations_placeholder = st.empty()
            cost_placeholder = st.empty()
            full_response = ""
            
            for response, chunk in chat.stream():
                full_response = response.content
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Display reasoning if available
            if hasattr(response, 'reasoning_content') and response.reasoning_content:
                with reasoning_placeholder.expander("🧠 Raisonnement du modèle"):
                    st.markdown(response.reasoning_content)
            
            # Calculate costs with actual token usage if available
            if hasattr(response, 'usage') and response.usage:
                completion_tokens = response.usage.completion_tokens
                reasoning_tokens = getattr(response.usage, 'reasoning_tokens', 0)
                # Estimate input tokens (3 per word as fallback)
                input_tokens = len(prompt.split()) * 3
                
                # Token cost calculation
                token_cost = (input_tokens * 3 / 1_000_000) + (completion_tokens * 15 / 1_000_000)
                if reasoning_tokens > 0:
                    token_cost += reasoning_tokens * 15 / 1_000_000  # Reasoning tokens cost same as output
            else:
                # Fallback to word estimation
                input_words = len(prompt.split())
                output_words = len(full_response.split())
                input_tokens = input_words * 3
                completion_tokens = output_words * 3
                reasoning_tokens = 0
                
                token_cost = (input_tokens * 3 / 1_000_000) + (completion_tokens * 15 / 1_000_000)
            
            citations = getattr(response, 'citations', None) if hasattr(response, 'citations') else None
            search_cost = 0
            num_sources = 0
            if citations:
                num_sources = len(citations)
                search_cost = num_sources * 0.025
            
            total_cost = token_cost + search_cost
            
            # Display cost breakdown
            with cost_placeholder.expander("💰 Coût de cette réponse"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_tokens = input_tokens + completion_tokens + reasoning_tokens
                    st.metric("Tokens utilisés", f"{total_tokens:,}")
                    caption = f"Input: {input_tokens:,} | Output: {completion_tokens:,}"
                    if reasoning_tokens > 0:
                        caption += f" | Raisonnement: {reasoning_tokens:,}"
                    st.caption(caption)
                with col2:
                    st.metric("Sources consultées", num_sources)
                    st.caption(f"Recherche: ${search_cost:.3f}")
                with col3:
                    st.metric("Coût total", f"${total_cost:.4f}")
                    st.caption(f"Tokens: ${token_cost:.4f}")
            
            citations = getattr(response, 'citations', None) if hasattr(response, 'citations') else None
            assistant_msg = {"role": "assistant", "content": full_response}
            if citations:
                assistant_msg["citations"] = citations
                with citations_placeholder.expander("Sources"):
                    for i, citation in enumerate(citations, 1):
                        if isinstance(citation, dict):
                            st.write(f"**{i}.** [{citation.get('title', 'Source')}]({citation.get('url', '#')})")
                            if citation.get('snippet'):
                                st.write(f"_{citation['snippet']}_")
                        else:
                            st.write(f"**{i}.** {citation}")
            
            st.session_state.messages.append(assistant_msg)
        
        except Exception as e:
            error_msg = f"Erreur: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Cost breakdown at bottom
st.markdown("---")
st.markdown("### 💰 Coût par question")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔤 Prix des tokens**")
    st.markdown("• Input: 3$ / 1M tokens")  
    st.markdown("• Output: 15$ / 1M tokens")

with col2:
    st.markdown("**🔍 Prix de recherche**")
    st.markdown("• 0.025$ par source consultée")
    st.markdown("• Maximum 6 sources = 0.15$ max")