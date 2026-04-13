import streamlit as st
import requests
import json

# Must be the first Streamlit command
st.set_page_config(
    page_title="HRL Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

# Constants
API_URL = "http://localhost:8000/ask"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Legal Assistant powered by Reinforcement Learning. Ask me any question related to the Indian Constitution!"}
    ]

# Custom CSS for modern styling
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        border-top: 1px solid #444;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #1e3a8a;
        color: #93c5fd;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .reward-text {
        color: #4ade80;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


st.title("⚖️ Legal QA Chatbot")
st.caption("Powered by Hierarchical Reinforcement Learning (Decompose → Retrieve → Generate → Combine)")

# Draw the chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If this assistant message contains complex reasoning metadata, display it elegantly
        if "metadata" in msg:
            meta = msg["metadata"]
            
            # Show badge for complex vs simple
            if meta.get("is_complex"):
                st.markdown('<span class="status-badge"> Multi-Hop HRL Active</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge"> Single-Hop Active</span>', unsafe_allow_html=True)
                
            with st.expander(" View Agent's Internal Thought Process"):
                st.write(f"**Complexity Score**: {meta.get('complexity_score', 0)}")
                st.write(f"**Actions Taken**: {' → '.join(meta.get('actions_taken', []))}")
                
                # Decomposed questions
                if meta.get("sub_questions"):
                    st.markdown("#####  1. Decomposed Questions")
                    for sq in meta.get("sub_questions", []):
                        st.write(f"- {sq}")
                        
                # Retrieved Articles
                if meta.get("retrieved_articles"):
                    st.markdown("#####  2. Retrieved Constitution Articles")
                    for art in meta.get("retrieved_articles", []):
                        st.markdown(f"**Article {art.get('article_num')}**: {art.get('title')}")
                        st.caption(f"{art.get('text_snippet')}...")
                
                # Sub-answers
                if meta.get("sub_answers"):
                    st.markdown("#####  3. Generated Sub-answers")
                    for sa in meta.get("sub_answers", []):
                        st.info(sa)
                
                # Rewards
                if meta.get("rewards"):
                    st.markdown("#####  4. RL Rewards")
                    st.markdown(f"Combined Reward: <span class='reward-text'>+{meta.get('combined_reward', 0)}</span>", unsafe_allow_html=True)
                    st.json(meta.get("rewards", {}))

# Handle new user input
if prompt := st.chat_input("E.g., What is the difference between Article 14 and Article 21?"):
    
    # 1. Add user message to history and show it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call backend and stream response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing question..."):
            try:
                response = requests.post(API_URL, json={"question": prompt}, timeout=120)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("final_answer", "Error: No answer provided.")
                    
                    st.markdown(answer)
                    
                    # Package metadata to save in history
                    metadata = {
                        "complexity_score": data.get("complexity_score"),
                        "is_complex": data.get("is_complex"),
                        "sub_questions": data.get("sub_questions"),
                        "retrieved_articles": data.get("retrieved_articles"),
                        "sub_answers": data.get("sub_answers"),
                        "actions_taken": data.get("actions_taken"),
                        "rewards": data.get("rewards"),
                        "combined_reward": data.get("combined_reward")
                    }
                    
                    # Optional: display internal logic immediately for this turn
                    if data.get("is_complex"):
                        st.markdown('<span class="status-badge"> Multi-Hop HRL Active</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-badge"> Single-Hop Active</span>', unsafe_allow_html=True)

                    with st.expander(" View Agent's Internal Thought Process"):
                        st.write(f"**Complexity Score**: {metadata['complexity_score']}")
                        st.write(f"**Actions Taken**: {' → '.join(metadata['actions_taken'])}")
                        # Decomposed questions
                        if metadata["sub_questions"]:
                            st.markdown("#####  1. Decomposed Questions")
                            for sq in metadata["sub_questions"]:
                                st.write(f"- {sq}")
                        # Retrieved Articles
                        if metadata["retrieved_articles"]:
                            st.markdown("#####  2. Retrieved Constitution Articles")
                            for art in metadata["retrieved_articles"]:
                                st.markdown(f"**Article {art.get('article_num')}**: {art.get('title')}")
                                st.caption(f"{art.get('text_snippet')}...")
                        # Sub-answers
                        if metadata["sub_answers"]:
                            st.markdown("#####  3. Generated Sub-answers")
                            for sa in metadata["sub_answers"]:
                                st.info(sa)
                        # Rewards
                        st.markdown("#####  4. RL Rewards")
                        st.markdown(f"Combined Reward: <span class='reward-text'>+{metadata['combined_reward']}</span>", unsafe_allow_html=True)
                    
                    # 3. Save assistant response logic to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata
                    })

                else:
                    st.error(f"Backend Error [{response.status_code}]: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Please ensure the FastAPI server is running with `uvicorn chatbot.app:app`")
