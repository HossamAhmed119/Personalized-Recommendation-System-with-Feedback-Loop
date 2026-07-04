import traceback
from pathlib import Path
from typing import Optional
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.models.recommendation_engine import RecommendationEngine
from src.rag.retriever import ProductRetriever

load_dotenv()

CONFIG_PATH = str(Path("configs/app_config.yaml").resolve())
DATA_CONFIG_PATH = str(Path("configs/data_config.yaml").resolve())

STRATEGY_LABELS = {
    "Hybrid (NCF+CBF)": "hybrid",
    "Collaborative Filtering (NCF Only)": "ncf_only",
    "Content-Based (CBF Only)": "cbf_only",
}


@st.cache_resource(show_spinner="Initializing Recommendation Engine...")
def initialize_engine() -> RecommendationEngine:
    engine = RecommendationEngine(config_path=CONFIG_PATH)
    engine.load_systems()
    return engine


@st.cache_resource(show_spinner="Initializing Conversational Retriever...")
def initialize_retriever(_engine: RecommendationEngine) -> ProductRetriever:
    retriever = ProductRetriever(
        app_config_path=CONFIG_PATH,
        data_config_path=DATA_CONFIG_PATH,
    )
    # Reuse the catalog already loaded by the engine instead of reading
    # the parquet file from disk a second time.
    retriever.load_products(_engine.products_df)
    retriever.index_products()
    return retriever


def render_instructions() -> None:
    st.markdown(
        """
        ### How to use this dashboard

        1. Select your User ID from the dropdown in the sidebar.
        2. Choose a recommendation strategy: Hybrid, Collaborative Filtering only, or Content-Based only.
        3. Select the product you are currently viewing, if any. This improves content-based scoring.
        4. Adjust the number of recommendations using the slider.
        5. Click "Generate Recommendations" to view your personalized product list.
        6. Use the chat assistant below to ask questions about the recommended products.
        """
    )
    st.markdown("---")


def get_item_options(engine: RecommendationEngine) -> list:
    try:
        items = sorted(engine.item_encoder.classes_.tolist())
        return items[:100] 
    except Exception:
        return []


@st.cache_data
def load_image_mapping(mapping_path: str) -> dict:
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def display_recommendations(results: pd.DataFrame, engine: RecommendationEngine) -> None:
    if results.empty:
        st.warning("No recommendations available for the selected inputs.")
        return

    st.success("Recommendations generated successfully.")

    image_mapping_path = Path(engine.config["database"]["image_mapping"]).resolve()
    image_mapping = load_image_mapping(str(image_mapping_path))
    
    default_image = str(Path(engine.config["ui"]["default_product_image"]).resolve())
    products_per_row = int(engine.config["ui"].get("products_per_row", 4))

    title_col = None
    for col in ["title", "product_title"]:
        if col in results.columns:
            title_col = col
            break

    cols = st.columns(products_per_row)
    
    for index, row in results.iterrows():
        col = cols[index % products_per_row]
        item_id = str(row[engine.item_id_col])
        img_source = image_mapping.get(item_id, default_image)
        
        raw_title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else "Unknown Product"
        display_title = raw_title[:47] + "..." if len(raw_title) > 50 else raw_title
        score = row["score"]

        with col:
            st.container(height=350, border=True)
            try:
                st.image(img_source, width='stretch')
            except Exception:
                st.image(default_image, width='stretch')
                
            st.markdown(f"**{display_title}**")
            st.caption(f"ID: {item_id}")
            st.caption(f"Score: {score:.4f}")


def render_chat_section(retriever: Optional[ProductRetriever], has_recommendations: bool) -> None:
    st.markdown("---")
    st.subheader("Chat with the Shopping Assistant")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Ask me anything, or ask about the recommended products...")

    if user_message:
        # History passed to retriever.chat() must NOT include the current
        # message yet, since chat() appends it itself (plain or RAG-augmented).
        history_so_far = list(st.session_state["chat_history"])

        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        if retriever is None:
            reply = "The conversational assistant is not available right now."
        else:
            try:
                # Recommendation intent is signaled simply by whether the
                # user has already generated a recommendation list in this
                # session; retriever.chat() decides internally whether to
                # run semantic_search and augment the prompt with product
                # context, or just answer the message directly via the model.
                reply = retriever.chat(
                    user_message=user_message,
                    conversation_history=history_so_far,
                    wants_recommendations=has_recommendations,
                )
            except Exception:
                reply = "An error occurred while generating a response."
                st.error(reply)
                st.code(traceback.format_exc())

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def main() -> None:
    st.set_page_config(
        page_title="Intelligent Shopping Assistant",
        layout="wide",
    )

    st.title("Intelligent Shopping Assistant")
    render_instructions()

    try:
        engine = initialize_engine()
    except Exception:
        st.error(
            "Critical System Failure: Could not load the recommendation engine. "
            "Please check configuration and model paths."
        )
        st.code(traceback.format_exc())
        return

    try:
        retriever = initialize_retriever(engine)
    except Exception:
        retriever = None
        st.warning(
            "The conversational assistant could not be initialized. "
            "Recommendations are still available."
        )
        st.code(traceback.format_exc())

    st.sidebar.header("Recommendation Settings")

    st.sidebar.header("Recommendation Settings")

    user_options = sorted(list(engine.valid_users))
    safe_user_options = user_options[:100] if len(user_options) > 100 else user_options

    selected_user = st.sidebar.selectbox(
        "Select User ID (Sample of 100):",
        options=safe_user_options,
        index=0 if safe_user_options else None,
    )

    strategy_label = st.sidebar.radio(
        "Recommendation Strategy:",
        options=list(STRATEGY_LABELS.keys()),
    )
    strategy = STRATEGY_LABELS[strategy_label]

    item_options = get_item_options(engine)
    selected_item = st.sidebar.selectbox(
        "Product Currently Viewing (optional):",
        options=["None"] + item_options,
    )
    item_id = None if selected_item == "None" else selected_item

    top_k = st.sidebar.slider(
        "Number of Recommendations:",
        min_value=1,
        max_value=50,
        value=10,
    )

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = pd.DataFrame()

    if st.sidebar.button("Generate Recommendations"):
        if not selected_user:
            st.sidebar.warning("Please select a valid User ID.")
        else:
            with st.spinner(f"Generating recommendations for {selected_user}..."):
                try:
                    results = engine.generate_recommendations(
                        user_id=selected_user,
                        item_id=item_id,
                        strategy=strategy,
                        top_k=top_k,
                    )
                    st.session_state["last_results"] = results
                except Exception:
                    st.error("An error occurred while generating recommendations.")
                    st.code(traceback.format_exc())
                    st.session_state["last_results"] = pd.DataFrame()

    display_recommendations(st.session_state["last_results"], engine)

    has_recommendations = not st.session_state["last_results"].empty
    render_chat_section(retriever, has_recommendations)


if __name__ == "__main__":
    main()