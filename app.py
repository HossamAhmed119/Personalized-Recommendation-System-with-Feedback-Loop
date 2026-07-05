import traceback
from pathlib import Path
from typing import Optional
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import yaml

from src.models.recommendation_engine import RecommendationEngine
from src.rag.retriever import ProductRetriever
from src.database.db_manager import DatabaseManager

load_dotenv()

CONFIG_PATH = str(Path("configs/app_config.yaml").resolve())
DATA_CONFIG_PATH = str(Path("configs/data_config.yaml").resolve())

STRATEGY_LABELS = {
    "Hybrid (NCF+CBF) 🧠": "hybrid",
    "Collaborative Filtering (NCF Only) 👥": "ncf_only",
    "Content-Based (CBF Only) 📦": "cbf_only",
}

@st.cache_resource(show_spinner="Initializing Recommendation Engine ⚙️...")
def initialize_engine() -> RecommendationEngine:
    engine = RecommendationEngine(config_path=CONFIG_PATH)
    engine.load_systems()
    return engine

@st.cache_resource(show_spinner="Initializing Conversational Retriever 🤖...")
def initialize_retriever() -> ProductRetriever:
    retriever = ProductRetriever(
        app_config_path=CONFIG_PATH,
        data_config_path=DATA_CONFIG_PATH,
    )
    return retriever

@st.cache_resource(show_spinner="Connecting to Database & Syncing Users 🗄️...")
def initialize_db(_engine: RecommendationEngine) -> DatabaseManager:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = config["database"]["interactions_db"]
    db = DatabaseManager(db_path)
    ml_users_list = [str(u) for u in _engine.valid_users]
    db.seed_users(ml_users_list)
    return db

def render_instructions() -> None:
    st.markdown(
        """
        ### 📖 How to use this dashboard
        1. 👤 Select or Create your User ID from the sidebar.
        2. 🎛️ Choose a recommendation strategy.
        3. 🚀 Click "Generate Recommendations" to view your personalized list.
        4. 👍 **Interact:** Like products to save them to your profile.
        5. 💬 Use the Chat Assistant or 📜 view your Interaction History in the tabs below.
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

def display_recommendations(results: pd.DataFrame, engine: RecommendationEngine, db: DatabaseManager, selected_user: str) -> None:
    if results.empty:
        st.warning("⚠️ No recommendations available for the selected inputs.")
        return

    st.success("✨ Recommendations generated successfully!")

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
        
        raw_title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else "Unknown Product ❓"
        display_title = raw_title[:47] + "..." if len(raw_title) > 50 else raw_title
        score = row["score"]

        with col:
            with st.container(height=420, border=True):
                try:
                    st.image(img_source, width='stretch')
                except Exception:
                    st.image(default_image, width='stretch')
                    
                st.markdown(f"**{display_title}**")
                st.caption(f"🆔 ID: {item_id} | 🎯 Score: {score:.4f}")
                
                if st.button("👍 Like", key=f"like_{item_id}_{index}", use_container_width=True):
                    db.record_interaction(user_id=selected_user, product_id=item_id, interaction_type="like")
                    st.toast(f"Saved {item_id} to your profile! 🎉", icon="✅")

def render_chat_section(retriever: Optional[ProductRetriever]) -> None:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("💭 Ask me anything about products...")

    if user_message:
        history_so_far = list(st.session_state["chat_history"])

        st.session_state["chat_history"].append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        if retriever is None:
            reply = "🚫 The conversational assistant is not available right now."
        else:
            try:
                with st.spinner("🔍 Searching catalog & generating response..."):
                    reply = retriever.chat(
                        user_message=user_message,
                        conversation_history=history_so_far,
                        wants_recommendations=True,
                    )
            except Exception:
                reply = "❌ An error occurred while generating a response."
                st.error(reply)
                st.code(traceback.format_exc())

        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

def main() -> None:
    st.set_page_config(page_title="Intelligent Shopping Assistant", layout="wide")
    st.title("🛒 Intelligent Shopping Assistant 🤖")
    render_instructions()

    try:
        engine = initialize_engine()
        db = initialize_db(engine)
    except Exception:
        st.error("🚨 Critical System Failure: Could not load the systems.")
        st.code(traceback.format_exc())
        return

    try:
        retriever = initialize_retriever()
    except Exception:
        retriever = None

    st.sidebar.header("👤 User Management")
    
    with st.sidebar.expander("➕ Create New User"):
        new_user_id = st.text_input("📝 Enter New User ID:")
        if st.button("✨ Create", use_container_width=True):
            if new_user_id.strip():
                db.add_new_user(new_user_id.strip())
                st.success("✅ User created successfully!")
                st.rerun()
            else:
                st.warning("⚠️ Please enter a valid ID.")

    all_users = db.get_all_users()
    user_ids = [u[0] for u in all_users]
    
    safe_user_options = user_ids[:100] if len(user_ids) > 100 else user_ids
    
    selected_user = st.sidebar.selectbox(
        "🧑‍💻 Select User ID (Sample of 100):",
        options=safe_user_options,
        index=0 if safe_user_options else None,
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Recommendation Settings")

    strategy_label = st.sidebar.radio("🎯 Recommendation Strategy:", options=list(STRATEGY_LABELS.keys()))
    strategy = STRATEGY_LABELS[strategy_label]

    item_options = get_item_options(engine)
    selected_item = st.sidebar.selectbox("🛍️ Product Currently Viewing (optional):", options=["None"] + item_options)
    item_id = None if selected_item == "None" else selected_item

    top_k = st.sidebar.slider("🔢 Number of Recommendations:", min_value=1, max_value=50, value=8)

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = pd.DataFrame()
        st.session_state["has_generated"] = False

    if st.sidebar.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        if not selected_user:
            st.sidebar.warning("⚠️ Please select or create a User ID first.")
        else:
            st.session_state["has_generated"] = True # تسجيل ضغطة الزرار
            with st.spinner(f"⏳ Generating recommendations for {selected_user}..."):
                try:
                    results = engine.generate_recommendations(
                        user_id=selected_user,
                        item_id=item_id,
                        strategy=strategy,
                        top_k=top_k,
                    )
                    st.session_state["last_results"] = results
                except Exception:
                    st.error("❌ An error occurred while generating recommendations.")
                    st.code(traceback.format_exc())
                    st.session_state["last_results"] = pd.DataFrame()

    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "💬 AI Assistant", "📜 Interaction History"])

    with tab1:
        st.subheader("🎯 Your Personalized Picks")
        if selected_user:
            if st.session_state["has_generated"]:
                display_recommendations(st.session_state["last_results"], engine, db, selected_user)
            else:
                st.info("👆 Click 'Generate Recommendations' in the sidebar to see your picks!")
        else:
            st.info("ℹ️ Select a user from the sidebar to view recommendations.")

    with tab2:
        st.subheader("🤖 Chat with the Shopping Assistant")
        render_chat_section(retriever)

    with tab3:
        st.subheader(f"📜 History for User: {selected_user}")
        if selected_user:
            history = db.get_user_interactions(selected_user)
            if history:
                history_df = pd.DataFrame(history, columns=["Product ID 📦", "Action 🎬", "Timestamp ⏱️"])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("👻 No past interactions found for this user. Like some products to see them here!")
        else:
            st.warning("⚠️ Please select a user first.")

if __name__ == "__main__":
    main()