"""Streamlit study workspace for the StudyBuddy AI API."""

import os
from typing import Any

import httpx
import streamlit as st

st.set_page_config(page_title="StudyBuddy AI", page_icon="📚", layout="wide")
API_BASE_URL = os.getenv("STUDYBUDDY_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_request(
    method: str,
    path: str,
    token: str | None = None,
    **kwargs: Any,
) -> httpx.Response | None:
    """Call the backend and display a clear API or connectivity error."""
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{path}",
            headers=headers,
            timeout=60,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the StudyBuddy API: {exc}")
        return None
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        st.error(f"API error ({response.status_code}): {detail}")
    return response


def authenticate() -> None:
    """Render login and registration before opening the study workspace."""
    st.title("StudyBuddy AI")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    for tab, action in ((login_tab, "login"), (register_tab, "register")):
        with tab:
            with st.form(f"{action}_form"):
                email = st.text_input("Email", key=f"{action}_email")
                password = st.text_input("Password", type="password", key=f"{action}_password")
                submitted = st.form_submit_button("Log in" if action == "login" else "Register")
            if submitted:
                response = api_request(
                    "POST",
                    f"/api/v1/auth/{action}",
                    json={"email": email, "password": password},
                )
                if response is not None and response.is_success:
                    st.session_state.token = response.json()["access_token"]
                    st.rerun()


def select_document(token: str) -> dict[str, Any] | None:
    """Upload a PDF and choose one of the signed-in user's documents."""
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.sidebar.button("Upload PDF"):
        response = api_request(
            "POST",
            "/api/v1/documents",
            token,
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
        )
        if response is not None and response.is_success:
            st.sidebar.success(f"Uploaded {response.json()['filename']}")

    response = api_request("GET", "/api/v1/documents", token)
    if response is None or response.is_error:
        return None
    documents = response.json()
    if not documents:
        st.info("Upload a text-based PDF to get started.")
        return None
    labels = {f"{doc['filename']} (#{doc['id']})": doc for doc in documents}
    selected = st.sidebar.selectbox("Study document", list(labels))
    return labels[selected]


def submit_chat(token: str, document_id: int, mode: str, quiz_count: int = 5) -> None:
    """Send a question or study action and render its answer and sources."""
    question = st.chat_input("Ask about the selected PDF")
    if not question:
        return
    st.chat_message("user").write(question)
    response = api_request(
        "POST",
        "/api/v1/chat",
        token,
        json={
            "document_id": document_id,
            "message": question,
            "mode": mode,
            "quiz_count": quiz_count,
        },
    )
    if response is None or response.is_error:
        return
    result = response.json()
    with st.chat_message("assistant"):
        st.write(result["answer"])
        if result.get("source") == "web":
            st.caption("Answer source: web")
            for url in result.get("urls", []):
                st.markdown(f"- [{url}]({url})")
        for citation in result.get("citations", []):
            st.caption(f"Page {citation['page']}: {citation['text']}")
        for question_item in result.get("quiz", []):
            st.markdown(f"**{question_item['question']}**")
            for option in question_item["options"]:
                st.write(f"- {option}")
            with st.expander("Answer and explanation"):
                st.write(question_item["answer"])
                st.write(question_item["explanation"])
                st.caption(f"Page {question_item['page']}")


def show_topics(token: str, document_id: int) -> None:
    """Load topic labels and plot their PCA coordinates."""
    response = api_request("GET", f"/api/v1/documents/{document_id}/topics", token)
    if response is None or response.is_error:
        return
    analysis = response.json()
    if not analysis["points"]:
        st.info("No topic points are available for this document.")
        return
    st.subheader("Main topics")
    for topic in analysis["topics"]:
        st.write(f"**{topic['label']}** ({topic['chunk_count']} passages)")
    st.subheader("Topic map")
    st.scatter_chart(analysis["points"], x="x", y="y", color="topic_id")


def main() -> None:
    """Render authentication or the signed-in study tabs."""
    st.title("StudyBuddy AI")
    if "token" not in st.session_state:
        authenticate()
        return

    if st.sidebar.button("Log out"):
        del st.session_state.token
        st.rerun()
    document = select_document(st.session_state.token)
    if document is None:
        return
    document_id = document["id"]
    chat_tab, summary_tab, quiz_tab, topics_tab = st.tabs(["Chat", "Summary", "Quiz", "Topics"])
    with chat_tab:
        submit_chat(st.session_state.token, document_id, "qa")
    with summary_tab:
        if st.button("Generate document summary"):
            response = api_request(
                "POST",
                "/api/v1/chat",
                st.session_state.token,
                json={"document_id": document_id, "message": "Summarize this document", "mode": "summary"},
            )
            if response is not None and response.is_success:
                result = response.json()
                st.write(result["answer"])
                for citation in result["citations"]:
                    st.caption(f"Page {citation['page']}: {citation['text']}")
    with quiz_tab:
        quiz_count = st.number_input("Number of questions", min_value=1, max_value=10, value=5)
        if st.button("Generate quiz"):
            response = api_request(
                "POST",
                "/api/v1/chat",
                st.session_state.token,
                json={
                    "document_id": document_id,
                    "message": "Create a multiple-choice quiz",
                    "mode": "quiz",
                    "quiz_count": quiz_count,
                },
            )
            if response is not None and response.is_success:
                quiz_key = f"quiz-{document_id}"
                for index, _ in enumerate(st.session_state.get(quiz_key, [])):
                    answer_key = f"quiz-answer-{document_id}-{index}"
                    if answer_key in st.session_state:
                        del st.session_state[answer_key]
                st.session_state[quiz_key] = response.json()["quiz"]

        quiz_items = st.session_state.get(f"quiz-{document_id}", [])
        correct_count = 0
        all_answered = bool(quiz_items)
        for index, item in enumerate(quiz_items):
            st.markdown(f"**{item['question']}**")
            selected_option = st.radio(
                "Choose an answer",
                item["options"],
                index=None,
                key=f"quiz-answer-{document_id}-{index}",
            )
            if selected_option is None:
                all_answered = False
            elif selected_option == item["answer"]:
                correct_count += 1
            with st.expander("Answer and explanation", expanded=selected_option is not None):
                if selected_option is not None:
                    if selected_option == item["answer"]:
                        st.success(f"Correct! The answer is {item['answer']}.")
                    else:
                        st.error("Wrong answer. Please review the explanation below.")
                        st.write(f"Correct answer: {item['answer']}")
                else:
                    st.write(item["answer"])
                st.write(item["explanation"])
                st.caption(f"Page {item['page']}")
        if all_answered:
            st.success(f"Quiz complete! Your score: {correct_count}/{len(quiz_items)}")
    with topics_tab:
        show_topics(st.session_state.token, document_id)


if __name__ == "__main__":
    main()
