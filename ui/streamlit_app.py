"""
Streamlit front end for Curiosity AI.

This simple UI allows a user to enter a seed topic, invoke the API and
visualise the resulting questions and detected contradictions.  It uses the
requests library to communicate with the FastAPI backend running on
`localhost:8000` by default.
"""
import json

import streamlit as st
import requests

# Endpoint of the running API.  Adjust the port if you run the server on a
# different port or host.  Using a relative path (e.g. "/ask") does not
# work in Streamlit; you must specify the full URL including protocol.
API_URL = "http://localhost:8000/ask"


def main() -> None:
    st.set_page_config(page_title="Curiosity AI", layout="wide")
    st.title("🧠 Curiosity AI")
    st.write(
        "Curiosity AI explores a topic by asking increasingly novel questions and "
        "highlighting contradictions. Enter a seed topic below to begin."
    )

    topic = st.text_input("Seed topic", value="Teacher training vs curriculum reform outcomes in Nigeria", key="topic_input")
    if st.button("Explore", key="explore_button"):
        if not topic.strip():
            st.error("Please enter a non‑empty topic.")
        else:
            with st.spinner("Generating questions..."):
                try:
                    response = requests.post(API_URL, json={"topic": topic.strip()})
                    if response.status_code != 200:
                        st.error(f"Error {response.status_code}: {response.text}")
                        return
                    data = response.json()
                    # Display the curiosity trail
                    st.subheader("Curiosity Trail")
                    if not data["session"]["trail"]:
                        st.write("No questions generated. Try a different topic.")
                    for idx, entry in enumerate(data["session"]["trail"]):
                        st.markdown(
                            f"**{idx + 1}.** {entry['q']} — novelty `{entry['novelty']:.2f}`"
                        )
                    # Display dissonance log
                    st.subheader("Dissonance Log")
                    if not data["dissonance"]:
                        st.write("No contradictions detected.")
                    else:
                        for item in data["dissonance"]:
                            st.markdown(
                                f"- ⚠️ {item['contradiction']:.2f}\n  • {item['a']}\n  • {item['b']}"
                            )
                except requests.RequestException as e:
                    st.error(f"Failed to call API: {e}")


if __name__ == "__main__":
    main()