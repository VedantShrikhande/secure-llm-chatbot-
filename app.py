# -*- coding: utf-8 -*-
"""app.py - Secure LLM Chatbot with Filters"""

import streamlit as st
from backend.chatbot import SafeChatbot

# Page configuration
st.set_page_config(page_title='Secure LLM Chatbot', layout='centered')

# Title and description
st.title('Secure LLM Chatbot')
st.write('A demo chatbot that filters PII and hate/bias content before replying.')

# Initialize chatbot
if 'bot' not in st.session_state:
    st.session_state['bot'] = SafeChatbot()

bot: SafeChatbot = st.session_state['bot']

# Initialize conversation state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Input form
with st.form('input_form', clear_on_submit=True):
    user_input = st.text_area('You:', height=120)
    submitted = st.form_submit_button('Send')

# Process user input

# Placeholder definitions for demonstration purposes
submitted = True  # Assume the form is submitted for demonstration
user_input = "This is a test message." # Assume some user input

# Placeholder for a bot object with a chat method
class MockBot:
    def chat(self, text):
        # Simple mock response - replace with actual bot logic
        if "hate" in text.lower():
            return "This message seems to contain hate speech.", "Hate Speech"
        else:
            return "Thank you for your message.", None

bot = MockBot()

# Use a regular dictionary to simulate session state in the notebook
session_state = {}

# Initialize conversation history
if 'conversation' not in session_state:
    session_state['conversation'] = []


if submitted and user_input.strip():
    # Get bot response
    response, blocked_reason = bot.chat(user_input)

    # Save conversation
    session_state['conversation'].append(("You", user_input))
    if blocked_reason:
        session_state['conversation'].append(("Bot (Filtered)", response + f" ⚠️ [Blocked: {blocked_reason}]"))
    else:
        session_state['conversation'].append(("Bot", response))

# Display the conversation state for demonstration
print(session_state['conversation'])

# Display conversation
for speaker, text in st.session_state['conversation']:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {text}")
    else:
        st.markdown(f"*{speaker}:* {text}")
