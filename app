import os
os.environ["STREAMLIT_WATCHDOG_POLL_INTERVAL"] = "5000"
import threading
threading.currentThread().setName("MainThread")
import streamlit as st
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import nltk
from nltk.translate.bleu_score import corpus_bleu
import scipy

# Download the required NLTK data
nltk.download('wordnet')

# Set up the model and tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

# Define the translation function
@st.cache_resource()
def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

# Define the BLEU score calculation function
def calculate_bleu_score(reference, candidate):
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference]
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    bleu_score = corpus_bleu([reference_tokens], [candidate_tokens])
    return bleu_score

# Define the similarity evaluation function
def evaluate_similarity(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    similarity_score = scipy.spatial.distance.jaccard(reference_tokens, candidate_tokens)
    return 1 - similarity_score

# Streamlit app
st.title("English to Hindi Text Translator")
english_text = st.text_area("Enter English text:")

if st.button("Translate"):
    hindi_translation = translate(english_text, "en", "hi")
    st.write("Hindi Translation:")
    st.write(hindi_translation)

    # Calculate and display BLEU score
    reference_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
    bleu_score = calculate_bleu_score([reference_text], hindi_translation)
    st.write(f"BLEU Score: {bleu_score:.2f}")

    # Evaluate and display similarity
    similarity = evaluate_similarity(reference_text, hindi_translation)
    st.write(f"Similarity Score: {similarity:.2f}")

    # Evaluate translation
    if similarity >= 0.8:
        st.write("Translation is accurate and similar to the reference.")
    elif similarity >= 0.5:
        st.write("Translation is somewhat accurate but not very similar to the reference.")
    else:
        st.write("Translation is inaccurate and dissimilar to the reference.")
