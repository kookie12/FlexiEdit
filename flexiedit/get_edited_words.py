import torch
# import clip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from difflib import SequenceMatcher

# NLTK data download
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_edited_phrases(p_src, p_tar):
    # Split the source and target prompts into words and POS tags
    src_tokens = pos_tag(word_tokenize(p_src.lower()))
    tar_tokens = pos_tag(word_tokenize(p_tar.lower()))
    
    matcher = SequenceMatcher(None, src_tokens, tar_tokens)
    edited_phrases = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'insert':
            phrase = ' '.join([token for token, pos in tar_tokens[j1:j2]])
            edited_phrases.append(phrase)
    
    return edited_phrases

def expand_adj_noun_pairs(phrases, p_tar):
    tar_tokens = word_tokenize(p_tar)
    tar_pos_tags = pos_tag(tar_tokens)
    
    expanded_phrases = []
    for phrase in phrases:
        words = word_tokenize(phrase)
        
        # Find nouns following an adjective
        for i in range(len(tar_pos_tags)):
            if tar_pos_tags[i][0] == words[0] and tar_pos_tags[i][1].startswith('JJ'):
                # If the next word is a noun
                if i + 1 < len(tar_pos_tags) and tar_pos_tags[i + 1][1].startswith('NN'):
                    expanded_phrases.append(f"{words[0]} {tar_pos_tags[i + 1][0]}")
                    break
                
    # Combine expanded phrases with the original ones (removing duplicates)
    final_phrases = set(phrases) | set(expanded_phrases)
    
    for word in phrases:
        for expanded_word in expanded_phrases:
            if word in expanded_word:
                final_phrases.remove(word)
    
    return list(final_phrases)

def remove_unnecessary_words(phrases):
    stop_words = set(stopwords.words('english'))
    cleaned_phrases = []
    for phrase in phrases:
        words = word_tokenize(phrase)
        cleaned_phrase = ' '.join([word for word in words if word.lower() not in stop_words])
        cleaned_phrases.append(cleaned_phrase)
    
    return cleaned_phrases

def calculate_clip_similarity(image, text):
    # Calculate CLIP embeddings for image and text
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    # Calculate similarity between image and text embeddings
    similarity = torch.cosine_similarity(image_features, text_features)
    
    return similarity.item()

def find_edited_phrases(image_path, p_src, p_tar, threshold_multiplier=1.0):
    image = Image.open(image_path)
    # Extract edited phrases
    edited_phrases = get_edited_phrases(p_src, p_tar)
    
    # Extract with nouns following adjectives
    expanded_phrases = expand_adj_noun_pairs(edited_phrases, p_tar)
    
    # Remove unnecessary articles
    cleaned_phrases = remove_unnecessary_words(expanded_phrases)
    
    # Calculate threshold for the source prompt
    threshold = calculate_clip_similarity(image, p_src) * threshold_multiplier

    # Calculate CLIP similarity for each edited phrase
    edited_phrase_similarities = {}
    for phrase in cleaned_phrases:
        similarity = calculate_clip_similarity(image, phrase)
        edited_phrase_similarities[phrase] = similarity

    # Select edited phrases with similarity below the threshold
    selected_edited_phrases = [phrase for phrase, similarity in edited_phrase_similarities.items() if similarity < threshold]

    return selected_edited_phrases

# examples
# image_path = "images/bicycle.jpg"
# p_src = "A photo of a bicycle"
# p_tar = "A photo of a yellow bicycle with a snowy background"

# image_path = "images/two_parrots.jpg"
# p_src = "Two parrots sitting on a branch"
# p_tar = "Two parrots kissing on a branch"

# image_path = "images/white_horse.jpg"
# p_src = "A horse is standing on the ground"
# p_tar = "A zebra is jumping on the ground"

# image_path = "images/fatty-corgi.jpg"
# p_src = "A sitting corgi"
# p_tar = "A corgi jumping high"

# edited_phrases = find_edited_phrases(image_path, p_src, p_tar)
# print("Edited phrases:", edited_phrases)
