from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/data/coursework_dataset

# importing libraries
import json
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")



# Loading data files
with open('train.json', 'r') as train_file:
    train_data = json.load(train_file)

with open('test.json', 'r') as test_file:
    test_data = json.load(test_file)

with open('val.json', 'r') as validation_file:
    validation_data = json.load(validation_file)

# Converting Json to Pandas DataFrame
train_dataset = pd.DataFrame(train_data)
test_dataset = pd.DataFrame(test_data)
validation_dataset = pd.DataFrame(validation_data)

# Making use of 'text_pipeline_spacy' from lab 3


def text_pipeline_spacy_special(text):
    tokens = []
    doc = nlp(text)
    for t in doc:
        if not t.is_punct and not t.is_space:
            tokens.append(t.text.lower())
    return tokens



# Splitting tokens into question,option and correct_index

# Training Data
train_questions_tokens = [text_pipeline_spacy_special(question) for question in train_dataset['question']]
train_options_tokens = [text_pipeline_spacy_special(str(option)) for options in train_dataset['options'] for option in options]
train_correct_index_tokens = [text_pipeline_spacy_special(str(correct_index)) for correct_index in train_dataset['correct_index']]


# Testing Data
test_questions_tokens = [text_pipeline_spacy_special(question) for question in test_dataset['question']]
test_options_tokens = [text_pipeline_spacy_special(str(option)) for options in test_dataset['options'] for option in options]
test_correct_index_tokens = [text_pipeline_spacy_special(str(correct_index)) for correct_index in test_dataset['correct_index']]


# Validation Data
validation_questions_tokens = [text_pipeline_spacy_special(question) for question in validation_dataset['question']]
validation_options_tokens = [text_pipeline_spacy_special(str(option)) for options in validation_dataset['options'] for option in options]
validation_correct_index_tokens = [text_pipeline_spacy_special(str(correct_index)) for correct_index in validation_dataset['correct_index']]

# Number of questions, options, and correct_index in each split after tokenization
num_train_questions = len(train_questions_tokens)
num_train_options = len(train_options_tokens)
num_train_correct_index = len(train_correct_index_tokens)
print('Total number of question and options in training dataset:')
print('Questions - ' ,num_train_questions)
print('Options - ',num_train_options)
print('Correct_Index - ' , num_train_correct_index)


num_test_questions = len(test_questions_tokens)
num_test_options = len(test_options_tokens)
num_test_correct_index = len(test_correct_index_tokens)
print('Total number of question and options in testing dataset:')
print('Questions - ' ,num_test_questions)
print('Options - ',num_test_options)
print('Correct_Index - ' , num_test_correct_index)

num_validation_questions = len(validation_questions_tokens)
num_validation_options = len(validation_options_tokens)
num_validation_correct_index = len(validation_correct_index_tokens)
print('Total number of question and options in validation dataset:')
print('Questions - ' ,num_validation_questions)
print('Options - ',num_validation_options)
print('Correct_Index - ' , num_test_correct_index)


# Average questions in train dataset
total_tokens_train = sum(len(tokens) for tokens in train_questions_tokens)
avg_train_dataset_question = total_tokens_train / num_train_questions
print('Average number of questions in training dataset:',avg_train_dataset_question)

# Average options in train dataset
total_tokens_options_train = sum(len(tokens) for tokens in train_options_tokens)
avg_train_dataset_option = total_tokens_options_train / num_train_options
print('Average number of options in training dataset:',avg_train_dataset_option)

# Average of correct index in train dataset

total_tokens_correct_choices_train = sum(len(tokens) for tokens in train_correct_index_tokens)
average_tokens_per_correct_choice_train = total_tokens_correct_choices_train / len(train_correct_index_tokens)
print('Average number of correct_index in training dataset:',average_tokens_per_correct_choice_train)

import matplotlib.pyplot as plt

# Distribution of question lengths
question_lengths = [len(text_pipeline_spacy_special(item["question"])) for item in train_data]

# Distribution of option lengths
option_lengths = [len(text_pipeline_spacy_special(option)) for item in train_data for option in item["options"]]
plt.figure(figsize=(12, 5))

# Plotting the distribution of question lengths
plt.subplot(1, 2, 1)
plt.hist(question_lengths, bins=50, color='red')
plt.title('Distribution of Question Lengths in Training Set')
plt.xlabel('Number of Tokens')
plt.ylabel('Number of Questions')

# Plotting the distribution of option lengths
plt.subplot(1, 2, 2)
plt.hist(option_lengths, bins=50, color='green')
plt.title('Distribution of Option Lengths in Training Set')
plt.xlabel('Number of Tokens')
plt.ylabel('Number of Options')

# Adjust layout
plt.tight_layout()
plt.show()


# Similarity code checks

# Overlap Coefficient
def overlap_coefficient_check(token_X, token_Y):
    intersection = len(token_X.intersection(token_Y))
    min_len = min(len(token_X), len(token_Y))
    if min_len == 0:
        return 0
    else:
        return intersection / min_len

# Sorensen Dice
def sorensen_dice_similarity_check(token_X, token_Y):
    intersection = len(token_X.intersection(token_Y))
    total = len(token_X) + len(token_Y)
    if total == 0:
        return 0
    else:
        return (2 * intersection) / total

# Jaccard
def jaccard_similarity_check(token_X, token_Y):
    intersection = len(token_X.intersection(token_Y))
    union = len(token_X.union(token_Y))
    if union == 0:
        return 0
    else:
        return intersection / union


# Choosing the best answer out of three
def best_answer(question, options, sim_check):
    tokenized_question = set(text_pipeline_spacy_special(question))
    scores = [sim_check(tokenized_question, set(text_pipeline_spacy_special(option))) for option in options]
    max_score = max(scores)
    best_option_index = scores.index(max_score)
    return best_option_index, scores

# Code for evaluating performance
def check_performance(data, sim_check):
    correct_answers = 0
    total_questions = len(data)
    ties_count = 0

    for item in data:
        best_option_index, scores = best_answer(item['question'], item['options'], sim_check)
        if scores.count(max(scores)) > 1:  # Check for ties
            ties_count += 1
        if best_option_index == item['correct_index']:
            correct_answers += 1

    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    return accuracy, ties_count

# Calculate and print accuracy for each similarity measure on the training set
accuracy_jaccard_train, ties_jaccard_train = check_performance(train_data, jaccard_similarity_check)
accuracy_dice_train, ties_dice_train = check_performance(train_data, sorensen_dice_similarity_check)
accuracy_overlap_train, ties_overlap_train = check_performance(train_data, overlap_coefficient_check)

print("Training Set Performance:")
print('Overlap Coefficient: Accuracy =', accuracy_overlap_train, 'and Ties =' , ties_overlap_train)
print('Sorensen-Dice Similarity: Accuracy =', accuracy_dice_train, 'and Ties = ',ties_dice_train)
print('Jaccard Similarity: Accuracy = ', accuracy_jaccard_train, 'and Ties = ' , ties_jaccard_train)


# Calculate and print accuracy for each similarity measure on the validation set
accuracy_jaccard_val, ties_jaccard_val = check_performance(validation_data, jaccard_similarity_check)
accuracy_dice_val, ties_dice_val = check_performance(validation_data, sorensen_dice_similarity_check)
accuracy_overlap_val, ties_overlap_val = check_performance(validation_data, overlap_coefficient_check)

print("\nValidation Set Performance:")
print('Overlap Coefficient: Accuracy =', accuracy_overlap_val, 'and Ties =' , ties_overlap_val)
print('Sorensen-Dice Similarity: Accuracy =', accuracy_dice_val, 'and Ties = ',ties_dice_val)
print('Jaccard Similarity: Accuracy = ', accuracy_jaccard_val, 'and Ties = ' , ties_jaccard_val)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Generate TF vectors for questions and answers
def generate_tf_vectors(dataset):
    corp = [item['question'] for item in dataset] + [option for item in dataset for option in item['options']]
    vectorizer_check = CountVectorizer(tokenizer=text_pipeline_spacy_special)
    tfvectors = vectorizer_check.fit_transform(corp)
    return tfvectors[:len(dataset)], tfvectors[len(dataset):]

# Code for calculating cosine similarity and best answer
def calculate_cosine_similarity_and_best_answer(data, tf_question_vectors, tf_answer_vectors):
    correct_predictions = 0
    for i, item in enumerate(data):
        question_vector = tf_question_vectors[i]
        answer_vectors = tf_answer_vectors[i * 4 : (i + 1) * 4]
        max_similarity_score = -1
        selected_answer_index = -1
        for ij, answer_vector in enumerate(answer_vectors):
            similarity_score = cosine_similarity(question_vector, answer_vector)[0][0]
            if similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                selected_answer_index = ij
        # Check if the selected answer is correct
        if selected_answer_index == item['correct_index']:
            correct_predictions += 1
    accuracy = correct_predictions / len(data)
    return accuracy

# Generating TF vectors and Calculate cosine similarity and accuracy for training and validation sets
train_question_vectors_tf, train_answer_vectors_tf = generate_tf_vectors(train_data)
valid_question_vectors_tf, valid_answer_vectors_tf = generate_tf_vectors(validation_data)
cosine_similar_accuracy_train = calculate_cosine_similarity_and_best_answer(train_data, train_question_vectors_tf, train_answer_vectors_tf)
cosine_similar_accuracy_val = calculate_cosine_similarity_and_best_answer(validation_data, valid_question_vectors_tf, valid_answer_vectors_tf)

# Output the results
print("Training Set:")
print("Cosine Similarity Accuracy:", cosine_similar_accuracy_train)
print("\nValidation Set:")
print("Cosine Similarity Accuracy:", cosine_similar_accuracy_val)

from sklearn.feature_extraction.text import TfidfVectorizer

# Code for generating TF-IDF vectors
def generate_tfidf_vectors(dataset):
    corp = [item['question'] for item in dataset] + [option for item in dataset for option in item['options']]
    vectorizer = TfidfVectorizer(tokenizer=text_pipeline_spacy_special)
    tfidf_vectors = vectorizer.fit_transform(corp)
    return tfidf_vectors[:len(dataset)], tfidf_vectors[len(dataset):]

# Calculate cosine similarity and accuracy for training and validation sets with TF-IDF weighting
train_question_vectors_tfidf, train_answer_vectors_tfidf = generate_tfidf_vectors(train_data)
valid_question_vectors_tfidf, valid_answer_vectors_tfidf = generate_tfidf_vectors(validation_data)
train_cosine_similarity_accuracy_tfidf = calculate_cosine_similarity_and_best_answer(train_data, train_question_vectors_tfidf, train_answer_vectors_tfidf)
valid_cosine_similarity_accuracy_tfidf = calculate_cosine_similarity_and_best_answer(validation_data, valid_question_vectors_tfidf, valid_answer_vectors_tfidf)

# Output the results
print("Training Set with tfidf:")
print("Cosine Similarity Accuracy:", train_cosine_similarity_accuracy_tfidf)
print("\nValidation Set with tfidf:")
print("Cosine Similarity Accuracy:", valid_cosine_similarity_accuracy_tfidf)

from transformers import BertTokenizer, BertModel
from transformers.models.bert.tokenization_bert import BasicTokenizer
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Extracting context vectors
def context_vector_extraction(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs[0][:, 0, :]
    return cls_embeddings

# Function to calculate cosine similarity and select the most similar answer
def calculate_cosine_similarity_and_best_answer(data):
    predictions = 0


    for item in data:
        question_vector = context_vector_extraction(item['question'])
        max_similar_score = -1
        selected_answer_index = -1
        for i, option in enumerate(item['options']):
            answer_vector = context_vector_extraction(option)

            similar_score = cosine_similarity(question_vector, answer_vector)[0][0]
            if similar_score > max_similar_score:
                max_similar_score = similar_score
                selected_answer_index = i

        # Correct answer check
        if selected_answer_index == item['correct_index']:
            predictions += 1
    accurate_score = predictions / len(data)
    return accurate_score

# Calculate cosine similarity and accuracy for training and validation sets
cosine_similar_accuracy_train = calculate_cosine_similarity_and_best_answer(train_data)
cosine_similar_accuracy_val = calculate_cosine_similarity_and_best_answer(validation_data)

# Output the results
print("Training Set:")
print("Cosine Similarity Accuracy:", cosine_similar_accuracy_train)

print("\nValidation Set:")
print("Cosine Similarity Accuracy:", cosine_similar_accuracy_val)

from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Given hyperparameters
learning_rate = 1e-5
epochs = 4
weight_decay = 0

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define dataset class
class DatasetOfQuestionAndOption(Dataset):
    def __init__(self, dataset):
        self.data = []
        for item in dataset:
            question = item['question']
            options = item['options']
            correct_index = item['correct_index']
            for i, option in enumerate(options):
                input_text = f"{question} [SEP] {option}"
                label = 1 if i == correct_index else 0
                self.data.append((input_text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create datasets and dataloaders
train_dataset = DatasetOfQuestionAndOption(train_data)
valid_dataset = DatasetOfQuestionAndOption(validation_data)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# Define training function
def model_training(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)
        labels = torch.tensor(batch[1]).to(device)
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)

# Define evaluation function
def evaluating_perform(model, valid_loader, device):
    model.eval()
    labels_used = []
    preds_used = []
    with torch.no_grad():
        for batch in valid_loader:
            inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.tensor(batch[1]).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels_used.extend(labels.cpu().numpy())
            preds_used.extend(preds.cpu().numpy())
    accuracy = accuracy_score(labels_used, preds_used)
    precision = precision_score(labels_used, preds_used)
    recall = recall_score(labels_used, preds_used)
    f1 = f1_score(labels_used, preds_used)
    return accuracy, precision, recall, f1


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(epochs):
    train_loss = model_training(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

# Evaluate on the validation set
accuracy, precision, recall, f1 = evaluating_perform(model, valid_loader, device)
# After the evaluation on the validation set
print("Evaluation Results on Validation Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

import pandas as pd

def optimal_answer_selection(model, loader, device):
    model.eval()
    correct_predictions = 0
    total_questions = 0
    with torch.no_grad():
        for batch in loader:
            inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = torch.tensor(batch[1]).to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_questions += labels.size(0)
    accuracy = correct_predictions / total_questions
    return accuracy

# Calculate accuracy for selecting the correct answer on the training set
train_selecting_correct_answer = optimal_answer_selection(model, train_loader, device)

# Calculate accuracy for selecting the correct answer on the validation set
valid_selecting_correct_answer = optimal_answer_selection(model, valid_loader, device)

print("Accuracy of Selecting Correct Answer:")
print(f"Training Set: {train_selecting_correct_answer:.4f}")
print(f"Validation Set: {valid_selecting_correct_answer:.4f}")
