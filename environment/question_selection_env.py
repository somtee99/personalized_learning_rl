import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import language_tool_python
import textstat
import tensorflow as tf
from detoxify import Detoxify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gym
from gym import spaces

class QuestionSelectionEnv(gym.Env):
    def __init__(self, 
                 course_content, 
                 lstm_model_path, 
                 skill2id, 
                 all_skills, 
                 question_types=None,
                 device='cpu'):
        super().__init__()
        self.course_content = course_content
        self.skill2id = skill2id
        self.all_skills = all_skills
        self.device = device
        self.question_types = question_types or ["Algebra", "Multiple Choice", "Fill in the Blank"]
        self.num_skills = len(self.all_skills)
        self.num_question_types = len(self.question_types)

        # Define action space: [skill_id, question_type_id]
        self.action_space = spaces.MultiDiscrete([self.num_skills, self.num_question_types])

        # Example observation space (can be customized as needed)
        # Here, just a placeholder: student's last performance per skill (vector)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_skills,), dtype=np.float32)

        # Load LSTM student model
        self.student_model = tf.keras.models.load_model(lstm_model_path)

        # Load language model for perplexity (naturalness)
        self.lm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.lm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        self.lm_model.eval()

        # Load BERT for semantic similarity
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # Grammar checker
        self.grammar_tool = language_tool_python.LanguageTool('en-US')

        # Load T5 for question generation
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        self.t5_model.eval()

        # Detoxify for bias/toxicity detection
        self.detoxify_model = Detoxify('original')

        # Example: curriculum checklist (keywords for each skill)
        self.curriculum_keywords = {skill: skill.lower().split() for skill in self.all_skills}

    def compute_perplexity(self, text):
        # Use masked LM negative log-likelihood as a proxy for perplexity
        inputs = self.lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return float(torch.exp(loss).cpu().numpy())

    def compute_naturalness(self, text):
        # Perplexity (lower is better)
        perp = self.compute_perplexity(text)
        perp_score = np.exp(-perp / 50)  # scale to (0,1), tweak denominator as needed

        # Grammar check (penalize if issues found)
        matches = self.grammar_tool.check(text)
        grammar_penalty = min(len(matches) / 10, 1.0)  # up to 1.0 penalty for many errors

        score = perp_score * (1 - grammar_penalty)
        return np.clip(score, 0, 1)

    def compute_relevance(self, question, skill):
        # Compare question to course content for the skill
        content = self.course_content.get(skill, "")
        if not content:
            return 0.0
        # Use BERTScore (or fallback to cosine similarity)
        try:
            P, R, F1 = bert_score([question], [content], lang="en", verbose=False)
            return float(F1[0])
        except Exception:
            # Fallback: cosine similarity
            q_emb = self.sbert.encode([question])[0]
            c_emb = self.sbert.encode([content])[0]
            sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-8)
            return float(sim)

    def compute_answerability(self, question, skill):
        # Simple proxy: check if skill keywords appear in question
        skill_keywords = skill.lower().split()
        q_words = question.lower().split()
        overlap = len(set(skill_keywords) & set(q_words))
        return min(1.0, 0.5 + 0.5 * (overlap > 0))  # 1.0 if any keyword matches, else 0.5

    def compute_difficulty(self, question):
        # Use Flesch-Kincaid grade as a proxy (higher = harder)
        try:
            grade = textstat.flesch_kincaid_grade(question)
            # Normalize: 0 (easy, grade 1) to 1 (hard, grade 12+)
            return np.clip((grade - 1) / 11, 0, 1)
        except Exception:
            return 0.5

    def get_student_performance(self, skill_id):
        """
        Predicts the student's performance for a skill using the LSTM student model.
        Stores and updates the student's interaction history for use as LSTM input.
        Decodes the skill_id to the same encoding used as LSTM input.
        """
        # Initialize history if not present
        if not hasattr(self, "student_history"):
            self.student_history = []

        # Prepare the latest sequence for prediction (pad/truncate as needed)
        # Assume self.student_history is a list of (skill_id, correct, question_type_id)
        max_seq_len = 100  # or use the same as in training
        num_skills = len(self.all_skills)
        num_question_types = len(self.question_types)
        if len(self.student_history) == 0:
            # If no history, return neutral performance
            return 0.5

        # Encode history as in DKT: x_encoded = skill + correct * num_skills + qtype * num_skills * 2
        x_seq = [
            s + c * num_skills + q * num_skills * 2
            for (s, c, q) in self.student_history
        ]
        # Pad sequence
        X_input = pad_sequences([x_seq], padding='post', maxlen=max_seq_len)

        # Dummy extra features (hint_count, attempt_count, overlap_time)
        extra_input = np.zeros((1, max_seq_len, 3), dtype=np.float32)

        # Predict with LSTM model
        y_pred = self.student_model.predict({'skill_input': X_input, 'extra_input': extra_input}, verbose=0)
        # y_pred shape: (1, max_seq_len, num_skills)
        # Get the last non-padded timestep
        last_idx = min(len(x_seq)-1, max_seq_len-1)
        skill_perf = y_pred[0, last_idx, skill_id]
        return float(skill_perf)

    def update_student_history(self, skill_id, correct, question_type_id):
        """
        Update the student's interaction history with the latest (skill_id, correct, question_type_id).
        """
        if not hasattr(self, "student_history"):
            self.student_history = []
        self.student_history.append((skill_id, correct, question_type_id))

        if len(self.student_history) > 100:
            self.student_history = self.student_history[-100:]

    def compute_bias_score(self, question):
        """
        Uses Detoxify to score toxicity/bias. Returns a penalty (0 to 1, higher = more bias/toxicity).
        """
        try:
            scores = self.detoxify_model.predict(question)
            relevant_scores = [scores.get(k, 0) for k in [
                "toxicity", "severe_toxicity", "identity_attack", "insult", "threat", "obscene"]]
            bias_penalty = max(relevant_scores)
        except Exception:
            bias_penalty = 0.0
        return bias_penalty

    def checklist_coverage(self, question, skill):
        """
        Simple checklist: does the question mention all curriculum keywords for the skill?
        Returns 1.0 if all keywords are present, else <1.0.
        """
        keywords = self.curriculum_keywords.get(skill, [])
        q_words = set(question.lower().split())
        if not keywords:
            return 1.0
        covered = sum(1 for k in keywords if k in q_words)
        return covered / len(keywords) if keywords else 1.0

    def calculate_reward(self, naturalness, relevance, answerability, difficulty, bias_penalty, checklist_score, target_difficulty=None):
        w_naturalness = 0.22
        w_relevance = 0.30
        w_answerability = 0.25
        w_difficulty = 0.08
        w_checklist = 0.10
        w_bias = 0.05  # penalty weight

        difficulty_penalty = 0
        if target_difficulty is not None:
            difficulty_penalty = abs(difficulty - target_difficulty)

        reward = (
            w_naturalness * naturalness +
            w_relevance * relevance +
            w_answerability * answerability +
            w_difficulty * (1 - difficulty_penalty) +
            w_checklist * checklist_score -
            w_bias * bias_penalty  # Subtract bias penalty
        )
        return reward

    def generate_question(self, skill, question_type="open", student_perf=None, max_length=32):
        """
        Generate a question using T5, conditioned on skill, question type, and student performance.
        """
        content = self.course_content.get(skill, "")
        if not content:
            return ""
        # You can customize the prompt for T5 based on question_type and student_perf
        prompt = f"Generate a {question_type} question about {skill} for a student with performance {student_perf:.2f}: {content}"
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )
        question = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

    def step(self, action, target_difficulty=None):
        """
        RL step: action is an array [skill_id, question_type_id]
        Returns: observation, reward, done, info
        """
        if not isinstance(action, (list, np.ndarray)) or len(action) != 2:
            raise ValueError("Action must be an array-like of [skill_id, question_type_id]")
        skill_id, question_type_id = int(action[0]), int(action[1])

        if skill_id < 0 or skill_id >= self.num_skills:
            raise ValueError(f"Skill id '{skill_id}' out of range.")
        if question_type_id < 0 or question_type_id >= self.num_question_types:
            raise ValueError(f"Question type id '{question_type_id}' out of range.")

        skill = self.all_skills[skill_id]
        question_type = self.question_types[question_type_id]

        student_perf = self.get_student_performance(skill_id)

        # Generate question based on action
        question = self.generate_question(skill, question_type, student_perf)

        naturalness = self.compute_naturalness(question)
        relevance = self.compute_relevance(question, skill)
        answerability = self.compute_answerability(question, skill)
        difficulty = self.compute_difficulty(question)
        bias_penalty = self.compute_bias_score(question)
        checklist_score = self.checklist_coverage(question, skill)

        reward = self.calculate_reward(
            naturalness, relevance, answerability, difficulty, bias_penalty, checklist_score, target_difficulty
        )

        # Example observation: student's predicted performance for all skills
        observation = np.array([
            self.get_student_performance(i) for i in range(self.num_skills)
        ], dtype=np.float32)

        info = {
            "naturalness": naturalness,
            "relevance": relevance,
            "answerability": answerability,
            "difficulty": difficulty,
            "student_perf": student_perf,
            "skill": skill,
            "skill_id": skill_id,
            "question": question,
            "question_type": question_type,
            "question_type_id": question_type_id,
            "bias_penalty": bias_penalty,
            "checklist_score": checklist_score
        }
        done = False  # Set to True if you want to end the episode (customize as needed)
        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment and student history.
        Returns initial observation.
        """
        self.student_history = []
        # Example: return neutral performance for all skills
        return np.ones(self.num_skills, dtype=np.float32) * 0.5

# Example usage:
# env = QuestionSelectionEnv(course_content, "../models/dkt_model_text_emb.keras", skill2id, all_skills)
# obs = env.reset()
# action = [skill2id["Addition"], env.question_types.index("Multiple Choice")]
# obs, reward, done, info = env.step(action)