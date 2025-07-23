import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
import language_tool_python
import textstat
import tensorflow as tf
from detoxify import Detoxify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gym
from gym import spaces
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSelectionEnv(gym.Env):
    def __init__(self, 
                 all_skills=None, 
                 lstm_model_path="./models/dkt_model_pretrained_384_128.keras",
                 question_types=["Algebra", "Multiple Choice", "Fill in the Blank"],
                 max_seq_len=100,
                 max_steps=200, 
                 device='cpu',
                 w_naturalness=0.22,
                 w_relevance=0.30,
                 w_answerability=0.25,
                 w_difficulty=0.08,
                 w_checklist=0.10,
                 w_bias=0.05):
        super().__init__()
        # Load all_skills from JSON if not provided
        if all_skills is None:
            self.all_skills = ["* () positive reals", "-", "/", "Absolute Value", "Acute", "Addition Whole Numbers", "Addition and Subtraction Fractions", "Addition and Subtraction Integers", "Addition and Subtraction Positive Decimals", "Algebraic Simplification", "Algebraic Solving", "Angles - Obtuse", "Angles on Parallel Lines Cut by a Transversal", "Area Circle", "Area Irregular Figure", "Area Parallelogram", "Area Rectangle", "Area Trapezoid", "Area Triangle", "Box and Whisker", "Calculation with + - * /", "Calculations with Similar Figures", "Choose an Equation from Given Information", "Circle Graph", "Circumference", "Coefficient", "Combinatorics", "Combining Like Terms", "Commutative Property", "Complementary and Supplementary Angles", "Composition of Function Adding", "Compound Interest", "Computation with Real Numbers", "Congruence", "Conversion of Fraction Decimals Percents", "Counting Methods", "D.4.8-understanding-concept-of-probabilities", "Definition Pi", "Distributive Property", "Divisibility Rules", "Division Fractions", "Division Whole Numbers", "Effect of Changing Dimensions of a Shape Prportionally", "English and Metric Terminology", "Equal As Balance Concept", "Equation Solving More Than Two Steps", "Equation Solving Two or Fewer Steps", "Equivalent Fractions", "Estimation", "Expanded", "Exponent", "Exponents", "Factoring Trinomials", "Finding Max and Min from a Quadratic Equation", "Finding Percents", "Finding Ratios", "Finding Slope From Equation", "Finding Slope From Situation", "Finding Slope from Graph", "Finding Slope from Ordered Pairs", "Finding fractions and ratios", "Finding y-intercept from Linear Equation", "Finding y-intercept from Linear Situation", "Fraction Of", "Geometric Definitions", "Graph Shape", "Graphing Inequalities on a number line", "Greatest Common Factor", "Histogram as Table or Graph", "Intercept", "Interior Angles Figures with More than 3 Sides", "Interior Angles Triangle", "Inverse Relations", "Least Common Multiple", "Line Plot", "Line Symmetry", "Line of Best-Fit", "Linear Equations", "Linear area volume conversion", "Mean", "Mean-Median-Mode-Range Differentiation", "Median", "Mode", "Monomial", "Multiplication Fractions", "Multiplication Whole Numbers", "Multiplication and Division Integers", "Multiplication and Division Positive Decimals", "Multiplying Monomials", "Multiplying non Monomial Polynomials", "Nets of 3D Figures", "Number Line", "Order of Operations +", "Order of Operations All", "Ordering Fractions", "Ordering Integers", "Ordering Positive Decimals", "Ordering Real Numbers", "Ordering Whole Numbers", "Parallel and Perpendicular Lines", "Parallel and Perpendicular Slopes", "Parts of a Polyomial", "Pattern Finding", "Percent Discount", "Percent Increase or Decrease", "Percent Of", "Percents", "Perimeter of a Polygon", "Picking Equation and Inequality from Choices", "Point Plotting", "Polynomial Factors", "Prime Number", "Probability of Two Distinct Events", "Probability of a Single Event", "Properties and Classification Quadrilaterals", "Properties and Classification Rectangular Prisms", "Properties and Classification Triangles", "Properties of Numbers", "Proportion", "Pythagorean Theorem", "Quadratic Equation Solving", "Range", "Rate", "Reading a Ruler or Scale", "Recognize Linear Pattern", "Recognizing Equivalent Expressions", "Reflection", "Rotations", "Rounding", "Sampling Techniques", "Scale Factor", "Scatter Plot", "Scientific Notation", "Similar Figures", "Simplifying Expressions positive exponents", "Slope", "Solve Quadratic Equations Using Factoring", "Solving Inequalities", "Solving System of Equation", "Solving Systems of Linear Equations", "Solving for a variable", "Square Root", "Square Roots", "Standard and Word Notation", "Stem and Leaf Plot", "Substitution", "Subtraction Whole Numbers", "Surface Area Cylinder", "Surface Area Rectangular Prism", "Surface Area Sphere", "Surface Area of 3D Objects", "Symbolization", "Table", "Terms", "Transformation", "Translations", "Unit Conversion Standard to Metric", "Unit Conversion Within a System", "Unit Rate", "Variable", "Venn Diagram", "Volume Cone", "Volume Cylinder", "Volume Prism", "Volume Pyramid", "Volume Rectangular Prism", "Volume Sphere", "Volume of 3D Objects", "Write Linear Equation from Graph", "Write Linear Equation from Ordered Pairs", "Write Linear Equation from Situation", "Writine Expression from Diagrams", "X-Y Graph Reading", "and Right"]
        else:
            self.all_skills = all_skills

        self.device = device
        self.max_seq_len = max_seq_len
        self.question_types = question_types
        self.num_skills = len(self.all_skills)
        self.num_question_types = len(self.question_types)
        self.max_steps = max_steps
        self.current_step = 0

        # Reward weights as parameters
        self.w_naturalness = w_naturalness
        self.w_relevance = w_relevance
        self.w_answerability = w_answerability
        self.w_difficulty = w_difficulty
        self.w_checklist = w_checklist
        self.w_bias = w_bias

        # Define action space: [skill_id, question_type_id]
        self.action_space = spaces.MultiDiscrete([self.num_skills, self.num_question_types])

        # student's last performance per skill (vector)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_skills,), dtype=np.float32)

        # Load LSTM student model
        self.student_model = keras.models.load_model(lstm_model_path)

        # Load language model for perplexity (naturalness)
        self.lm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.lm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)
        self.lm_model.eval()

        # Load BERT for semantic similarity
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.skill_embs = self.sbert.encode(self.all_skills) 

        # Grammar checker
        self.grammar_tool = language_tool_python.LanguageTool('en-US')

        # Load T5 for question generation
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        self.t5_model.eval()

        # Detoxify for bias/toxicity detection
        self.detoxify_model = Detoxify('original')

        # Set up student history performance tracking
        self.student_history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5

    def compute_perplexity(self, text):
        inputs = self.lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.lm_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return float(torch.exp(loss).cpu().numpy())

    def compute_naturalness(self, text):
        perp = self.compute_perplexity(text)

        # Alternative scaling, adjust denominator after experiments
        perp_score = np.exp(-perp / 50)  

        matches = self.grammar_tool.check(text)
        grammar_penalty = min(len(matches) / 10, 1.0)  

        # Combine scores
        score = perp_score * (1 - grammar_penalty)

        # Clamp between 0 and 1
        return np.clip(score, 0, 1)


    def compute_relevance(self, question, k=3):
        """
        Compute the average relevance between the question and top-k most semantically similar skills.
        Assumes self.all_skills is a list of strings (skill names).
        """
        try:
            # Get embeddings for all skills and the question
            q_emb = self.sbert.encode([question])[0]
            skill_embs = self.skill_embs

            # Compute cosine similarities
            sims = cosine_similarity(skill_embs, q_emb.reshape(1, -1)).flatten()

            # Get top-k indices
            top_k_indices = np.argsort(sims)[-k:]

            # Compute BERTScore relevance with top-k skill names
            top_skills = [self.all_skills[i] for i in top_k_indices]
            P, R, F1 = bert_score([question] * k, top_skills, lang="en", verbose=False)
            avg_f1 = float(F1.mean())

            return avg_f1
        except Exception as e:
            print(f"Error in compute_relevance for question: {question}\n{e}")
            print("Falling back to cosine similarity.")
            return float(np.mean(sims[top_k_indices]))

    def compute_answerability(self, question, k=3):
        """
        Estimate how answerable a question is based on the student's performance on top-k relevant skills.
        """
        try:
            q_emb = self.sbert.encode([question])[0]
            skill_embs = self.skill_embs
            sims = cosine_similarity(skill_embs, q_emb.reshape(1, -1)).flatten()

            top_k_indices = np.argsort(sims)[-k:]
            weights = sims[top_k_indices]
            weights = weights / (np.sum(weights) + 1e-8)  # Normalize

            performances = [self.get_skill_performance(i) for i in top_k_indices]
            weighted_perf = np.dot(weights, performances)

            return weighted_perf
        except Exception as e:
            print(f"Error in compute_answerability: {e}")
            return 0.5

    def compute_difficulty(self, question, k=3):
        """
        Compute difficulty as the difference between linguistic complexity and student's readiness for the top-k relevant skills.
        """
        try:
            # Step 1: Get linguistic difficulty (Flesch-Kincaid grade, normalized)
            grade = textstat.flesch_kincaid_grade(question)
            linguistic_difficulty = np.clip((grade - 1) / 11, 0, 1)

            # Step 2: Get question embedding
            q_emb = self.sbert.encode([question])[0]
            skill_embs = self.skill_embs

            # Step 3: Compute cosine similarity to all skills
            sims = cosine_similarity(skill_embs, q_emb.reshape(1, -1)).flatten()

            # Step 4: Top-k most related skills
            top_k_indices = np.argsort(sims)[-k:]
            performances = [self.get_skill_performance(i) for i in top_k_indices]
            avg_perf = np.mean(performances)

            # Step 5: Difficulty = mismatch between question complexity and student readiness
            mismatch = np.clip(linguistic_difficulty - avg_perf, 0, 1)
            return mismatch
        except Exception as e:
            print(f"Error computing difficulty for question: {question} — {e}")
            return 0.5  # Neutral default


    def get_student_performance(self, new_skill_id=None, new_question_type_id=None):
        """
        Predicts the student's performance using the LSTM student model.
        Stores and updates the student's interaction history for use as LSTM input.
        Decodes the skill_id to the same encoding used as LSTM input.
        
        Args:
            new_skill_id: If provided, adds this skill to the sequence for prediction
            new_question_type_id: If provided, adds this question type to the sequence for prediction
        """

        # Prepare the latest sequence for prediction (pad/truncate as needed)
        # Assume self.student_history is a list of (skill_id, correct, question_type_id)
        max_seq_len = self.max_seq_len  # or use the same as in training
        num_skills = len(self.all_skills)

        if len(self.student_history) == 0 and new_skill_id is None:
            # If no history and no new skill, return neutral performance for all skills
            return np.ones(num_skills, dtype=np.float32) * 0.5

        # Start with existing history
        history_to_encode = self.student_history.copy()
        
        # Add new skill/question type if provided
        if new_skill_id is not None and new_question_type_id is not None:
            # Use current performance for this skill as assumption
            if hasattr(self, 'student_performance') and self.student_performance is not None:
                assumed_correct = self.student_performance[new_skill_id] 
            
            history_to_encode.append((new_skill_id, assumed_correct, new_question_type_id))

        if len(history_to_encode) == 0:
            return np.ones(num_skills, dtype=np.float32) * 0.5

        # Encode history as in DKT: x_encoded = skill + correct * num_skills + qtype * num_skills * 2
        x_seq = [
            s + c * num_skills + q * num_skills * 2
            for (s, c, q) in history_to_encode
        ]
        
        # Pad sequence
        X_input = pad_sequences([x_seq], padding='post', maxlen=max_seq_len)

        # Predict with LSTM model
        y_pred = self.student_model.predict({'skill_input': X_input}, verbose=0)

        # Get the last non-padded timestep
        last_idx = min(len(x_seq)-1, max_seq_len-1)
        self.student_performance = y_pred[0, last_idx, :]  # Store latest performance for all skills
            
        return self.student_performance
    
    def get_skill_performance(self, skill_id):
        """
        Get the student's performance for a specific skill.
        This is a wrapper around get_skill_performance for clarity.
        """
        return self.student_performance[skill_id]

    def update_student_history(self, skill_id, correct, question_type_id):
        """
        Update the student's interaction history with the latest (skill_id, correct, question_type_id).
        """
        self.student_history.append((skill_id, correct, question_type_id))

    def compute_bias_score(self, question):
        """
        Uses Detoxify to score toxicity/bias. Returns a penalty (0 to 1, higher = more bias/toxicity).
        """
        relevant_keys = [
            "toxicity",
            "severe_toxicity",
            "identity_attack",
            "insult",
            "threat",
            "obscene",
        ]
        try:
            scores = self.detoxify_model.predict(question)
            bias_penalty = max(scores.get(k, 0) for k in relevant_keys)
            #Apply scaling or thresholding 
            bias_penalty = min(1.0, bias_penalty * 1.1)  # example scaling
        except Exception:
            bias_penalty = 0.0
        return bias_penalty


    def checklist_coverage(self, question, weak_skill_threshold=0.3):
        """
        Analyzes student performance gaps across all skills and checks if the question
        addresses the most needed skills based on performance history using SBERT similarity.
        """
        if not hasattr(self, "student_history") or not self.student_history:
            return 1.0
        
        # Calculate performance for each skill
        skill_performances = {}
        for skill_id in range(self.num_skills):
            skill_performances[skill_id] = self.get_skill_performance(skill_id)
        
        # Identify the bottom X% of skills (most needed) based on threshold
        sorted_skills = sorted(skill_performances.items(), key=lambda x: x[1])
        bottom_percent = int(weak_skill_threshold * len(sorted_skills))
        weak_skills = [skill_id for skill_id, _ in sorted_skills[:bottom_percent]]
        
        # Use SBERT to find similarity between question and weak skills
        if not weak_skills:
            return 1.0
        
        # Get embeddings for the question and weak skills
        question_embedding = self.sbert.encode([question])[0]
        weak_skill_names = [self.all_skills[skill_id] for skill_id in weak_skills]
        weak_skill_embeddings = self.sbert.encode(weak_skill_names)
        
        # Calculate cosine similarities
        similarities = np.dot(weak_skill_embeddings, question_embedding) / (
            np.linalg.norm(weak_skill_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
        )
        
        # Weight similarities by how weak each skill is (lower performance = higher weight)
        weighted_similarities = []
        for i, skill_id in enumerate(weak_skills):
            weakness_weight = 1.0 - skill_performances[skill_id]  # Higher weight for weaker skills
            weighted_similarity = similarities[i] * weakness_weight
            weighted_similarities.append(weighted_similarity)
        
        # Calculate coverage score based on weighted average of similarities
        if weighted_similarities:
            # Average weighted similarity to weak skills
            avg_weighted_similarity = np.mean(weighted_similarities)
            # Scale to 0.5-1.0 range (0.5 base + 0.5 * similarity)
            coverage_score = 0.5 + 0.5 * avg_weighted_similarity
        else:
            coverage_score = 0.5
        
        return min(1.0, max(0.0, coverage_score))


    def calculate_reward(self, naturalness, relevance, answerability, difficulty, bias_penalty, checklist_score):
        difficulty_score = 1.0 - difficulty  
        reward = (
            self.w_naturalness * naturalness +
            self.w_relevance * relevance +
            self.w_answerability * answerability +
            self.w_difficulty * difficulty_score +
            self.w_checklist * checklist_score -
            self.w_bias * bias_penalty  # Bias is a penalty, so subtract
        )
        return reward

    def generate_question(self, skill, question_type="open", student_perf=None, max_length=32):
        """
        Generate a question using T5, conditioned on skill, question type, and student performance.
        """

        prompt = f"Generate a {question_type} question about {skill} for a student with performance {student_perf:.2f} regarding the following skills {self.all_skills}."
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )
        question = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

    def step(self, action):
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
        
        student_performance = self.get_student_performance(skill_id, question_type_id)

        # Generate question based on action
        question = self.generate_question(skill, question_type, student_performance)

        naturalness = self.compute_naturalness(question)
        relevance = self.compute_relevance(question, skill)
        answerability = self.compute_answerability(question, skill)
        difficulty = self.compute_difficulty(question)
        bias_penalty = self.compute_bias_score(question)
        checklist_score = self.checklist_coverage(question, skill)

        reward = self.calculate_reward(
            naturalness, relevance, answerability, difficulty, bias_penalty, checklist_score
        )

        #  bservation: student's predicted performance for all skills
        observation = np.array([student_performance], dtype=np.float32)

        info = {
            "naturalness": naturalness,
            "relevance": relevance,
            "answerability": answerability,
            "difficulty": difficulty,
            "bias_penalty": bias_penalty,
            "checklist_score": checklist_score,
            "question": question,
            "skill": skill,
            # "skill_id": skill_id,
            "question_type": question_type,
            # "question_type_id": question_type_id,
            "student_performance": student_performance      
        }

        self.current_step += 1
        done = self.current_step >= self.max_steps  # <-- done when max_steps reached

        print('Step ' + self.current_step + ' Complete')
        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment and student history.
        Returns initial observation.
        """
        self.student_history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5
        self.current_step = 0 
    
        # return neutral performance for all skills
        return np.ones(self.num_skills, dtype=np.float32) * 0.5