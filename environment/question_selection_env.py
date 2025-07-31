import numpy as np
import torch
import json
from transformers import AutoTokenizer, GenerationConfig, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM
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
                 w_bias=0.05,
                 top_k=3):
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
        self.student_model = tf.keras.models.load_model(lstm_model_path)

        self.top_k = top_k  # Store top_k globally
        self.top_k_indices = None  # Will be set after question generation

        # Load BERT for semantic similarity
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.skill_embs = self.sbert.encode(self.all_skills) 

        # Grammar checker
        # self.grammar_tool = language_tool_python.LanguageTool('en-US')

        # Load T5 for question generation (valhalla/t5-base-qg-hl)
        self.t5_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl").to(self.device)
        self.t5_model.eval()

        # Detoxify for bias/toxicity detection
        self.detoxify_model = Detoxify('original')

        # Load DeepSeek Math 7B RL for question generation
        self.deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-rl")
        self.deepseek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-rl").to(self.device)
        self.deepseek_model.generation_config = GenerationConfig.from_pretrained("deepseek-ai/deepseek-math-7b-rl")
        self.deepseek_model.generation_config.pad_token_id = self.deepseek_model.generation_config.eos_token_id
        
        self.action_history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5
        self.student_performance_history = []

        self.current_question_embedding = None 
        self.questions_history = []  # Store all generated questions
       

    def compute_difficulty(self, question, k=3):
        """
        Compute difficulty as the difference between linguistic complexity and student's readiness for the top-k relevant skills.
        """
        try:
            # Step 1: Get linguistic difficulty (Flesch-Kincaid grade, normalized)
            grade = textstat.flesch_kincaid_grade(question)
            linguistic_difficulty = np.clip((grade - 1) / 11, 0, 1)

            # Step 2: Get question embedding
            q_emb = self.current_question_embedding
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
        # Assume self.action_history is a list of (skill_id, correct, question_type_id)
        max_seq_len = self.max_seq_len  # or use the same as in training
        num_skills = len(self.all_skills)

        if len(self.action_history) == 0 and new_skill_id is None:
            # If no history and no new skill, return neutral performance for all skills
            return np.ones(num_skills, dtype=np.float32) * 0.5

        # Start with existing history
        history_to_encode = self.action_history.copy()
        
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

    def update_action_history(self, skill_id, correct, question_type_id):
        """
        Update the student's interaction history with the latest (skill_id, correct, question_type_id).
        """
        self.action_history.append((skill_id, correct, question_type_id))

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


    def checklist_coverage(self, weak_skill_threshold=0.6):
        """
        Analyzes student performance gaps across all skills and checks if the question
        addresses the most needed skills based on performance history using SBERT similarity.
        """
        if not hasattr(self, "action_history") or not self.action_history:
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
        question_embedding = self.current_question_embedding
        weak_skill_names = [self.all_skills[skill_id] for skill_id in weak_skills]
        weak_skill_embeddings = self.sbert.encode(weak_skill_names)
        
        # Calculate cosine similarities using sklearn
        similarities = cosine_similarity(weak_skill_embeddings, question_embedding.reshape(1, -1)).flatten()
        
        # Weight similarities by how weak each skill is (lower performance = higher weight)
        weighted_similarities = []
        for i, skill_id in enumerate(weak_skills):
            weakness_weight = 1.0 - skill_performances[skill_id]  # Higher weight for weaker skills
            weighted_similarity = similarities[i] * weakness_weight
            weighted_similarities.append(weighted_similarity)
        
        # Calculate coverage score based on weighted average of similarities
        if weighted_similarities:
            avg_weighted_similarity = np.mean(weighted_similarities)
            coverage_score = 0.5 + 0.5 * avg_weighted_similarity
        else:
            coverage_score = 0.5
        
        return min(1.0, max(0.0, coverage_score))


    def calculate_reward(self, answerability, bias_penalty, checklist_score):

        if len(self.student_performance_history) > 1:
            improvement_reward = float(np.mean(self.student_performance_history[-1])) - float(np.mean(self.student_performance_history[-2])) 
        else:
            improvement_reward = 0.0  # No improvement on first step
    
        reward = (
            # self.w_naturalness * naturalness +
            # self.w_relevance * relevance +
            improvement_reward +
            self.w_answerability * answerability +
            # self.w_difficulty * difficulty_score +
            self.w_checklist * checklist_score -
            self.w_bias * bias_penalty  # Bias is a penalty, so subtract
        )
        return reward

    def generate_question(self, skill, question_type="open", max_length=100):
        """
        Generate a question using DeepSeek Math 7B RL, conditioned on skill and question type.
        Uses the official chat template for best results.
        """
        # Compose the prompt as recommended by DeepSeekMath
        prompt = (
            f"Generate a {question_type} question about {skill}.\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        input_tensor = self.deepseek_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        input_tensor = input_tensor.to(self.deepseek_model.device)
        with torch.no_grad():
            outputs = self.deepseek_model.generate(
                input_tensor, max_new_tokens=max_length
            )
        # Only decode the newly generated tokens
        result = self.deepseek_tokenizer.decode(
            outputs[0][input_tensor.shape[1]:], skip_special_tokens=True
        )
        question = result.strip()
        self.current_question_embedding = self.sbert.encode([question])[0]

        # Compute cosine similarities and store top_k_indices
        sims = cosine_similarity(self.skill_embs, self.current_question_embedding.reshape(1, -1)).flatten()
        self.top_k_indices = np.argsort(sims)[-self.top_k:]
        top_skills = [self.all_skills[i] for i in self.top_k_indices]

        # Predicted correctness for this skill
        predicted_correctness = float(self.student_performance[self.all_skills.index(skill)]) \
            if skill in self.all_skills else None

        # Initialize question_history if not present
        if not hasattr(self, "questions_history"):
            self.question_history = []

        self.question_history.append({
            "question": question,
            "skill": skill,
            "predicted_correctness_for_skill": predicted_correctness,
            "question_type": question_type,
            "top_related_skills": top_skills
        })
        return question

    def compute_relevance(self, question):
        """
        Compute the average relevance between the question and top-k most semantically similar skills.
        Uses self.top_k_indices set during question generation.
        """
        try:
            top_skills = [self.all_skills[i] for i in self.top_k_indices]
            P, R, F1 = bert_score([question] * self.top_k, top_skills, lang="en", verbose=False)
            avg_f1 = float(F1.mean())
            return avg_f1
        except Exception as e:
            print(f"Error in compute_relevance for question: {question}\n{e}")
            print("Falling back to default relevance.")
            return 0.5

    def compute_answerability(self):
        """
        Estimate how answerable a question is based on the student's performance on top-k relevant skills.
        Uses self.top_k_indices set during question generation.
        """
        try:
            performances = [self.get_skill_performance(i) for i in self.top_k_indices]
            weights = np.ones(len(performances)) / (len(performances) + 1e-8)  # Uniform weights
            weighted_perf = np.dot(weights, performances)
            return weighted_perf
        except Exception as e:
            print(f"Error in compute_answerability: {e}")
            return 0.5

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
        question = self.generate_question(skill, question_type)

        # naturalness = self.compute_naturalness(question)
        # relevance = self.compute_relevance(question)
        answerability = self.compute_answerability(question)
        bias_penalty = self.compute_bias_score(question)
        checklist_score = self.checklist_coverage(question)

        reward = self.calculate_reward(
            # naturalness, relevance, 
            answerability, bias_penalty, checklist_score
        )

        # Observation: student's predicted performance for all skills
        observation = np.array([student_performance], dtype=np.float32)

        # Calculate student's average performance across all skills
        avg_student_performance = float(np.mean(student_performance))

        # Store average performance history
        self.student_performance_history.append(student_performance)

        info = {
            # "naturalness": naturalness,
            # "relevance": relevance,
            "answerability": answerability,
            "bias_penalty": bias_penalty,
            "checklist_score": checklist_score,
            "question": question,
            "skill": skill,
            # "skill_id": skill_id,
            "question_type": question_type,
            # "question_type_id": question_type_id,
            "student_avg_performance": avg_student_performance,
        }

        self.current_step += 1
        done = self.current_step >= self.max_steps  # <-- done when max_steps reached

        print(f'Step {self.current_step} Complete')
        print(info)
        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment and student history.
        Returns initial observation.
        """
        self.action_history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5
        self.current_step = 0 
        self.question_history = []
        self.current_question_embedding = None
    
        # Reset student performance history
        self.student_performance_history = []

        # return neutral performance for all skills
        return np.ones(self.num_skills, dtype=np.float32) * 0.5

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self, "action_space"):
            self.action_space.seed(seed)
        if hasattr(self, "observation_space"):
            self.observation_space.seed(seed)
        return [seed]