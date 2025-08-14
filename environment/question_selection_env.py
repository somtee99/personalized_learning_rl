import numpy as np
import torch
import os
import json
from transformers import AutoTokenizer, GenerationConfig, T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from bert_score import score
import textstat
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gym
from gym import spaces
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSelectionEnv(gym.Env):
    def __init__(self, 
                 questions_df,
                 all_skills=None, 
                 lstm_model_path="./models/dkt_model_pretrained_seq50.keras",
                 question_types=["Algebra", "Multiple Choice", "Fill in the Blank"],
                 max_seq_len=50,
                 max_steps=250, 
                 device='cpu',
                 w_answerability=50,
                 w_improvement=100,
                 w_coverage=0.8,
                 top_k=3,
                 weak_skills_threshold=0.7
                 ):
        super().__init__()
        # Load all_skills from JSON if not provided
        if all_skills is None:
            self.all_skills = [
                "* () positive reals",
                "-",
                "/",
                "Absolute Value",
                "Addition Whole Numbers",
                "Addition and Subtraction Fractions",
                "Addition and Subtraction Integers",
                "Addition and Subtraction Positive Decimals",
                "Algebraic Simplification",
                "Algebraic Solving",
                "Angles - Acute",
                "Angles - Obtuse",
                "Angles - Right",
                "Angles on Parallel Lines Cut by a Transversal",
                "Area Circle",
                "Area Irregular Figure",
                "Area Parallelogram",
                "Area Rectangle",
                "Area Trapezoid",
                "Area Triangle",
                "Box and Whisker",
                "Calculation with + - * /",
                "Calculations with Similar Figures",
                "Choose an Equation from Given Information",
                "Circle Graph",
                "Circumference",
                "Coefficient",
                "Combinatorics",
                "Combining Like Terms",
                "Commutative Property",
                "Complementary and Supplementary Angles",
                "Composition of Function Adding",
                "Compound Interest",
                "Computation with Real Numbers",
                "Congruence",
                "Conversion of Fraction Decimals Percents",
                "Counting Methods",
                "Definition Pi",
                "Distributive Property",
                "Divisibility Rules",
                "Division Fractions",
                "Division Whole Numbers",
                "Effect of Changing Dimensions of a Shape Proportionally",
                "English and Metric Terminology",
                "Equal As Balance Concept",
                "Equation Solving More Than Two Steps",
                "Equation Solving Two or Fewer Steps",
                "Equivalent Fractions",
                "Estimation",
                "Expanded",
                "Exponent",
                "Exponents",
                "Factoring Trinomials",
                "Finding Max and Min from a Quadratic Equation",
                "Finding Percents",
                "Finding Ratios",
                "Finding Slope From Equation",
                "Finding Slope From Situation",
                "Finding Slope from Graph",
                "Finding Slope from Ordered Pairs",
                "Finding fractions and ratios",
                "Finding y-intercept from Linear Equation",
                "Finding y-intercept from Linear Situation",
                "Fraction Of",
                "Geometric Definitions",
                "Graph Shape",
                "Graphing Inequalities on a number line",
                "Greatest Common Factor",
                "Histogram as Table or Graph",
                "Intercept",
                "Interior Angles Figures with More than 3 Sides",
                "Interior Angles Triangle",
                "Inverse Relations",
                "Least Common Multiple",
                "Line Plot",
                "Line Symmetry",
                "Line of Best-Fit",
                "Linear Equations",
                "Linear area volume conversion",
                "Mean",
                "Mean-Median-Mode-Range Differentiation",
                "Median",
                "Mode",
                "Monomial",
                "Multiplication Fractions",
                "Multiplication Whole Numbers",
                "Multiplication and Division Integers",
                "Multiplication and Division Positive Decimals",
                "Multiplying Monomials",
                "Multiplying non Monomial Polynomials",
                "Nets of 3D Figures",
                "Number Line",
                "Order of Operations",
                "Order of Operations All",
                "Ordering Fractions",
                "Ordering Integers",
                "Ordering Positive Decimals",
                "Ordering Real Numbers",
                "Ordering Whole Numbers",
                "Parallel and Perpendicular Lines",
                "Parallel and Perpendicular Slopes",
                "Parts of a Polyomial",
                "Pattern Finding",
                "Percent Discount",
                "Percent Increase or Decrease",
                "Percent Of",
                "Percents",
                "Perimeter of a Polygon",
                "Picking Equation and Inequality from Choices",
                "Point Plotting",
                "Polynomial Factors",
                "Prime Number",
                "Probability of Two Distinct Events",
                "Probability of a Single Event",
                "Properties and Classification Quadrilaterals",
                "Properties and Classification Rectangular Prisms",
                "Properties and Classification Triangles",
                "Properties of Numbers",
                "Proportion",
                "Pythagorean Theorem",
                "Quadratic Equation Solving",
                "Range",
                "Rate",
                "Reading a Ruler or Scale",
                "Recognize Linear Pattern",
                "Recognizing Equivalent Expressions",
                "Reflection",
                "Rotations",
                "Rounding",
                "Sampling Techniques",
                "Scale Factor",
                "Scatter Plot",
                "Scientific Notation",
                "Similar Figures",
                "Simplifying Expressions positive exponents",
                "Slope",
                "Solve Quadratic Equations Using Factoring",
                "Solving Inequalities",
                "Solving System of Equation",
                "Solving Systems of Linear Equations",
                "Solving for a variable",
                "Square Root",
                "Square Roots",
                "Standard and Word Notation",
                "Stem and Leaf Plot",
                "Substitution",
                "Subtraction Whole Numbers",
                "Surface Area Cylinder",
                "Surface Area Rectangular Prism",
                "Surface Area Sphere",
                "Surface Area of 3D Objects",
                "Symbolization",
                "Table",
                "Terms",
                "Transformation",
                "Translations",
                "Understanding Concept of Probabilities",
                "Unit Conversion Standard to Metric",
                "Unit Conversion Within a System",
                "Unit Rate",
                "Variable",
                "Venn Diagram",
                "Volume Cone",
                "Volume Cylinder",
                "Volume Prism",
                "Volume Pyramid",
                "Volume Rectangular Prism",
                "Volume Sphere",
                "Volume of 3D Objects",
                "Write Linear Equation from Graph",
                "Write Linear Equation from Ordered Pairs",
                "Write Linear Equation from Situation",
                "Writing Expression from Diagrams",
                "X-Y Graph Reading"
            ]
        else:
            self.all_skills = all_skills

        self.device = device
        self.max_seq_len = max_seq_len
        self.weak_skills_threshold = weak_skills_threshold
        self.question_types = question_types
        self.num_skills = len(self.all_skills)
        self.num_question_types = len(self.question_types)
        self.max_steps = max_steps
        self.current_step = 0

        # Reward weights as parameters
        self.w_answerability = w_answerability
        self.w_improvement = w_improvement
        self.w_coverage = w_coverage

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
        
        # Load questions
        self.questions_df = questions_df

        self.history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5
        self.student_performance_history = []

        self.current_question_embedding = None 


    def predict_student_performance(self, new_skill_id=None, new_question_type_id=None):
        """
        Predicts the student's performance using the LSTM student model.
        Stores and updates the student's interaction history for use as LSTM input.
        Decodes the skill_id to the same encoding used as LSTM input.
        
        Args:
            new_skill_id: If provided, adds this skill to the sequence for prediction
            new_question_type_id: If provided, adds this question type to the sequence for prediction
        """

        # Prepare the latest sequence for prediction (pad/truncate as needed)
        max_seq_len = self.max_seq_len  # or use the same as in training
        num_skills = self.num_skills

        if len(self.history) == 0 and new_skill_id is None:
            # If no history and no new skill, return neutral performance for all skills
            return np.ones(num_skills, dtype=np.float32) * 0.5

        # Only use relevant fields for encoding
        history_to_encode = [
            (h["skill_id"], h["predicted_correctness_for_skill"], h["question_type_id"])
            for h in self.history
            if "skill_id" in h and "predicted_correctness_for_skill" in h and "question_type_id" in h
        ]
        
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

    def update_history(self, info):
        """
        Update the student's interaction history.
        """
        self.history.append(info)


    def weak_skill_coverage(self):
        """
        Analyzes student performance gaps across all skills and checks if the question
        addresses the most needed skills based on performance history using SBERT similarity.
        """
        if not hasattr(self, "history") or not self.history:
            return 1.0
        
        # Calculate performance for each skill
        skill_performances = {}
        for skill_id in range(self.num_skills):
            skill_performances[skill_id] = self.get_skill_performance(skill_id)
        
        # Identify the bottom X% of skills (most needed) based on threshold
        sorted_skills = sorted(skill_performances.items(), key=lambda x: x[1])
        bottom_percent = int(self.weak_skills_threshold * len(sorted_skills))
        weak_skills = [skill_id for skill_id, _ in sorted_skills[:bottom_percent]]
        
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
        
          # Calculate cosine similarities
        similarities = cosine_similarity(weak_skill_embeddings, question_embedding.reshape(1, -1)).flatten()

        # Use average similarity as alignment score, normalized to [0, 1]
        if len(similarities) > 0:
            alignment_score = np.clip(np.mean(similarities), 0, 1)
        else:
            alignment_score = 0.5  # Neutral if no weak skills

        return alignment_score


    def calculate_reward(self, improvement, answerability, coverage):
        improvement = self.w_improvement * improvement
        answerability = self.w_answerability * answerability
        coverage = self.w_coverage * coverage

        reward = (
            improvement + #tanh (-1 to 1)
            answerability + #sigmoid (0-1)
            coverage 
        )
        return reward

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

    def get_question_from_bank(self, skill, question_type):
        # Filter for matching skill and question_type
        matches = self.questions_df[
            (self.questions_df['skill'] == skill) &
            (self.questions_df['question_type'] == question_type)
        ]
        if matches.empty:
            # Fallback: pick any question with the skill
            matches = self.questions_df[self.questions_df['skill'] == skill]
        if matches.empty:
            # Fallback: pick any question with the question_type
            matches = self.questions_df[self.questions_df['question_type'] == question_type]
        if matches.empty:
            # Fallback: pick any question
            matches = self.questions_df
        # Randomly select one
        question_row = matches.sample(1).iloc[0]
        question = question_row['question_text']

        self.current_question_embedding = self.sbert.encode([question])[0]

        # Compute cosine similarities and store top_k_indices
        sims = cosine_similarity(self.skill_embs, self.current_question_embedding.reshape(1, -1)).flatten()
        self.top_k_indices = np.argsort(sims)[-self.top_k:]

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
        
        student_performance = self.predict_student_performance(skill_id, question_type_id)
        # Store performance history
        self.student_performance_history.append(student_performance)

        # Get question from bank 
        question = self.get_question_from_bank(skill, question_type)

        # Use average performance improvement across all skills
        if len(self.student_performance_history) > 1:
            prev_avg_perf = float(np.mean(self.student_performance_history[-2]))
            curr_avg_perf = float(np.mean(self.student_performance_history[-1]))
            improvement_reward = curr_avg_perf - prev_avg_perf
        else:
            improvement_reward = 0.0

        answerability = self.compute_answerability()
        weak_skill_coverage = self.weak_skill_coverage()

        reward = self.calculate_reward(improvement_reward, answerability, weak_skill_coverage)

        # Observation: student's predicted performance for all skills
        observation = np.array([student_performance], dtype=np.float32)

        info = {
            "question": question,
            "skill": skill,
            "improvement": float(improvement_reward*self.w_improvement),
            "answerability": float(answerability*self.w_answerability),
            "coverage": float(weak_skill_coverage*self.w_coverage),
            "question_type": question_type,
            "predicted_correctness_for_skill": float(student_performance[skill_id]),
            "student_performance_per_skill": {k: float(v) for k, v in zip(self.all_skills, student_performance.tolist())},
            "skill_id": int(skill_id),
            "question_type_id": int(question_type_id),
            "reward": float(reward),
        }
        # Update action history
        self.update_history(info)

        self.current_step += 1
        done = self.current_step >= self.max_steps  # <-- done when max_steps reached

        if done:
            self.save_history_json()

        print(
            f"Step {self.current_step} | "
            f"Question: {info['question']} | "
            f"Skill: {info['skill']} | "
            f"Type: {info['question_type']} | "
            f"AvgPerf: {(np.mean(student_performance) * self.w_improvement):.3f} | "
            f"Improvement: {info['improvement']:.3f} | "
            f"Answerability: {info['answerability']:.3f} | "
            f"Coverage: {info['coverage']:.3f} | "
            f"Reward: {reward:.3f}"
        )
        print(f'Step {self.current_step} Complete')

        return observation, reward, done, info

    def save_history_json(self, path="trainings/history/history.json"):
        """
        Save the environment's history to a JSON file.
        Ensures the directory exists before saving.
        """
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def reset(self):
        """
        Reset the environment and student history.
        Returns initial observation.
        """
        self.history = []
        self.student_performance = np.ones(self.num_skills, dtype=np.float32) * 0.5
        self.current_step = 0 
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