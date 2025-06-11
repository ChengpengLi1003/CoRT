"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from deepscaler.system_prompts import ORM_PROMPT
from deepscaler.utils import call_gemini_llm, call_oai_rm_llm, call_oai_grm

ORM_USER_TEMPLATE = r"""
You are a skilled mathematics and programming expert. Your task is to evaluate how effectively Python tools are utilized in mathematical problem-solving processes. Please analyze the given mathematical reasoning trajectory and rate it based on the following criteria:

Score rules:
- Score 0 (Poor): Many opportunities to use Python were missed where it would have been highly beneficial
- Score 1 (Partial): Python was used for some calculations, but several appropriate opportunities for Python implementation were missed
- Score 2 (Excellent): Python was appropriately used in all or most suitable scenarios, OR the problem didn't present any suitable opportunities for Python usage

Key scenarios where Python should be considered:

1. Basic Number Theory:
   - Prime factorization, GCD, LCM
   - Modular arithmetic and modular inverse
   - Finding divisors and primality testing
   - Euler's totient function
   - Chinese remainder theorem
   - Generating sequences (Fibonacci, etc.)
   - Diophantine equations

2. Algebra:
   - Systems of equations (linear/non-linear)
   - Polynomial operations and GCD
   - De Moivre's theorem applications
   - Function inverses
   - Conic sections calculations
   - Symbolic mathematics (using sympy)
   - Factor rings and ideals

3. Complex Calculations:
   - Large number arithmetic
   - Complex number manipulations
   - Logarithmic/exponential computations
   - Root finding of polynomials
   - Series expansions

4. Calculus:
   - Integration (definite, indefinite)
   - Differentiation
   - Arclength calculations
   - Vector calculus (divergence, curl, gradients)
   - Jacobian and Laplacian computations
   - Multiple integrals
   - Series convergence tests

5. Linear Algebra:
   - Matrix operations (determinant, rank)
   - Eigenvalues and eigenvectors
   - Characteristic polynomials
   - Reduced row echelon form
   - Matrix decompositions
   - Vector space operations
   - Linear transformations

6. Geometry:
   - Triangle measurements (area, inradius, circumradius)
   - Polygon angles and properties
   - Polyhedron measurements
   - Coordinate geometry problems
   - Geometric transformations
   - Distance calculations
   - Intersection problems

7. Statistics and Probability:
   - Expectation calculations
   - Variance and covariance
   - Geometric and harmonic means
   - KL divergence
   - Probability distributions
   - Combinatorial calculations
   - Hypothesis testing

8. Optimization Problems:
   - Linear programming
   - Non-linear optimization
   - Constraint satisfaction
   - Minimization/maximization
   - Gradient descent
   - Lagrange multipliers

9. Numerical Methods:
   - Numerical integration
   - Interpolation
   - Approximation methods
   - Root finding algorithms
   - Numerical differentiation
   - Error analysis

10. Graph Theory:
    - Path finding algorithms
    - Graph properties
    - Network analysis
    - Tree operations
    - Isomorphism checking
    - Graph coloring

Please provide:
1. A list of scenarios in the reasoning where Python could/should have been used
2. An analysis of how Python was actually used (if at all)
3. A final score in the format \boxed{X} where X is 0, 1, or 2

Note: Python usage should not be recommended for simple calculations that can be done mentally or with basic calculator operations.

Input format: I will provide a mathematical problem and its solution trajectory.
Please analyze and score accordingly.

Problem:
{Problem}

Solution Trajectory:
{Solution}
""".strip()

def extract_programs_and_outputs(text: str) -> list[tuple[str, str]]:
    """
    Extract all Python code blocks and their corresponding output blocks from the text.
    Returns a list of tuples, each tuple contains (program, output).
    If a program has no output block, the output will be an empty string.
    Incomplete or empty blocks are skipped.
    """
    # 新增: 辅助函数用于删除代码块的共同缩进
    def dedent_code(code: str) -> str:
        if not code:
            return code
        
        # 分割成行
        lines = code.splitlines()
        # 找出所有非空行的最小缩进
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped:  # 忽略空行
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)
        
        if min_indent == float('inf'):
            return code
            
        # 对每行删除共同的缩进空白
        dedented_lines = []
        for line in lines:
            if line.strip():  # 非空行
                dedented_lines.append(line[min_indent:])
            else:  # 空行保持原样
                dedented_lines.append(line)
        
        return '\n'.join(dedented_lines)

    results = []
    lines = text.split("\n")
    i = 0
    
    while i < len(lines):
        # Skip until we find a Python code block start
        while i < len(lines) and not lines[i].strip() == "```python":
            i += 1
            
        if i >= len(lines):
            break  # No more Python code blocks
            
        # Start processing Python block
        i += 1  # Skip ```python line
        code_block = ""
        code_complete = False
        
        # Extract code until closing backticks
        while i < len(lines):
            if lines[i].strip() == "```":
                code_complete = True
                i += 1  # Skip closing backticks
                break
            code_block += lines[i] + "\n"
            i += 1
            
        # Skip incomplete or empty code blocks
        if not code_complete or not code_block.strip():
            continue

        # 修改: 在这里对代码块进行dedent处理
        code_block = dedent_code(code_block)
            
        # Now look for an output block
        j = i
        output_block = ""
        output_found = False
        
        # Skip until output block or another Python block
        while j < len(lines):
            if lines[j].strip() == "```output":
                # Found potential output block
                j += 1  # Skip ```output marker
                output_tmp = ""
                output_complete = False
                
                # Extract output until closing backticks
                while j < len(lines):
                    if lines[j].strip() == "```":
                        output_complete = True
                        j += 1  # Skip closing backticks
                        break
                    output_tmp += lines[j] + "\n"
                    j += 1
                    
                if output_complete:
                    output_block = output_tmp
                    output_found = True
                    i = j  # Update main pointer
                    break
                # If incomplete, continue looking
                
            elif lines[j].strip() == "```python":
                # Found another Python block first
                break
                
            j += 1
        
        # Add code-output pair to results
        results.append((code_block, output_block))
    
    return results


def all_error_output(result: str):
    programs_and_outputs = extract_programs_and_outputs(result)

    def is_error_output(output: str) -> bool:
        """Check if output contains error messages"""
        if output is None:
            return True
        if output.strip() == "Done":
            return True
        if output.strip() == "":
            return True
        if output.strip() == "[]":
            return True
        error_keywords = [
            'error', 
            'exception', 
            'traceback',
        ]
        output = output.lower()
        return any(keyword in output for keyword in error_keywords)

    error_outputs = []
    for program, output in programs_and_outputs:
        error_outputs.append( is_error_output(output) )
    return all(error_outputs)

class RewardMathTIRFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        penalty = 0

        if self.config.use_code_exe_reward:
            if all_error_output(model_response):
                code_exe_penalty = -0.1
            else:
                code_exe_penalty = 0.0
            print(f"code_exe_penalty: {code_exe_penalty}")
            penalty += code_exe_penalty

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            problem = problem.replace("Please integrate natural language reasoning with python programs to solve the problem above, and put your final answer within \\boxed{}.", "").strip()
            grm_response = call_oai_grm(
                prompt=ORM_USER_TEMPLATE.replace("{Problem}", problem).replace("{Solution}", model_response),
            )
            print(f"grm_response: {grm_response}")
            grm_score = extract_answer(grm_response)
            if grm_score is None:
                grm_penalty = 0
            else:
                try:
                    grm_score = float(str(grm_score))
                    if grm_score == 0:
                        grm_penalty = -0.1
                    elif grm_score == 1:
                        grm_penalty = 0
                    elif grm_score == 2:
                        grm_penalty = 0.1
                    else:
                        grm_penalty = 0
                except:
                    grm_penalty = 0
            print(f"grm_penalty: {grm_penalty}")
            penalty += grm_penalty

        correct_reward = 1.0
        incorrect_reward = 0.0
        
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return incorrect_reward + penalty
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return incorrect_reward + penalty

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return incorrect_reward + penalty
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return incorrect_reward + penalty

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return correct_reward + penalty
                
        return incorrect_reward + penalty

def hybrid_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = True, enable_code_reward = True):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_config.use_code_exe_reward = enable_code_reward
    reward_fn = RewardMathTIRFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response

def code_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False, enable_code_reward = True):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_config.use_code_exe_reward = enable_code_reward
    reward_fn = RewardMathTIRFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response

def orm_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = True, enable_code_reward = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_config.use_code_exe_reward = enable_code_reward
    reward_fn = RewardMathTIRFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)