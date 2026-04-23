"""Ground-truth question banks with verifiable canonical answers.

Each question carries:
- prompt: the question text shown to the model
- canonical_answer: the known-correct answer
- check_method: how to verify (sympy_equiv, code_unit_tests, exact_mc,
  numeric_exact, exact_string)
- domain / subdomain / difficulty / source (GSM8K-style, HumanEval-style, etc.)

Problems are either:
1. Hard-coded from benchmark distributions (GSM8K grade-school math, MATH
   competition math, HumanEval-style code-with-tests, MMLU multi-choice,
   LogiQA logic). Answers are known because humans curated them.
2. Programmatically constructed — the generator computes the answer as it
   builds the problem, so ground truth is guaranteed.

Grading uses sympy for math, sandboxed unit tests for code, exact-match for
MC/numeric. No substring "contains" matching for final correctness.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd, isclose
from typing import Callable, Optional


@dataclass
class GroundTruthQuestion:
    prompt: str
    canonical_answer: str
    check_method: str  # sympy_equiv | code_unit_tests | exact_mc | numeric_exact | exact_string
    domain: str
    subdomain: str
    difficulty: str = "medium"
    source: str = "curated"
    # For code_unit_tests, this is a list of assertion strings run against
    # the model's extracted function.
    unit_tests: list[str] = None
    # Function name the model is asked to implement (for code_unit_tests).
    entry_point: str = ""
    # Optional numeric tolerance for numeric_exact.
    tol: float = 1e-6
    # H2 (hot_spots): AST-level forbidden symbols for code_unit_tests problems
    # whose prompt imposes an algorithmic constraint the I/O tests can't
    # enforce (e.g. "without built-in sort", "binary search in O(log n)").
    # Each entry is a dotted name or bare name: "sorted", "list.sort",
    # "list.index". Presence of a matching Call or Attribute anywhere in the
    # response AST fails the grader.
    forbidden_symbols: list[str] = None


# ---------------------------------------------------------------------------
# Hard-coded benchmark-style questions (GSM8K / MATH / HumanEval / MMLU / LogiQA)
# ---------------------------------------------------------------------------
# These are curated in the SPIRIT of each benchmark, not verbatim copies — so
# we don't ship the benchmark contents, but the distribution is preserved.

# GSM8K-style: grade-school multi-step arithmetic word problems.
# Answers are exact integers or decimals.
_GSM8K_STYLE = [
    ("Janet has 3 dozen eggs. She gives away 8 eggs to her neighbor and uses 5 for breakfast. How many eggs does she have left?", "23"),
    ("A bookstore sells paperbacks for $12 and hardcovers for $25. If Maria buys 4 paperbacks and 2 hardcovers, how much does she spend?", "98"),
    ("A train travels 60 miles per hour for 2.5 hours, then 45 miles per hour for 3 hours. How many miles total?", "285"),
    ("Tom weighs 180 pounds. He loses 2 pounds per week for 12 weeks, then gains back 5 pounds. What is his final weight?", "161"),
    ("A recipe needs 3 cups of flour per loaf. If Lucy has 20 cups of flour and makes as many complete loaves as possible, how many cups are left over?", "2"),
    ("A store discounts a $80 jacket by 25%, then adds 8% sales tax. What is the final price?", "64.80"),
    ("Sam runs 3 miles on Monday, 4 miles on Tuesday, 5 miles on Wednesday, and rests Thursday. He runs double his Tuesday distance on Friday. Total miles?", "20"),
    ("Alice has twice as many apples as Bob. Bob has 5 fewer apples than Carol. If Carol has 12 apples, how many apples does Alice have?", "14"),
    ("A tank holds 500 gallons. It fills at 8 gallons per minute and drains at 3 gallons per minute. Starting empty, how many minutes to fill?", "100"),
    ("A class has 30 students. 40% are boys. How many girls are there?", "18"),
    ("A farmer has 15 cows and 22 chickens. How many legs total on the farm (animals only)?", "104"),
    ("Kim saves $50 per month. After 8 months she spends $120 on a gift. How much remains?", "280"),
    ("A pizza is cut into 12 equal slices. Dan eats 3 slices, Ellie eats 4. What fraction of the pizza remains? (Answer as a fraction in lowest terms.)", "5/12"),
    ("A rectangle has length 15 cm and width 8 cm. What is the area in square cm?", "120"),
    ("A marathon is 26.2 miles. If a runner completes it at 7 minutes per mile pace, how many minutes does it take? (Round to one decimal.)", "183.4"),
    ("If 5 workers build a wall in 6 days, how many days do 10 workers take (at the same rate per worker)?", "3"),
    ("A box has 24 chocolates. 1/3 are dark, 1/4 are milk, the rest are white. How many are white?", "10"),
    ("A jacket is marked up 50% from a wholesale price of $40. What is the retail price?", "60"),
    ("Pat buys 3 shirts at $18 each and 2 pairs of pants at $35 each. He pays with a $200 bill. How much change?", "76"),
    ("A school has 480 students. The ratio of boys to girls is 5:3. How many boys?", "300"),
    ("A tree is 8 feet tall and grows 15% per year. How tall after 2 years? (Round to 2 decimals.)", "10.58"),
    ("A train leaves at 9:15 AM and arrives at 2:45 PM. How many minutes is the trip?", "330"),
    ("Jake reads 22 pages per day. A book has 308 pages. How many days to finish?", "14"),
    ("A square has perimeter 36 inches. What is the area in square inches?", "81"),
    ("A store had 200 items. They sold 15% on Monday, 20% of the REMAINING on Tuesday. How many items are left?", "136"),
    ("A phone costs $720 on a 24-month plan. The same phone costs $600 upfront. How much more do you pay on the plan?", "120"),
    ("A mixture is 30% salt by weight. How many grams of salt in 250 grams of mixture?", "75"),
    ("Ana has $15. Ben has 3 times as much as Ana. Cal has $7 less than Ben. How much does Cal have?", "38"),
    ("A field is 80 meters by 60 meters. A fence goes around the perimeter. How many meters of fence?", "280"),
    ("A bus leaves every 12 minutes starting at 6:00 AM. What time does the 10th bus leave?", "7:48 AM"),
    ("If 4 pens cost $6, how much do 18 pens cost (same rate)?", "27"),
    ("A number is tripled, then decreased by 5, giving 19. What is the original number?", "8"),
    ("A loan of $1000 earns 5% simple interest per year. Total owed after 3 years?", "1150"),
    ("A swimming pool is 50 m long. A swimmer does 30 laps (one lap = one length). How many meters swum?", "1500"),
    ("A basket has 3 red, 5 blue, and 2 green balls. What fraction are blue (lowest terms)?", "1/2"),
    ("A car uses 3 gallons to travel 90 miles. How many miles per gallon?", "30"),
    ("A worker earns $18/hour for the first 40 hours and time-and-a-half for overtime. How much for a 45-hour week?", "855"),
    ("A parking lot has 4 rows with 25 spots each. If 73 spots are taken, how many are free?", "27"),
    ("A tank is 3/4 full with 150 liters. What is the tank's full capacity?", "200"),
    ("A rope is 48 feet. It is cut into equal pieces of 3 feet. How many pieces?", "16"),
    ("If x + 7 = 22, what is x?", "15"),
    ("A circle has radius 10. What is its circumference? (Use pi ≈ 3.14159, round to 2 decimals.)", "62.83"),
    ("A store buys items at $8 and sells at $14. What is the profit margin percent (profit/selling, rounded to whole %)?", "43"),
    ("Kelly starts with 40 stickers. She gives away 1/5 and then buys 12 more. How many now?", "44"),
    ("A recipe for 6 servings uses 2 cups of milk. How many cups for 15 servings?", "5"),
    ("A ladder leans against a wall. The base is 5 ft from the wall, the ladder is 13 ft long. How high up the wall does it reach?", "12"),
    ("A worker packs 8 boxes per hour. How many boxes in a 7.5-hour shift?", "60"),
    ("A triangle has sides 6, 8, 10. What is its area?", "24"),
    ("A number doubled plus 9 equals 31. What is the number?", "11"),
    ("A 5% bonus is added to a $2400 paycheck. What is the total?", "2520"),
    ("Ed walks 1.2 miles to school each way, 5 days a week. How many miles in a week?", "12"),
]


# MATH-competition style: harder symbolic/numeric. Answers machine-checkable.
_MATH_STYLE = [
    ("Compute the derivative of x^3 + 2x^2 - 5x + 7.", "3*x**2 + 4*x - 5", "sympy_equiv"),
    ("Compute the derivative of sin(x) * cos(x).", "cos(2*x)", "sympy_equiv"),
    ("Compute the derivative of ln(x^2 + 1).", "2*x/(x**2 + 1)", "sympy_equiv"),
    ("Integrate 3x^2 + 2x with respect to x (omit the +C).", "x**3 + x**2", "sympy_equiv"),
    ("Integrate 1/x from x=1 to x=e. Give the exact value.", "1", "sympy_equiv"),
    ("Evaluate lim x->0 of sin(5x)/x.", "5", "sympy_equiv"),
    ("Evaluate lim x->infinity of (3x^2 + x)/(x^2 - 2).", "3", "sympy_equiv"),
    ("Simplify (x^2 - 9)/(x - 3) for x != 3.", "x + 3", "sympy_equiv"),
    ("Factor x^2 - 5x + 6. Write as (x-a)(x-b) with a<b; answer as the polynomial expanded form a*b - (a+b)x + x^2? Just give the factored polynomial multiplied out: x**2 - 5x + 6.", "x**2 - 5*x + 6", "sympy_equiv"),
    ("Solve for x: 2x + 5 = 17.", "6", "numeric_exact"),
    ("Solve for x: 3x - 4 = 2x + 11.", "15", "numeric_exact"),
    ("Find both roots of x^2 - 7x + 12 = 0. List as a sorted set, comma-separated.", "3, 4", "exact_string"),
    ("Find both roots of x^2 - 4 = 0. List as a sorted set, comma-separated.", "-2, 2", "exact_string"),
    ("Compute 7! (7 factorial).", "5040", "numeric_exact"),
    ("Compute C(10, 3), the binomial coefficient 10 choose 3.", "120", "numeric_exact"),
    ("Compute the GCD of 84 and 126.", "42", "numeric_exact"),
    ("Compute 2^10.", "1024", "numeric_exact"),
    ("What is log base 2 of 64?", "6", "numeric_exact"),
    ("Evaluate the sum 1 + 2 + 3 + ... + 100.", "5050", "numeric_exact"),
    ("What is the 10th Fibonacci number (with F(1)=1, F(2)=1)?", "55", "numeric_exact"),
    ("How many positive divisors does 60 have?", "12", "numeric_exact"),
    ("Compute the sine of pi/6 as an exact value.", "1/2", "sympy_equiv"),
    ("Compute cos(pi/3) as an exact value.", "1/2", "sympy_equiv"),
    ("Compute tan(pi/4) as an exact value.", "1", "sympy_equiv"),
    ("What is the value of e^0?", "1", "numeric_exact"),
    ("Compute 17 mod 5.", "2", "numeric_exact"),
    ("How many prime numbers are there less than 20?", "8", "numeric_exact"),
    ("What is the smallest prime greater than 50?", "53", "numeric_exact"),
    ("Expand (x+2)^3 completely.", "x**3 + 6*x**2 + 12*x + 8", "sympy_equiv"),
    ("Expand (x+1)(x+2)(x+3).", "x**3 + 6*x**2 + 11*x + 6", "sympy_equiv"),
    ("Simplify sin(x)^2 + cos(x)^2.", "1", "sympy_equiv"),
    ("Derivative of e^(3x).", "3*exp(3*x)", "sympy_equiv"),
    ("Derivative of tan(x).", "1/cos(x)**2", "sympy_equiv"),
    ("Integrate cos(x).", "sin(x)", "sympy_equiv"),
    ("Solve: log_2(x) = 5. What is x?", "32", "numeric_exact"),
    ("Solve: 3^x = 81. What is x?", "4", "numeric_exact"),
    ("What is the sum of interior angles of a pentagon (in degrees)?", "540", "numeric_exact"),
    ("A right triangle has legs 9 and 12. What is the hypotenuse?", "15", "numeric_exact"),
    ("The mean of 5, 7, 10, 14 is what?", "9", "numeric_exact"),
    ("The median of 1, 3, 5, 7, 9, 11 is what?", "6", "numeric_exact"),
    ("Evaluate |3 - 8|.", "5", "numeric_exact"),
    ("Derivative of x*sin(x).", "sin(x) + x*cos(x)", "sympy_equiv"),
    ("What is the determinant of the 2x2 matrix [[3, 4], [2, 5]]?", "7", "numeric_exact"),
    ("Find x if x/3 + 4 = 10.", "18", "numeric_exact"),
    ("Compute gcd(1071, 462).", "21", "numeric_exact"),
]


# HumanEval-style: function signature + docstring + unit tests.
# Model must implement the function; we run unit tests in the sandbox.
_HUMANEVAL_STYLE = [
    {
        "prompt": ("Write a Python function `is_palindrome(s: str) -> bool` that returns True if the "
                   "string `s` reads the same forwards and backwards (case-sensitive, whitespace counts)."),
        "entry_point": "is_palindrome",
        "tests": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('') == True",
            "assert is_palindrome('a') == True",
            "assert is_palindrome('ab') == False",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `factorial(n: int) -> int` that returns n! for n >= 0. "
                   "factorial(0) must return 1."),
        "entry_point": "factorial",
        "tests": [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
            "assert factorial(10) == 3628800",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `fibonacci(n: int) -> int` that returns the n-th Fibonacci "
                   "number with fibonacci(0) == 0, fibonacci(1) == 1."),
        "entry_point": "fibonacci",
        "tests": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55",
            "assert fibonacci(20) == 6765",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `is_prime(n: int) -> bool` that returns True if `n` is a "
                   "prime number. Must handle n <= 1 by returning False."),
        "entry_point": "is_prime",
        "tests": [
            "assert is_prime(2) == True",
            "assert is_prime(3) == True",
            "assert is_prime(4) == False",
            "assert is_prime(1) == False",
            "assert is_prime(0) == False",
            "assert is_prime(17) == True",
            "assert is_prime(25) == False",
            "assert is_prime(97) == True",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `reverse_string(s: str) -> str` that returns `s` reversed."),
        "entry_point": "reverse_string",
        "tests": [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
            "assert reverse_string('ab cd') == 'dc ba'",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `count_vowels(s: str) -> int` that returns the number of "
                   "vowels (a, e, i, o, u, case-insensitive) in `s`."),
        "entry_point": "count_vowels",
        "tests": [
            "assert count_vowels('hello') == 2",
            "assert count_vowels('AEIOU') == 5",
            "assert count_vowels('xyz') == 0",
            "assert count_vowels('') == 0",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `gcd(a: int, b: int) -> int` that computes the greatest "
                   "common divisor of two non-negative integers. gcd(a, 0) == a."),
        "entry_point": "gcd",
        "tests": [
            "assert gcd(12, 18) == 6",
            "assert gcd(100, 75) == 25",
            "assert gcd(7, 13) == 1",
            "assert gcd(10, 0) == 10",
            "assert gcd(0, 10) == 10",
        ],
        "difficulty": "medium",
    },
    {
        "prompt": ("Write a Python function `merge_sorted(a: list, b: list) -> list` that merges two "
                   "already-sorted lists into one sorted list, without using built-in sort."),
        "entry_point": "merge_sorted",
        "tests": [
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]",
            "assert merge_sorted([], [1, 2]) == [1, 2]",
            "assert merge_sorted([1, 2], []) == [1, 2]",
            "assert merge_sorted([], []) == []",
            "assert merge_sorted([1, 1, 1], [1, 1]) == [1, 1, 1, 1, 1]",
        ],
        "difficulty": "medium",
        "forbidden_symbols": ["sorted", "list.sort"],
    },
    {
        "prompt": ("Write a Python function `binary_search(arr: list, target) -> int` that returns the "
                   "index of `target` in the sorted list `arr`, or -1 if not present."),
        "entry_point": "binary_search",
        "tests": [
            "assert binary_search([1, 2, 3, 4, 5], 3) == 2",
            "assert binary_search([1, 2, 3, 4, 5], 6) == -1",
            "assert binary_search([], 1) == -1",
            "assert binary_search([1], 1) == 0",
            "assert binary_search([1, 3, 5, 7, 9, 11], 11) == 5",
        ],
        "difficulty": "medium",
        "forbidden_symbols": ["list.index"],
    },
    {
        "prompt": ("Write a Python function `flatten(nested: list) -> list` that flattens a nested list "
                   "(arbitrary depth) into a single flat list preserving order."),
        "entry_point": "flatten",
        "tests": [
            "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]",
            "assert flatten([]) == []",
            "assert flatten([1, 2, 3]) == [1, 2, 3]",
            "assert flatten([[[1]]]) == [1]",
        ],
        "difficulty": "medium",
    },
    {
        "prompt": ("Write a Python function `sum_digits(n: int) -> int` that returns the sum of the "
                   "decimal digits of n (use abs(n) for negatives)."),
        "entry_point": "sum_digits",
        "tests": [
            "assert sum_digits(123) == 6",
            "assert sum_digits(0) == 0",
            "assert sum_digits(9999) == 36",
            "assert sum_digits(-45) == 9",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `anagram(a: str, b: str) -> bool` that returns True if `a` "
                   "and `b` are anagrams of each other (case-sensitive, whitespace counts)."),
        "entry_point": "anagram",
        "tests": [
            "assert anagram('listen', 'silent') == True",
            "assert anagram('hello', 'world') == False",
            "assert anagram('', '') == True",
            "assert anagram('a', 'ab') == False",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `sieve(n: int) -> list` that returns all primes less than "
                   "or equal to `n` using the Sieve of Eratosthenes. sieve(1) returns []."),
        "entry_point": "sieve",
        "tests": [
            "assert sieve(10) == [2, 3, 5, 7]",
            "assert sieve(1) == []",
            "assert sieve(2) == [2]",
            "assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]",
        ],
        "difficulty": "medium",
    },
    {
        "prompt": ("Write a Python function `rotate_list(lst: list, k: int) -> list` that rotates "
                   "`lst` to the right by `k` positions. Handle k larger than len(lst)."),
        "entry_point": "rotate_list",
        "tests": [
            "assert rotate_list([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]",
            "assert rotate_list([1, 2, 3], 0) == [1, 2, 3]",
            "assert rotate_list([1, 2, 3], 5) == [2, 3, 1]",
            "assert rotate_list([], 3) == []",
        ],
        "difficulty": "medium",
    },
    {
        "prompt": ("Write a Python function `count_words(s: str) -> dict` that returns a dict mapping "
                   "each whitespace-separated word to its count."),
        "entry_point": "count_words",
        "tests": [
            "assert count_words('a b a') == {'a': 2, 'b': 1}",
            "assert count_words('') == {}",
            "assert count_words('hello') == {'hello': 1}",
        ],
        "difficulty": "easy",
    },
    {
        "prompt": ("Write a Python function `power(base: float, exp: int) -> float` that computes "
                   "base**exp for non-negative integer exp, using iteration (no ** operator, no pow)."),
        "entry_point": "power",
        "tests": [
            "assert power(2, 10) == 1024",
            "assert power(5, 0) == 1",
            "assert power(3, 4) == 81",
            "assert power(1, 100) == 1",
        ],
        "difficulty": "medium",
    },
    {
        "prompt": ("Write a Python function `second_largest(lst: list) -> int` that returns the "
                   "second-largest DISTINCT value in `lst`. Assume len(set(lst)) >= 2."),
        "entry_point": "second_largest",
        "tests": [
            "assert second_largest([1, 2, 3, 4, 5]) == 4",
            "assert second_largest([5, 5, 4, 3]) == 4",
            "assert second_largest([10, 20]) == 10",
            "assert second_largest([-1, -2, -3]) == -2",
        ],
        "difficulty": "medium",
    },
]


# MMLU-style: multiple-choice factual knowledge. Exact-match on letter choice.
_MMLU_STYLE = [
    # (question, choices_dict, correct_letter, subdomain)
    ("Which planet is known as the Red Planet?",
     {"A": "Venus", "B": "Mars", "C": "Jupiter", "D": "Saturn"}, "B", "science"),
    ("The chemical symbol for gold is:",
     {"A": "Go", "B": "Gd", "C": "Au", "D": "Ag"}, "C", "science"),
    ("The speed of light in vacuum is approximately:",
     {"A": "3 × 10^5 m/s", "B": "3 × 10^8 m/s", "C": "3 × 10^6 m/s", "D": "3 × 10^10 m/s"},
     "B", "science"),
    ("Which is NOT a noble gas?",
     {"A": "Helium", "B": "Neon", "C": "Nitrogen", "D": "Argon"}, "C", "science"),
    ("DNA is structured as a:",
     {"A": "Single helix", "B": "Double helix", "C": "Triple helix", "D": "Linear chain"},
     "B", "science"),
    ("Water's chemical formula is:",
     {"A": "HO", "B": "H2O", "C": "H2O2", "D": "HO2"}, "B", "science"),
    ("Photosynthesis converts carbon dioxide and water, using sunlight, into:",
     {"A": "Oxygen and glucose", "B": "Nitrogen and sugar",
      "C": "Methane and water", "D": "Carbon and hydrogen"}, "A", "science"),
    ("The largest organ in the human body is the:",
     {"A": "Liver", "B": "Brain", "C": "Skin", "D": "Lungs"}, "C", "science"),
    ("The unit of electrical resistance is the:",
     {"A": "Volt", "B": "Watt", "C": "Ohm", "D": "Ampere"}, "C", "science"),
    ("Absolute zero in Celsius is approximately:",
     {"A": "0 °C", "B": "-100 °C", "C": "-273 °C", "D": "-459 °C"}, "C", "science"),
    ("Which U.S. President signed the Emancipation Proclamation?",
     {"A": "Washington", "B": "Lincoln", "C": "Jefferson", "D": "Roosevelt"}, "B", "history"),
    ("World War II ended in:",
     {"A": "1918", "B": "1939", "C": "1945", "D": "1950"}, "C", "history"),
    ("The capital of Australia is:",
     {"A": "Sydney", "B": "Melbourne", "C": "Canberra", "D": "Perth"}, "C", "geography"),
    ("The Great Wall of China was primarily built during which dynasty's peak era of construction? (choose the most associated)",
     {"A": "Tang", "B": "Ming", "C": "Qing", "D": "Yuan"}, "B", "history"),
    ("The longest river in the world is the:",
     {"A": "Amazon", "B": "Nile", "C": "Yangtze", "D": "Mississippi"}, "B", "geography"),
    ("The Pacific Ocean is the:",
     {"A": "Smallest ocean", "B": "Second largest", "C": "Largest ocean", "D": "Third largest"},
     "C", "geography"),
    ("The currency of Japan is the:",
     {"A": "Won", "B": "Yuan", "C": "Yen", "D": "Ringgit"}, "C", "geography"),
    ("Shakespeare wrote:",
     {"A": "Oliver Twist", "B": "Hamlet", "C": "War and Peace", "D": "Ulysses"}, "B", "literature"),
    ("The author of '1984' is:",
     {"A": "Aldous Huxley", "B": "George Orwell",
      "C": "Ray Bradbury", "D": "Kurt Vonnegut"}, "B", "literature"),
    ("Python was created by:",
     {"A": "James Gosling", "B": "Guido van Rossum",
      "C": "Bjarne Stroustrup", "D": "Dennis Ritchie"}, "B", "computing"),
    ("HTTP stands for:",
     {"A": "Hyper Transfer Text Protocol", "B": "Hypertext Transfer Protocol",
      "C": "Hyperlink Transfer Text Protocol", "D": "High Transfer Text Protocol"},
     "B", "computing"),
    ("In Big-O notation, binary search on a sorted array is:",
     {"A": "O(1)", "B": "O(log n)", "C": "O(n)", "D": "O(n log n)"}, "B", "computing"),
    ("A byte is how many bits?",
     {"A": "4", "B": "8", "C": "16", "D": "32"}, "B", "computing"),
    ("The primary function of RAM is to:",
     {"A": "Store files permanently",
      "B": "Provide temporary working memory",
      "C": "Process instructions",
      "D": "Display graphics"}, "B", "computing"),
    ("SQL stands for:",
     {"A": "Structured Query Language",
      "B": "Simple Query Language",
      "C": "Standard Query Logic",
      "D": "System Quality Language"}, "A", "computing"),
    ("The Sun is classified as a:",
     {"A": "Red giant", "B": "White dwarf", "C": "Yellow dwarf (G-type main sequence)",
      "D": "Neutron star"}, "C", "science"),
    ("Mitochondria primarily function in:",
     {"A": "Protein synthesis", "B": "Energy production (ATP)",
      "C": "Waste removal", "D": "DNA storage"}, "B", "science"),
    ("The pH of pure water at 25 °C is:",
     {"A": "5", "B": "6", "C": "7", "D": "8"}, "C", "science"),
    ("The hardest natural substance known is:",
     {"A": "Steel", "B": "Diamond", "C": "Quartz", "D": "Gold"}, "B", "science"),
    ("The force pulling objects toward Earth is called:",
     {"A": "Magnetism", "B": "Gravity", "C": "Friction", "D": "Inertia"}, "B", "science"),
    ("The mathematical constant pi is approximately:",
     {"A": "2.718", "B": "3.141", "C": "1.618", "D": "1.414"}, "B", "math"),
    ("The mathematical constant e (Euler's number) is approximately:",
     {"A": "2.718", "B": "3.141", "C": "1.618", "D": "1.414"}, "A", "math"),
    ("The golden ratio is approximately:",
     {"A": "2.718", "B": "3.141", "C": "1.618", "D": "1.414"}, "C", "math"),
    ("A prime number has exactly:",
     {"A": "One divisor", "B": "Two divisors (1 and itself)",
      "C": "Three divisors", "D": "An infinite number of divisors"}, "B", "math"),
    ("In a right triangle, the Pythagorean theorem states:",
     {"A": "a + b = c", "B": "a^2 + b^2 = c^2",
      "C": "a*b = c", "D": "a^2 - b^2 = c^2"}, "B", "math"),
]


# LogiQA-style: logical reasoning problems. Exact-match answer choice.
_LOGIQA_STYLE = [
    ("All cats are mammals. All mammals are animals. Therefore, all cats are animals. Is this argument valid?",
     {"A": "Valid", "B": "Invalid"}, "A"),
    ("All birds can fly. Penguins are birds. Therefore, penguins can fly. Is the argument VALID? (Ignore factual truth of premises — assess validity only.)",
     {"A": "Valid", "B": "Invalid"}, "A"),
    ("Some dogs are brown. All brown things are warm. Therefore, some dogs are warm. Is this argument valid?",
     {"A": "Valid", "B": "Invalid"}, "A"),
    ("If it rains, the ground is wet. The ground is wet. Therefore, it rained. What fallacy is this?",
     {"A": "Denying the antecedent", "B": "Affirming the consequent",
      "C": "Modus ponens (valid)", "D": "Modus tollens (valid)"}, "B"),
    ("If it rains, the ground is wet. It is not raining. Therefore, the ground is not wet. What fallacy is this?",
     {"A": "Denying the antecedent", "B": "Affirming the consequent",
      "C": "Modus ponens (valid)", "D": "Modus tollens (valid)"}, "A"),
    ("If P then Q. P is true. Therefore Q. What is this?",
     {"A": "Denying the antecedent", "B": "Affirming the consequent",
      "C": "Modus ponens (valid)", "D": "Modus tollens (valid)"}, "C"),
    ("If P then Q. Q is false. Therefore P is false. What is this?",
     {"A": "Denying the antecedent", "B": "Affirming the consequent",
      "C": "Modus ponens (valid)", "D": "Modus tollens (valid)"}, "D"),
    ("'Everyone agrees, so it must be true.' What fallacy is this?",
     {"A": "Ad hominem", "B": "Bandwagon (argumentum ad populum)",
      "C": "Straw man", "D": "False dilemma"}, "B"),
    ("'You either love this country or you hate it.' What fallacy is this?",
     {"A": "Ad hominem", "B": "Bandwagon",
      "C": "Straw man", "D": "False dilemma"}, "D"),
    ("No A are B. All C are A. Therefore, no C are B. Is this argument valid?",
     {"A": "Valid", "B": "Invalid"}, "A"),
    ("Some A are B. Some B are C. Therefore, some A are C. Is this argument valid?",
     {"A": "Valid", "B": "Invalid"}, "B"),
    ("Alice is taller than Bob. Bob is taller than Carol. Who is tallest?",
     {"A": "Alice", "B": "Bob", "C": "Carol", "D": "Cannot determine"}, "A"),
    ("Xi finishes before Yara. Yara finishes before Zeke. Who finishes last?",
     {"A": "Xi", "B": "Yara", "C": "Zeke", "D": "Cannot determine"}, "C"),
    ("All poets are dreamers. No dreamers are bankers. Therefore, no poets are bankers. Valid?",
     {"A": "Valid", "B": "Invalid"}, "A"),
    ("If the train is late, I'll be late. If I'm late, I'll miss the meeting. The train is late. Therefore:",
     {"A": "I will miss the meeting", "B": "I will not miss the meeting",
      "C": "Cannot determine", "D": "The meeting is cancelled"}, "A"),
    ("A person says: 'This statement is false.' This is known as:",
     {"A": "Modus ponens", "B": "A valid syllogism",
      "C": "The liar paradox", "D": "Affirming the consequent"}, "C"),
    ("All humans are mortal. Socrates is human. Therefore, Socrates is mortal. This is an example of:",
     {"A": "Inductive reasoning", "B": "Deductive reasoning",
      "C": "Abductive reasoning", "D": "Fallacious reasoning"}, "B"),
    ("You see 100 white swans and conclude all swans are white. This is:",
     {"A": "Deductive reasoning", "B": "Inductive reasoning (fallible)",
      "C": "Valid deduction", "D": "Modus tollens"}, "B"),
    ("'P or Q. Not P. Therefore Q.' This is:",
     {"A": "Disjunctive syllogism (valid)", "B": "Affirming the consequent",
      "C": "Denying the antecedent", "D": "Invalid"}, "A"),
    ("'Either you're with us or against us' when there are clearly other options is a:",
     {"A": "Straw man", "B": "False dilemma",
      "C": "Slippery slope", "D": "Red herring"}, "B"),
]


# ---------------------------------------------------------------------------
# Programmatic generators — generators that CONSTRUCT the answer as they build
# the problem, guaranteeing ground truth.
# ---------------------------------------------------------------------------

def gen_arithmetic(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Generate a multi-step arithmetic problem with known integer answer."""
    if difficulty == "easy":
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        op = rng.choice(["+", "-", "*"])
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        prompt = f"Compute: {a} {op} {b}."
    elif difficulty == "hard":
        a, b, c, d = [rng.randint(5, 50) for _ in range(4)]
        ans = (a + b) * c - d
        prompt = f"Compute: ({a} + {b}) * {c} - {d}."
    elif difficulty == "expert":
        a, b, c, d, e = [rng.randint(2, 30) for _ in range(5)]
        ans = a * b + c * d - e
        prompt = f"Compute: {a} * {b} + {c} * {d} - {e}."
    else:  # medium
        a, b, c = [rng.randint(3, 40) for _ in range(3)]
        op1, op2 = rng.choice([("+", "*"), ("-", "*"), ("*", "+")])
        if op2 == "*":
            ans = a + b * c if op1 == "+" else a - b * c
            prompt = f"Compute: {a} {op1} {b} * {c}."
        else:
            ans = a * b + c if op1 == "*" else a * b - c
            prompt = f"Compute: {a} * {b} {op1} {c}."
    return GroundTruthQuestion(
        prompt=prompt + " Give only the final integer answer.",
        canonical_answer=str(ans),
        check_method="numeric_exact",
        domain="math", subdomain="arithmetic", difficulty=difficulty,
        source="programmatic",
    )


def gen_linear_system(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Generate a 2x2 linear system with known integer solution."""
    lo, hi = (-10, 10) if difficulty in ("easy", "medium") else (-20, 20)
    # Pick solution first, then construct coefficients
    x = rng.randint(lo, hi)
    y = rng.randint(lo, hi)
    a1, b1 = rng.randint(1, 9), rng.randint(1, 9)
    a2, b2 = rng.randint(1, 9), rng.randint(1, 9)
    # Ensure determinant nonzero (so unique solution)
    while a1 * b2 - a2 * b1 == 0:
        a2 = rng.randint(1, 9)
        b2 = rng.randint(1, 9)
    c1 = a1 * x + b1 * y
    c2 = a2 * x + b2 * y
    prompt = (f"Solve the system: {a1}x + {b1}y = {c1}; {a2}x + {b2}y = {c2}. "
              f"Give the answer as 'x=<value>, y=<value>' with integer values.")
    return GroundTruthQuestion(
        prompt=prompt,
        canonical_answer=f"x={x}, y={y}",
        check_method="exact_string",
        domain="math", subdomain="algebra", difficulty=difficulty,
        source="programmatic",
    )


def gen_gcd(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """GCD problem — we know the answer because we multiply it in."""
    if difficulty == "easy":
        g = rng.randint(2, 10)
        a, b = g * rng.randint(2, 10), g * rng.randint(2, 10)
    else:
        g = rng.randint(3, 30)
        a, b = g * rng.randint(3, 50), g * rng.randint(3, 50)
    # True gcd might be higher
    true_g = gcd(a, b)
    return GroundTruthQuestion(
        prompt=f"What is gcd({a}, {b})? Give only the integer.",
        canonical_answer=str(true_g),
        check_method="numeric_exact",
        domain="math", subdomain="number_theory", difficulty=difficulty,
        source="programmatic",
    )


def gen_modular(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    m = rng.randint(3, 20) if difficulty == "easy" else rng.randint(5, 100)
    a = rng.randint(m, m * 20)
    ans = a % m
    return GroundTruthQuestion(
        prompt=f"Compute {a} mod {m}. Give only the integer.",
        canonical_answer=str(ans),
        check_method="numeric_exact",
        domain="math", subdomain="number_theory", difficulty=difficulty,
        source="programmatic",
    )


def gen_combinatorics(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    n = rng.randint(4, 8) if difficulty == "easy" else rng.randint(6, 15)
    k = rng.randint(1, n - 1)
    ans = comb(n, k)
    return GroundTruthQuestion(
        prompt=f"Compute C({n}, {k}), the number of ways to choose {k} items from {n}. Give only the integer.",
        canonical_answer=str(ans),
        check_method="numeric_exact",
        domain="math", subdomain="combinatorics", difficulty=difficulty,
        source="programmatic",
    )


def gen_coin_probability(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    n = rng.randint(3, 6) if difficulty == "easy" else rng.randint(4, 10)
    k = rng.randint(0, n)
    num = comb(n, k)
    den = 2 ** n
    g = gcd(num, den)
    num //= g
    den //= g
    ans = f"{num}/{den}" if den > 1 else f"{num}"
    return GroundTruthQuestion(
        prompt=(f"You flip a fair coin {n} times. What is the probability of getting exactly {k} heads? "
                f"Answer as a simplified fraction like 'a/b' (or an integer if denominator is 1)."),
        canonical_answer=ans,
        check_method="exact_string",
        domain="math", subdomain="probability", difficulty=difficulty,
        source="programmatic",
    )


def gen_sequence(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Arithmetic or geometric sequence next-term."""
    if rng.random() < 0.5:
        a, d = rng.randint(1, 20), rng.randint(1, 10)
        seq = [a + i * d for i in range(5)]
        nxt = a + 5 * d
    else:
        a, r = rng.randint(1, 5), rng.randint(2, 4)
        seq = [a * r ** i for i in range(4)]
        nxt = a * r ** 4
    return GroundTruthQuestion(
        prompt=f"What is the next number in this sequence: {', '.join(map(str, seq))}? Give only the integer.",
        canonical_answer=str(nxt),
        check_method="numeric_exact",
        domain="reasoning", subdomain="sequence", difficulty=difficulty,
        source="programmatic",
    )


def gen_syllogism(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Valid or invalid syllogism with deterministic answer."""
    cases = [
        ("All A are B. All B are C. Therefore, all A are C.", "Valid"),
        ("All A are B. Some B are C. Therefore, all A are C.", "Invalid"),
        ("No A are B. All C are A. Therefore, no C are B.", "Valid"),
        ("Some A are B. Some B are C. Therefore, some A are C.", "Invalid"),
        ("All A are B. No B are C. Therefore, no A are C.", "Valid"),
        ("Some A are B. All B are C. Therefore, some A are C.", "Valid"),
        ("No A are B. No B are C. Therefore, no A are C.", "Invalid"),
    ]
    template, answer = rng.choice(cases)
    nouns = rng.sample(["cats", "dogs", "birds", "fish", "plants", "rocks",
                        "metals", "liquids", "toys", "tools"], 3)
    text = template.replace("A", nouns[0]).replace("B", nouns[1]).replace("C", nouns[2])
    return GroundTruthQuestion(
        prompt=text + " Answer exactly 'Valid' or 'Invalid'.",
        canonical_answer=answer,
        check_method="exact_string",
        domain="logic", subdomain="syllogism", difficulty=difficulty,
        source="programmatic",
    )


def gen_ordering(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Transitive ordering puzzle. The generator knows the order."""
    names = rng.sample(["Alice", "Bob", "Carol", "Dave", "Eve"], 4)
    # names are in order from tallest to shortest
    facts = [
        f"{names[0]} is taller than {names[1]}.",
        f"{names[1]} is taller than {names[2]}.",
        f"{names[2]} is taller than {names[3]}.",
    ]
    rng.shuffle(facts)
    prompt = " ".join(facts) + f" Who is tallest: {names[0]}, {names[1]}, {names[2]}, or {names[3]}? Answer with exactly the name."
    return GroundTruthQuestion(
        prompt=prompt,
        canonical_answer=names[0],
        check_method="exact_string",
        domain="reasoning", subdomain="ordering", difficulty=difficulty,
        source="programmatic",
    )


def gen_truth_table(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Evaluate a boolean expression for given P, Q."""
    exprs = [
        ("P and Q", lambda p, q: p and q),
        ("P or Q", lambda p, q: p or q),
        ("not P or Q", lambda p, q: (not p) or q),
        ("P and not Q", lambda p, q: p and not q),
        ("(P or Q) and not (P and Q)", lambda p, q: (p or q) and not (p and q)),
    ]
    expr, fn = rng.choice(exprs)
    p = rng.choice([True, False])
    q = rng.choice([True, False])
    ans = "True" if fn(p, q) else "False"
    return GroundTruthQuestion(
        prompt=(f"Let P = {p}, Q = {q}. Evaluate: {expr}. "
                f"Answer exactly 'True' or 'False'."),
        canonical_answer=ans,
        check_method="exact_string",
        domain="logic", subdomain="propositional", difficulty=difficulty,
        source="programmatic",
    )


def gen_derivative(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Polynomial derivative with sympy-checkable answer."""
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    c = rng.randint(1, 9)
    n = rng.randint(2, 5) if difficulty != "easy" else 2
    prompt = f"Compute the derivative of {a}*x**{n} + {b}*x - {c} with respect to x."
    d_coeff = a * n
    n_minus = n - 1
    if n_minus == 1:
        ans = f"{d_coeff}*x + {b}"
    else:
        ans = f"{d_coeff}*x**{n_minus} + {b}"
    return GroundTruthQuestion(
        prompt=prompt + " Give the expression only (no explanation).",
        canonical_answer=ans,
        check_method="sympy_equiv",
        domain="math", subdomain="calculus", difficulty=difficulty,
        source="programmatic",
    )


def gen_code_arithmetic(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    """Generate a simple 'implement this pure function' problem with unit tests.
    The generator KNOWS the expected behavior because it specifies it exactly."""
    specs = [
        ("add_one", "returns n + 1",
         [("add_one(0)", 1), ("add_one(5)", 6), ("add_one(-3)", -2)]),
        ("double_it", "returns 2*n",
         [("double_it(0)", 0), ("double_it(7)", 14), ("double_it(-4)", -8)]),
        ("abs_val", "returns absolute value of n",
         [("abs_val(5)", 5), ("abs_val(-5)", 5), ("abs_val(0)", 0)]),
        ("square", "returns n**2",
         [("square(3)", 9), ("square(-4)", 16), ("square(0)", 0)]),
        ("cube", "returns n**3",
         [("cube(2)", 8), ("cube(-3)", -27), ("cube(0)", 0)]),
        ("max_of_two", "returns max of a and b",
         [("max_of_two(3, 5)", 5), ("max_of_two(-1, -9)", -1), ("max_of_two(7, 7)", 7)]),
        ("min_of_two", "returns min of a and b",
         [("min_of_two(3, 5)", 3), ("min_of_two(-1, -9)", -9), ("min_of_two(7, 7)", 7)]),
        ("is_even", "returns True if n is even, else False",
         [("is_even(2)", True), ("is_even(3)", False), ("is_even(0)", True)]),
        ("is_odd", "returns True if n is odd, else False",
         [("is_odd(2)", False), ("is_odd(3)", True), ("is_odd(0)", False)]),
        ("sum_list", "returns the sum of the list elements",
         [("sum_list([])", 0), ("sum_list([1,2,3])", 6), ("sum_list([-1,1])", 0)]),
        ("product_list", "returns the product of the list elements (empty list -> 1)",
         [("product_list([])", 1), ("product_list([2,3,4])", 24), ("product_list([1,0,5])", 0)]),
        ("length", "returns len(s) for a string s",
         [("length('')", 0), ("length('abc')", 3), ("length('xy z')", 4)]),
        ("last_elem", "returns the last element of a non-empty list",
         [("last_elem([1,2,3])", 3), ("last_elem([42])", 42), ("last_elem(['a','b'])", "'b'")]),
        ("first_elem", "returns the first element of a non-empty list",
         [("first_elem([1,2,3])", 1), ("first_elem([42])", 42), ("first_elem(['a','b'])", "'a'")]),
        ("string_length", "returns len(s)",
         [("string_length('hi')", 2), ("string_length('')", 0)]),
        ("negate", "returns -n",
         [("negate(5)", -5), ("negate(-3)", 3), ("negate(0)", 0)]),
        ("add", "returns a + b",
         [("add(2, 3)", 5), ("add(-1, 1)", 0), ("add(0, 0)", 0)]),
        ("multiply", "returns a * b",
         [("multiply(2, 3)", 6), ("multiply(-2, 4)", -8), ("multiply(0, 9)", 0)]),
        ("average_two", "returns (a + b) / 2 as a float",
         [("average_two(2, 4)", 3.0), ("average_two(1, 2)", 1.5)]),
        ("is_positive", "returns True if n > 0, else False",
         [("is_positive(1)", True), ("is_positive(0)", False), ("is_positive(-1)", False)]),
        ("count_positive", "returns how many elements of lst are > 0",
         [("count_positive([1,-1,2,-2,3])", 3), ("count_positive([])", 0)]),
        ("concat_strings", "returns a + b (string concatenation)",
         [("concat_strings('ab','cd')", "'abcd'"), ("concat_strings('','x')", "'x'")]),
        ("contains", "returns True if x in lst, else False",
         [("contains([1,2,3], 2)", True), ("contains([1,2,3], 9)", False), ("contains([], 0)", False)]),
        ("head", "returns lst[0] for non-empty lst",
         [("head([7,8,9])", 7), ("head([0])", 0)]),
        ("tail", "returns lst[1:] for non-empty lst",
         [("tail([1,2,3])", [2, 3]), ("tail([5])", [])]),
    ]
    name, spec, tests = rng.choice(specs)
    test_lines = [f"assert {call} == {expected}" for call, expected in tests]
    return GroundTruthQuestion(
        prompt=(f"Write a Python function `{name}` that {spec}. "
                f"Provide the function in a ```python``` block."),
        canonical_answer=name,
        check_method="code_unit_tests",
        domain="code", subdomain="implementation", difficulty=difficulty,
        source="programmatic",
        unit_tests=test_lines,
        entry_point=name,
    )


def gen_percentage(rng: random.Random, difficulty: str = "medium") -> GroundTruthQuestion:
    base = rng.randint(50, 500)
    pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 75])
    ans = Fraction(base * pct, 100)
    if ans.denominator == 1:
        ans_str = str(ans.numerator)
    else:
        ans_str = f"{float(ans):.2f}"
    return GroundTruthQuestion(
        prompt=f"What is {pct}% of {base}? Give only the number.",
        canonical_answer=ans_str,
        check_method="numeric_exact",
        domain="math", subdomain="percentage", difficulty=difficulty,
        source="programmatic",
    )


# ---------------------------------------------------------------------------
# Bank assembly
# ---------------------------------------------------------------------------

def _gsm8k_to_questions() -> list[GroundTruthQuestion]:
    out = []
    for prompt, ans in _GSM8K_STYLE:
        out.append(GroundTruthQuestion(
            prompt=prompt + " Give only the final numeric answer.",
            canonical_answer=ans,
            check_method="numeric_exact",
            domain="math", subdomain="word_problem", difficulty="medium",
            source="gsm8k_style",
        ))
    return out


def _math_to_questions() -> list[GroundTruthQuestion]:
    out = []
    for prompt, ans, method in _MATH_STYLE:
        out.append(GroundTruthQuestion(
            prompt=prompt,
            canonical_answer=ans,
            check_method=method,
            domain="math", subdomain="symbolic", difficulty="hard",
            source="math_style",
        ))
    return out


def _humaneval_to_questions() -> list[GroundTruthQuestion]:
    out = []
    for item in _HUMANEVAL_STYLE:
        out.append(GroundTruthQuestion(
            prompt=(item["prompt"] + "\n\nProvide the function in a ```python``` block. "
                    "Do not include tests or usage examples."),
            canonical_answer=item["entry_point"],
            check_method="code_unit_tests",
            domain="code", subdomain="implementation",
            difficulty=item.get("difficulty", "medium"),
            source="humaneval_style",
            unit_tests=list(item["tests"]),
            entry_point=item["entry_point"],
            forbidden_symbols=(list(item["forbidden_symbols"])
                               if item.get("forbidden_symbols") else None),
        ))
    return out


def _mmlu_to_questions() -> list[GroundTruthQuestion]:
    out = []
    for q, choices, correct, sub in _MMLU_STYLE:
        letters_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
        prompt = (f"{q}\n{letters_text}\n\n"
                  f"Respond with exactly one letter: A, B, C, or D.")
        # map subdomain to domain
        domain = {
            "science": "science", "history": "common_sense",
            "geography": "common_sense", "literature": "language_understanding",
            "computing": "code", "math": "math",
        }.get(sub, "common_sense")
        out.append(GroundTruthQuestion(
            prompt=prompt,
            canonical_answer=correct,
            check_method="exact_mc",
            domain=domain, subdomain=sub, difficulty="medium",
            source="mmlu_style",
        ))
    return out


def _logiqa_to_questions() -> list[GroundTruthQuestion]:
    out = []
    for q, choices, correct in _LOGIQA_STYLE:
        letters_text = "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
        prompt = (f"{q}\n{letters_text}\n\n"
                  f"Respond with exactly one letter from the choices above.")
        out.append(GroundTruthQuestion(
            prompt=prompt,
            canonical_answer=correct,
            check_method="exact_mc",
            domain="logic", subdomain="reasoning", difficulty="medium",
            source="logiqa_style",
        ))
    return out


# Generators per domain. Called with (rng, difficulty) -> GroundTruthQuestion.
_PROGRAMMATIC: dict[str, list[Callable]] = {
    "math": [gen_arithmetic, gen_linear_system, gen_gcd, gen_modular,
             gen_combinatorics, gen_coin_probability, gen_derivative,
             gen_percentage],
    "logic": [gen_syllogism, gen_truth_table],
    "reasoning": [gen_sequence, gen_ordering, gen_arithmetic],
    "code": [gen_code_arithmetic],
}


# Cached hard-coded banks (built lazily).
_CURATED_CACHE: Optional[dict[str, list[GroundTruthQuestion]]] = None


def _build_curated() -> dict[str, list[GroundTruthQuestion]]:
    global _CURATED_CACHE
    if _CURATED_CACHE is not None:
        return _CURATED_CACHE
    by_domain: dict[str, list[GroundTruthQuestion]] = {}
    for q in (_gsm8k_to_questions() + _math_to_questions() +
              _humaneval_to_questions() + _mmlu_to_questions() +
              _logiqa_to_questions()):
        by_domain.setdefault(q.domain, []).append(q)
    _CURATED_CACHE = by_domain
    return by_domain


def build_ground_truth_bank(
    domain: str,
    n_target: int,
    rng: random.Random,
    difficulty_mix: Optional[dict] = None,
    holdout_fraction: float = 0.0,
    holdout_rng: Optional[random.Random] = None,
) -> tuple[list[GroundTruthQuestion], list[GroundTruthQuestion]]:
    """Return (probe_questions, holdout_questions) for a domain.

    - Curated hard-coded problems come FIRST (the most trustworthy ground truth).
    - Programmatic generators fill the remainder and provide fresh variants
      each cycle.
    - Held-out set is a disjoint subset of curated items (deterministic given
      `holdout_rng`) that is never used for the training-signal path.
    """
    curated = list(_build_curated().get(domain, []))
    rng.shuffle(curated)

    holdout: list[GroundTruthQuestion] = []
    if holdout_fraction > 0 and curated:
        n_hold = max(1, int(len(curated) * holdout_fraction))
        hr = holdout_rng or random.Random(hash(domain) & 0xFFFFFFFF)
        # Deterministic holdout selection by domain
        shuffled = curated[:]
        hr.shuffle(shuffled)
        holdout = shuffled[:n_hold]
        holdout_set = {id(q) for q in holdout}
        curated = [q for q in curated if id(q) not in holdout_set]

    probe: list[GroundTruthQuestion] = list(curated[:n_target])

    # Programmatic top-off
    gens = _PROGRAMMATIC.get(domain, [])
    if gens and len(probe) < n_target:
        difficulty_mix = difficulty_mix or {
            "easy": 0.25, "medium": 0.40, "hard": 0.25, "expert": 0.10}
        attempts = 0
        seen_prompts: set[str] = {q.prompt for q in probe}
        while len(probe) < n_target and attempts < n_target * 4:
            attempts += 1
            r = rng.random()
            cum = 0.0
            diff = "medium"
            for d in ("easy", "medium", "hard", "expert"):
                cum += difficulty_mix.get(d, 0.0)
                if r <= cum:
                    diff = d
                    break
            gen = rng.choice(gens)
            try:
                q = gen(rng, diff)
            except Exception:
                continue
            if not q or q.prompt in seen_prompts:
                continue
            seen_prompts.add(q.prompt)
            probe.append(q)

    return probe, holdout


def total_curated_count(domain: str) -> int:
    return len(_build_curated().get(domain, []))


# ---------------------------------------------------------------------------
# Rigorous grading — no substring fallback for final correctness.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_final_numeric(text: str) -> Optional[float]:
    """Pull the LAST numeric token from `text`, which is typically where a
    model writes its final answer. Strip commas used as thousands separators."""
    if not text:
        return None
    # Prefer a marker like 'answer:' / 'answer is' / '= X' / last line
    markers = ["final answer:", "answer:", "answer is", "result:", "="]
    lower = text.lower()
    # Find the last occurrence among markers
    best_idx = -1
    for m in markers:
        i = lower.rfind(m)
        if i > best_idx:
            best_idx = i + len(m)
    search_text = text[best_idx:] if best_idx >= 0 else text
    matches = _NUM_RE.findall(search_text)
    if not matches:
        matches = _NUM_RE.findall(text)
    if not matches:
        return None
    last = matches[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def _extract_mc_letter(text: str) -> Optional[str]:
    """Pull the MC letter the model settled on (A/B/C/D)."""
    if not text:
        return None
    # Look for explicit final-answer markers first
    for marker in ("final answer:", "answer:", "answer is", "the answer is",
                   "correct answer is", "choose"):
        idx = text.lower().rfind(marker)
        if idx >= 0:
            tail = text[idx + len(marker):]
            m = re.search(r"\b([A-D])\b", tail)
            if m:
                return m.group(1).upper()
    # Fall back to the LAST standalone A-D letter
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
    if matches:
        return matches[-1].upper()
    return None


def _check_numeric_exact(response: str, canonical: str, tol: float = 1e-4) -> bool:
    try:
        want = float(canonical.replace(",", ""))
    except ValueError:
        # Non-numeric canonical — fall back to exact string
        return canonical.strip() in response
    got = _extract_final_numeric(response)
    if got is None:
        return False
    # For integer-looking canonical, also require near-integer match
    if "." not in canonical:
        return abs(got - want) < 0.5  # integer rounding tolerance
    return isclose(got, want, rel_tol=tol, abs_tol=tol * max(1.0, abs(want)))


def _check_exact_mc(response: str, canonical: str) -> bool:
    got = _extract_mc_letter(response)
    return got == canonical.strip().upper()


def _check_exact_string(response: str, canonical: str) -> bool:
    """Loose exact-string: case-insensitive, whitespace/punct-normalized
    match anywhere in the FINAL portion of the response."""
    if not response:
        return False
    want = canonical.strip().lower()
    tail = response.strip().lower()
    # Check last 300 chars (final answer tends to be near end)
    tail_window = tail[-300:] if len(tail) > 300 else tail
    # Normalize whitespace
    want_n = re.sub(r"\s+", " ", want)
    tail_n = re.sub(r"\s+", " ", tail_window)
    # Prefer a word-boundary match so 'x=3' doesn't hit on 'x=13'
    escaped = re.escape(want_n)
    pattern = r"(?<!\w)" + escaped + r"(?!\w)"
    try:
        return bool(re.search(pattern, tail_n))
    except re.error:
        return want_n in tail_n


def _check_sympy_equiv(response: str, canonical: str) -> bool:
    """Uses sympy to check expression equivalence. Delegates to the existing
    sanitized check for safety — we replicate a tight version here."""
    try:
        from sympy.parsing.sympy_parser import parse_expr
        from sympy import simplify, expand, Symbol
    except Exception:
        return False

    def _extract(text: str) -> str:
        if not text:
            return ""
        markers = ["final answer:", "answer:", "=", "result:", "is:", " is "]
        lower = text.lower()
        best = -1
        for m in markers:
            i = lower.rfind(m)
            if i > best:
                best = i + len(m)
        seg = text[best:] if best >= 0 else text
        seg = seg.split("\n")[0]
        seg = re.split(r"[,.](?:\s|$)", seg)[0]
        seg = seg.strip().strip("`'\"")
        # Greedy-grab the longest trailing math-ish substring
        m = re.search(r"[\d\w()+\-*/^\s.]+$", seg)
        if m:
            seg = m.group(0)
        # Strip leading non-math words (keep leading sign/paren/digit/var).
        seg = re.sub(r"^[A-Za-z]+(?=\s)", "", seg).strip()
        return seg

    def _sanitize(t: str) -> str:
        t = t.replace("^", "**").replace(" ", "")
        # Insert explicit '*' for implicit multiplication: "6x" -> "6*x",
        # ")x" -> ")*x", "x(" -> "x*(".  Models write "6x + 4" etc.
        # Negative lookahead preserves scientific notation — "6e5" must NOT
        # become "6*e5" (which sympy parses as 6×variable_e5 and silently
        # rejects numerically-correct answers like 6e5 == 600000).
        t = re.sub(r"(\d)(?!e\d|e[+\-]\d|E\d|E[+\-]\d)([a-zA-Z(])", r"\1*\2", t)
        t = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", t)
        t = re.sub(r"\be\*\*\(([^)]+)\)", r"exp(\1)", t)
        t = re.sub(r"\be\*\*(\w+)", r"exp(\1)", t)
        if re.search(r"(?:import|exec|eval|open|system|compile|getattr|setattr|"
                     r"__|lambda|exit|quit|input|print|breakpoint|globals|locals|vars)", t):
            return ""
        if re.search(r"\*\*\d{4,}", t):
            return ""
        if not re.match(r"^[\d\w+\-*/().]+$", t):
            return ""
        if len(t) > 200:
            return ""
        return t

    resp_raw = _extract(response)
    safe_r = _sanitize(resp_raw)
    safe_c = _sanitize(canonical)
    if not safe_r or not safe_c:
        return False
    try:
        x = Symbol("x")
        locals_ = {"x": x}
        # Empty global_dict breaks parse_expr (it injects Integer/Float calls
        # that must resolve). After the strict _sanitize() above the input is
        # safe; let parse_expr use its default globals.
        re_expr = parse_expr(safe_r, local_dict=locals_)
        ca_expr = parse_expr(safe_c, local_dict=locals_)
        diff = expand(re_expr - ca_expr)
        if diff == 0:
            return True
        # Try numeric at a few points
        try:
            for v in (0.37, 1.29, -0.73, 2.11):
                d = complex(diff.subs(x, v).evalf())
                if abs(d) > 1e-6:
                    return False
            return True
        except Exception:
            return False
    except Exception:
        return False


def _extract_code_block(response: str) -> str:
    if "```python" in response:
        parts = response.split("```python")[1:]
        blocks = [p.split("```")[0] for p in parts]
        blocks = [b for b in blocks if b.strip()]
        if blocks:
            return max(blocks, key=len).strip()
    if "```" in response:
        parts = response.split("```")
        blocks = []
        for i in range(1, len(parts), 2):
            b = parts[i]
            first_nl = b.find("\n")
            if first_nl != -1 and first_nl < 20 and b[:first_nl].strip().lower() in (
                "python", "py", "python3"
            ):
                b = b[first_nl + 1:]
            blocks.append(b.strip())
        blocks = [b for b in blocks if b]
        if blocks:
            return max(blocks, key=len)
    # Fall back: grab everything from first `def` onward
    if "def " in response:
        return response[response.index("def "):]
    return ""


def _has_forbidden_symbol(code: str, forbidden: list[str]) -> str:
    """Return the first matching forbidden symbol found in code's AST, or ''.

    Handles bare names ("sorted") as ast.Call on ast.Name, and dotted names
    ("list.sort", "list.index") as any ast.Attribute whose attr matches the
    suffix. The type prefix ("list.") is informational — we check the attribute
    name anywhere it's used as a method call, since the receiver type is not
    statically known.
    """
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    bare = {f for f in forbidden if "." not in f}
    attrs = {f.split(".", 1)[1] for f in forbidden if "." in f}
    for node in ast.walk(tree):
        # sorted(...) / any bare call
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in bare:
                return node.func.id
            if isinstance(node.func, ast.Attribute) and node.func.attr in attrs:
                return f".{node.func.attr}"
        # Also catch bare Name references even outside Call (e.g., sorted passed as arg).
        if isinstance(node, ast.Name) and node.id in bare:
            return node.id
    return ""


def _check_code_unit_tests(response: str, tests: list[str], entry_point: str,
                            timeout_s: int = 8,
                            forbidden_symbols: list[str] = None) -> bool:
    """Run unit tests against the model's code in the shared sandbox."""
    from ..utils.sandbox import run_python_sandboxed
    code = _extract_code_block(response)
    if not code:
        return False
    # Basic syntactic sanity
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return False
    # Require the entry_point function to be defined at top level
    if entry_point and not re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", code):
        # Allow class methods: `class X: def entry(...)` is not acceptable — model was
        # asked for a free function. Reject.
        return False
    # H2: prompt-level algorithmic constraints that unit tests can't enforce
    # (e.g., "without built-in sort", "binary search in O(log n)"). Reject the
    # response at AST level if it calls any forbidden symbol — faster and safer
    # than trying to infer complexity from execution traces.
    if forbidden_symbols:
        hit = _has_forbidden_symbol(code, forbidden_symbols)
        if hit:
            return False
    test_block = "\n".join(tests)
    full = code + "\n\n# unit tests\n" + test_block + "\n"
    ok, _ = run_python_sandboxed(full, timeout_s=timeout_s, memory_mb=256)
    return ok


def _score_code_unit_tests(
    response: str, tests: list[str], entry_point: str,
    *, timeout_s: int = 8, forbidden_symbols: list[str] = None,
) -> tuple[bool, float]:
    """Run unit tests individually and return (all_pass, pass_fraction).

    Task #28: populate a continuous score ∈ [0,1] on per_question records
    so the continuous-paired-delta MDE path (task #25 wedge 1) does not
    degenerate to {0, 1}. For code tasks, the natural continuous score is
    the fraction of unit tests the model's code passes — a 60%-passing
    solution is more informative than "wrong" (0) would suggest.

    Implementation: wrap each test in its own try/except inside one
    sandbox subprocess and count sentinel lines. Single subprocess keeps
    cost identical to the all-or-nothing grader (_check_code_unit_tests
    was already sandboxed once); per-test granularity comes for free.

    Returns (bool_all_pass, fraction_passed). Early-rejection paths
    (no code, syntax error, missing entry_point, forbidden symbols)
    return (False, 0.0) — consistent with the binary grader.
    """
    from ..utils.sandbox import run_python_sandboxed
    code = _extract_code_block(response)
    if not code:
        return False, 0.0
    try:
        compile(code, "<string>", "exec")
    except SyntaxError:
        return False, 0.0
    if entry_point and not re.search(rf"(?m)^def\s+{re.escape(entry_point)}\s*\(", code):
        return False, 0.0
    if forbidden_symbols:
        hit = _has_forbidden_symbol(code, forbidden_symbols)
        if hit:
            return False, 0.0

    if not tests:
        # No tests to grade. Match the binary grader's behavior: if
        # run_python_sandboxed accepts the raw code, count it as 1/1.
        ok, _ = run_python_sandboxed(code, timeout_s=timeout_s, memory_mb=256)
        return bool(ok), (1.0 if ok else 0.0)

    # Build a harness that runs each test in its own try/except and
    # prints a sentinel line per test. Counting is stdout-based so it
    # survives subprocess death (partial stdout buffered through
    # run_python_sandboxed's tail return).
    harness_lines = [code, ""]
    for i, t in enumerate(tests):
        # Each test becomes:
        #   try:
        #       <test>
        #       print("__TASK28_PASS__", i)
        #   except Exception:
        #       print("__TASK28_FAIL__", i)
        # Indentation is critical — the test body is often `assert ...`
        # which must be inside the try block.
        harness_lines.append("try:")
        for line in t.splitlines() or [""]:
            harness_lines.append("    " + line)
        harness_lines.append(f"    print('__TASK28_PASS__ {i}')")
        harness_lines.append("except Exception:")
        harness_lines.append(f"    print('__TASK28_FAIL__ {i}')")
    harness = "\n".join(harness_lines) + "\n"
    ok, tail = run_python_sandboxed(harness, timeout_s=timeout_s, memory_mb=256)
    # Process hard-fail (e.g. SIGKILL from memory limit) before any test
    # sentinel prints — treat as 0/N. Otherwise count sentinels even if
    # the process exited nonzero (an individual failing assert inside the
    # try/except shouldn't crash the harness).
    if not tail:
        return False, 0.0
    passes = sum(1 for _ in re.finditer(r"__TASK28_PASS__ \d+", tail))
    fails = sum(1 for _ in re.finditer(r"__TASK28_FAIL__ \d+", tail))
    total = passes + fails
    if total == 0:
        # No sentinels parsed — the harness itself failed (import error,
        # top-level SyntaxError after our compile() check, etc.). Report
        # 0/N conservatively.
        return False, 0.0
    # Guard against over-counting (e.g. tests that themselves print
    # "__TASK28_PASS__"). Cap at len(tests).
    if total > len(tests):
        # Take the FIRST len(tests) sentinels in order.
        sentinels = re.findall(
            r"__TASK28_(PASS|FAIL)__ \d+", tail,
        )[:len(tests)]
        passes = sum(1 for s in sentinels if s == "PASS")
        total = len(sentinels)
    fraction = passes / total if total > 0 else 0.0
    all_pass = (passes == len(tests)) and (fails == 0)
    # Also require the sandbox ok=True when all tests passed — matches
    # the binary grader's semantics for the "correct" field.
    all_pass = all_pass and bool(ok)
    return all_pass, fraction


def _score_logprob_of_gold(
    model_loader, prompt: str, gold_answer: str,
) -> float:
    """Score a non-code item by the model's mean log-prob on the gold tokens.

    Motivation (ρ-killer fix): the continuous-paired-delta MDE at N=600
    depends on the paired-sample correlation ρ. When the per-item score
    collapses to {0, 1}, ρ is bounded by the Pearson correlation of two
    Bernoulli sequences — empirically ~0.46 on the live run (cycle 2).
    A continuous log-prob-of-gold signal is far smoother: adapters that
    nudge gold-token probabilities only a little still produce paired
    deltas, raising ρ toward 0.8+ (cf. Anthropic HH-RLHF paper: log-prob
    proxies typically achieve ρ in the 0.7–0.9 band on paired eval).

    Implementation: teacher-forced scoring via vLLM ``prompt_logprobs``.
    We submit ``prompt + "\\n" + gold_answer`` with ``max_tokens=1`` and
    ``prompt_logprobs=1`` — vLLM returns the per-token logprob of every
    prompt token without running the decoder. We average the logprobs
    over the gold-answer tokens only (the question tokens are discarded)
    and return ``exp(mean_logprob)`` ∈ [0, 1]: the geometric mean of
    per-token probabilities, i.e. the model's self-normalised
    probability mass on the gold sequence per token.

    Normalisation choice: ``exp(mean_logprob)`` beats raw sum-logprob
    because long gold strings otherwise dominate variance. It also beats
    ``-NLL / T`` (no absolute scale) and rank-based scoring (collapses
    to binary for greedy-vs-perturbed comparisons). exp(mean) is bounded
    [0, 1], dimensionless, and monotone in per-token quality — the three
    properties the paired-delta needs.

    Robust fallback: returns ``float('nan')`` when vLLM did not return
    logprobs (older versions, HF-backend fallback, or the loader lacks
    ``_llm``). The caller collapses NaN to the binary score so nothing
    regresses — this function is strictly additive.
    """
    if model_loader is None:
        return float('nan')
    # Require vLLM backend with an active LLM instance; HF fallback
    # doesn't expose prompt_logprobs cheaply.
    llm = getattr(model_loader, '_llm', None)
    sp_cls = getattr(model_loader, '_sampling_params_cls', None)
    tok = getattr(model_loader, '_tokenizer', None) or getattr(
        model_loader, 'tokenizer', None,
    )
    if llm is None or sp_cls is None or tok is None:
        return float('nan')
    if not gold_answer:
        return float('nan')

    try:
        # Tokenise prompt and (prompt + newline + gold) so we know the
        # boundary — the gold-slice is the tail beyond len(prompt_ids).
        full_text = prompt.rstrip() + "\n" + gold_answer
        try:
            prompt_ids = tok(prompt.rstrip() + "\n",
                             add_special_tokens=False)["input_ids"]
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        except Exception:
            # Some tokenizers expose encode() directly.
            prompt_ids = tok.encode(prompt.rstrip() + "\n",
                                    add_special_tokens=False)
            full_ids = tok.encode(full_text, add_special_tokens=False)
        if len(full_ids) <= len(prompt_ids):
            return float('nan')
        n_gold = len(full_ids) - len(prompt_ids)

        params = sp_cls(
            max_tokens=1, temperature=0.0, top_p=1.0,
            prompt_logprobs=1, logprobs=0,
        )
        outputs = llm.generate([full_text], params)
        if not outputs:
            return float('nan')
        plp = getattr(outputs[0], 'prompt_logprobs', None)
        if not plp:
            return float('nan')
        # plp is a list with one entry per prompt token; the first token
        # has no predecessor so its entry is None. Take the tail n_gold
        # entries (these are the gold tokens under the model).
        gold_slice = plp[-n_gold:]
        total, count = 0.0, 0
        for tok_lp in gold_slice:
            if tok_lp is None:
                continue
            # vLLM's prompt_logprobs entry is a dict {token_id: Logprob}.
            # We want the logprob of the *actual* token at this position,
            # which vLLM always includes (it's the forced token).
            if isinstance(tok_lp, dict):
                # Smallest-key-with-max-logprob is unreliable; instead
                # find the entry whose .rank == 1 OR fall back to max.
                best_lp = None
                for v in tok_lp.values():
                    lp = getattr(v, 'logprob', None)
                    if lp is None:
                        lp = float(v) if isinstance(v, (int, float)) else None
                    if lp is None:
                        continue
                    if best_lp is None or lp > best_lp:
                        # NOTE: teacher-forced prompt always includes the
                        # ACTUAL token with its logprob; picking max is
                        # safe when prompt_logprobs=1 (only top-1 + actual
                        # are returned, and the actual token's logprob is
                        # the value we need — max covers both branches).
                        best_lp = lp
                if best_lp is None:
                    continue
                total += float(best_lp)
                count += 1
            else:
                lp = getattr(tok_lp, 'logprob', None)
                if lp is None:
                    continue
                total += float(lp)
                count += 1
        if count == 0:
            return float('nan')
        mean_lp = total / count
        import math
        score = math.exp(mean_lp)
        # Guard against numerical drift outside [0, 1].
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return float(score)
    except Exception:
        return float('nan')


def grade_ground_truth(question: GroundTruthQuestion, response: str,
                       code_timeout: int = 8) -> bool:
    """Dispatch to the appropriate rigorous grader. NEVER uses substring."""
    method = question.check_method
    if method == "numeric_exact":
        return _check_numeric_exact(response, question.canonical_answer, tol=question.tol)
    if method == "exact_mc":
        return _check_exact_mc(response, question.canonical_answer)
    if method == "exact_string":
        return _check_exact_string(response, question.canonical_answer)
    if method == "sympy_equiv":
        return _check_sympy_equiv(response, question.canonical_answer)
    if method == "code_unit_tests":
        return _check_code_unit_tests(
            response, question.unit_tests or [], question.entry_point,
            timeout_s=code_timeout,
            forbidden_symbols=question.forbidden_symbols or None,
        )
    return False


def grade_ground_truth_score(
    question: GroundTruthQuestion, response: str,
    code_timeout: int = 8,
    model_loader=None,
    use_logprob_continuous_score: bool = False,
) -> tuple[bool, float]:
    """Return (all_correct, continuous_score) for the response.

    Task #28: continuous score ∈ [0,1] per per_question record so the
    wedge-1 continuous-paired-delta MDE path (≤1% at N=600) sees real
    variance instead of degenerating to {0,1}.

    Per-method behavior:
      - code_unit_tests: fraction of tests passing (run individually in
        one sandbox subprocess — same cost as the binary grader).
      - numeric_exact / sympy_equiv / exact_mc / exact_string: score
        collapses to 1.0 when correct else 0.0. No rubric-based partial
        credit at this layer yet — log-prob-of-gold upgrade is a
        follow-up task. `correct` and `score` still agree bit-for-bit
        in the binary case, so downstream paired_delta behaves exactly
        like today on non-code domains.

    The returned `all_correct` is the same bool the legacy
    grade_ground_truth() returns, so existing callers that only need
    correctness can ignore the score.
    """
    method = question.check_method
    if method == "code_unit_tests":
        ok, frac = _score_code_unit_tests(
            response, question.unit_tests or [], question.entry_point,
            timeout_s=code_timeout,
            forbidden_symbols=question.forbidden_symbols or None,
        )
        return ok, float(frac)
    # Non-code methods: compute binary correctness, then optionally
    # augment with a log-prob-of-gold continuous score so downstream
    # paired-delta eval sees real variance (ρ-killer fix — see
    # _score_logprob_of_gold docstring for the math).
    correct = grade_ground_truth(question, response, code_timeout=code_timeout)
    binary_score = 1.0 if correct else 0.0
    if not use_logprob_continuous_score or model_loader is None:
        return correct, binary_score
    lp_score = _score_logprob_of_gold(
        model_loader, question.prompt, question.canonical_answer,
    )
    # Fallback: NaN / inf → binary (backward compat for old vLLM or
    # HF-backend fallback).
    import math
    if lp_score is None or math.isnan(lp_score) or math.isinf(lp_score):
        return correct, binary_score
    # Blend: preserve the correctness signal (bit cannot be swapped by
    # a lucky logprob) while injecting continuous variance. A correct
    # answer lands in [0.5, 1.0] scaled by per-token gold-prob; a wrong
    # answer lands in [0.0, 0.5]. The monotone map is
    #   score = 0.5 + 0.5 * lp_score  when correct
    #   score = 0.5 * lp_score        when wrong
    # so Δscore tracks Δ(gold-prob) continuously — the signal the MDE
    # path needs — and mean(score) still tracks mean(correct) within
    # ±0.25 (the inner width of each half-interval).
    if correct:
        score = 0.5 + 0.5 * float(lp_score)
    else:
        score = 0.5 * float(lp_score)
    # Clamp for numerical safety.
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return correct, float(score)


def question_to_dict(q: GroundTruthQuestion) -> dict:
    """Serialize to the dict format used by DiagnosticsEngine._probe_domain."""
    return {
        "prompt": q.prompt,
        "expected": q.canonical_answer,
        "canonical_answer": q.canonical_answer,
        "check_method": q.check_method,
        # Map to a legacy check_type so DiagnosticResult/WeaknessReport
        # machinery stays backward-compatible.
        "check_type": {
            "numeric_exact": "numeric",
            "sympy_equiv": "math_equiv",
            "code_unit_tests": "code_executes",
            "exact_mc": "exact",
            "exact_string": "exact",
        }.get(q.check_method, "exact"),
        "subdomain": q.subdomain,
        "difficulty": q.difficulty,
        "source": q.source,
        "unit_tests": list(q.unit_tests) if q.unit_tests else [],
        "entry_point": q.entry_point,
        "forbidden_symbols": list(q.forbidden_symbols) if q.forbidden_symbols else [],
    }
