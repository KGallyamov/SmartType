import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math


class KeyboardMetrics:
    """Comprehensive metrics for evaluating keyboard layouts."""

    def __init__(self):
        # Standard QWERTY layout for reference
        self.qwerty = [
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
            ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
        ]

        # Finger assignments for standard touch typing
        # 0-3: left pinky to index, 4-5: thumbs, 6-9: right index to pinky
        self.finger_map = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 3,  # top row left
            (0, 5): 6, (0, 6): 6, (0, 7): 7, (0, 8): 8, (0, 9): 9,  # top row right
            (1, 0): 0, (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 3,  # home row left
            (1, 5): 6, (1, 6): 6, (1, 7): 7, (1, 8): 8, (1, 9): 9,  # home row right
            (2, 0): 0, (2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 4): 3,  # bottom row left
            (2, 5): 6, (2, 6): 6, (2, 7): 7, (2, 8): 8, (2, 9): 9,  # bottom row right
        }

        # Finger strength weights (1.0 = strongest)
        self.finger_strength = {
            0: 0.5,  # left pinky
            1: 0.7,  # left ring
            2: 0.9,  # left middle
            3: 1.0,  # left index
            4: 0.8,  # left thumb
            5: 0.8,  # right thumb
            6: 1.0,  # right index
            7: 0.9,  # right middle
            8: 0.7,  # right ring
            9: 0.5,  # right pinky
        }

        # Base typing speed for each finger (chars/second)
        self.finger_speed = {
            0: 2.0, 1: 2.5, 2: 3.0, 3: 3.5,  # left hand
            4: 3.0, 5: 3.0,  # thumbs
            6: 3.5, 7: 3.0, 8: 2.5, 9: 2.0  # right hand
        }

        # Penalty for row deviation from home row
        self.row_penalty = {0: 1.3, 1: 1.0, 2: 1.2}  # top, home, bottom

    def create_position_map(self, keyboard: List[List[str]]) -> Dict[str, Tuple[int, int]]:
        """Create a mapping from character to position."""
        pos_map = {}
        for i, row in enumerate(keyboard):
            for j, char in enumerate(row):
                if char and char != ' ':
                    pos_map[char.lower()] = (i, j)
        return pos_map

    def calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two key positions."""
        # Assuming standard key spacing: 1.9cm horizontal, 1.9cm vertical
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) * 1.9

    # ============= METRIC 1: Normalized Typing Time =============

    def normalized_typing_time(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate normalized typing time based on finger travel distance and speed.
        Lower is better.

        Formula: T = Σ(d_ij / v_f + t_press) / n_chars
        where d_ij = distance between keys i and j
              v_f = travel speed of finger f
              t_press = key press time
        """
        pos_map = self.create_position_map(keyboard)
        total_time = 0
        total_chars = 0

        for text in texts:
            text = text.lower()
            prev_pos = (1, 3)  # Start at 'f' (left index home position)

            for char in text:
                if char in pos_map:
                    curr_pos = pos_map[char]

                    # Calculate travel time
                    distance = self.calculate_distance(prev_pos, curr_pos)
                    finger = self.finger_map.get(curr_pos, 3)
                    travel_speed = 15.0  # cm/s average finger travel speed
                    travel_time = distance / travel_speed

                    # Calculate press time based on finger and row
                    press_time = 0.08 / self.finger_speed[finger]  # base press time
                    press_time *= self.row_penalty[curr_pos[0]]

                    total_time += travel_time + press_time
                    total_chars += 1
                    prev_pos = curr_pos

        return total_time / max(total_chars, 1)

    # ============= METRIC 2: Finger Stress Index =============

    def finger_stress_index(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate cumulative stress on fingers based on usage frequency and strength.
        Lower is better.

        Formula: S = Σ_f (usage_f^2 / strength_f * row_penalty_f)
        """
        pos_map = self.create_position_map(keyboard)
        finger_usage = defaultdict(int)
        finger_row_penalty = defaultdict(float)

        for text in texts:
            text = text.lower()
            for char in text:
                if char in pos_map:
                    pos = pos_map[char]
                    finger = self.finger_map.get(pos, 3)
                    finger_usage[finger] += 1
                    finger_row_penalty[finger] += self.row_penalty[pos[0]]

        total_chars = sum(finger_usage.values())
        if total_chars == 0:
            return 0

        stress = 0
        for finger, usage in finger_usage.items():
            usage_ratio = usage / total_chars
            avg_row_penalty = finger_row_penalty[finger] / max(usage, 1)
            # Quadratic stress model (stress increases non-linearly with usage)
            stress += (usage_ratio ** 2) / self.finger_strength[finger] * avg_row_penalty

        return stress

    # ============= METRIC 3: Bigram Transition Time =============

    def bigram_transition_time(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate average transition time between character pairs (bigrams).
        Considers same-finger penalties and hand alternation benefits.
        Lower is better.

        Formula: T_bigram = Σ(t_ij * freq_ij) / Σ(freq_ij)
        where t_ij includes distance, same-finger penalty, and alternation bonus
        """
        pos_map = self.create_position_map(keyboard)
        bigram_times = defaultdict(float)
        bigram_counts = defaultdict(int)

        for text in texts:
            text = text.lower()
            for i in range(len(text) - 1):
                if text[i] in pos_map and text[i + 1] in pos_map:
                    bigram = text[i] + text[i + 1]

                    pos1 = pos_map[text[i]]
                    pos2 = pos_map[text[i + 1]]

                    # Base transition time (distance-based)
                    distance = self.calculate_distance(pos1, pos2)
                    base_time = distance / 20.0  # 20 cm/s transition speed

                    # Same finger penalty (typing with same finger is slow)
                    finger1 = self.finger_map.get(pos1, 3)
                    finger2 = self.finger_map.get(pos2, 3)

                    if finger1 == finger2 and pos1 != pos2:
                        base_time *= 2.5  # Heavy penalty for same-finger different keys

                    # Hand alternation bonus
                    hand1 = 0 if finger1 < 5 else 1
                    hand2 = 0 if finger2 < 5 else 1
                    if hand1 != hand2:
                        base_time *= 0.8  # 20% speed bonus for alternating hands

                    bigram_times[bigram] += base_time
                    bigram_counts[bigram] += 1

        if sum(bigram_counts.values()) == 0:
            return 0

        total_time = sum(bigram_times[bg] for bg in bigram_times)
        total_count = sum(bigram_counts.values())

        return total_time / total_count

    # ============= METRIC 4: Home Row Usage =============

    def home_row_usage(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate percentage of keystrokes on the home row.
        Higher is better (we'll return negative for optimization).

        Formula: H = count(home_row_chars) / count(total_chars)
        """
        pos_map = self.create_position_map(keyboard)
        home_row_count = 0
        total_count = 0

        for text in texts:
            text = text.lower()
            for char in text:
                if char in pos_map:
                    pos = pos_map[char]
                    if pos[0] == 1:  # Home row
                        home_row_count += 1
                    total_count += 1

        if total_count == 0:
            return 0

        return -home_row_count / total_count  # Negative because higher is better

    # ============= METRIC 5: Hand Balance =============

    def hand_balance(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate balance of usage between left and right hands.
        Perfect balance = 0, imbalance increases the metric.

        Formula: B = |left_usage - right_usage|
        """
        pos_map = self.create_position_map(keyboard)
        left_usage = 0
        right_usage = 0

        for text in texts:
            text = text.lower()
            for char in text:
                if char in pos_map:
                    pos = pos_map[char]
                    finger = self.finger_map.get(pos, 3)
                    if finger < 5:
                        left_usage += 1
                    else:
                        right_usage += 1

        total = left_usage + right_usage
        if total == 0:
            return 0

        return abs(left_usage - right_usage) / total

    # ============= METRIC 6: Trigram Roll Efficiency =============

    def trigram_roll_efficiency(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Calculate efficiency of three-key sequences (rolls).
        Inward/outward rolls on same hand are efficient.
        Lower is better.

        Formula: R = Σ(roll_penalty * freq) / total_trigrams
        """
        pos_map = self.create_position_map(keyboard)
        trigram_penalty = 0
        trigram_count = 0

        for text in texts:
            text = text.lower()
            for i in range(len(text) - 2):
                chars = [text[i], text[i + 1], text[i + 2]]
                if all(c in pos_map for c in chars):
                    positions = [pos_map[c] for c in chars]
                    fingers = [self.finger_map.get(p, 3) for p in positions]

                    # Check if all on same hand
                    hands = [0 if f < 5 else 1 for f in fingers]
                    if hands[0] == hands[1] == hands[2]:
                        # Same hand trigram
                        # Check for roll (consecutive fingers)
                        if abs(fingers[1] - fingers[0]) == 1 and abs(fingers[2] - fingers[1]) == 1:
                            # Good roll
                            trigram_penalty += 0.8
                        elif fingers[0] == fingers[1] or fingers[1] == fingers[2]:
                            # Same finger repeated (bad)
                            trigram_penalty += 2.0
                        else:
                            # Random same-hand pattern
                            trigram_penalty += 1.2
                    else:
                        # Mixed hands (generally good)
                        trigram_penalty += 1.0

                    trigram_count += 1

        if trigram_count == 0:
            return 0

        return trigram_penalty / trigram_count

    # ============= METRIC 7: Awkward Bigram Penalty =============

    def awkward_bigram_penalty(self, keyboard: List[List[str]], texts: List[str]) -> float:
        """
        Penalize awkward bigrams (e.g., bottom row to top row with same hand).
        Lower is better.
        """
        pos_map = self.create_position_map(keyboard)
        awkward_penalty = 0
        bigram_count = 0

        for text in texts:
            text = text.lower()
            for i in range(len(text) - 1):
                if text[i] in pos_map and text[i + 1] in pos_map:
                    pos1 = pos_map[text[i]]
                    pos2 = pos_map[text[i + 1]]
                    finger1 = self.finger_map.get(pos1, 3)
                    finger2 = self.finger_map.get(pos2, 3)

                    # Same hand check
                    hand1 = 0 if finger1 < 5 else 1
                    hand2 = 0 if finger2 < 5 else 1

                    if hand1 == hand2:
                        row_jump = abs(pos1[0] - pos2[0])
                        if row_jump == 2:  # Top to bottom or vice versa
                            awkward_penalty += 2.0
                        elif row_jump == 1 and abs(pos1[1] - pos2[1]) > 2:
                            # Large horizontal + vertical movement
                            awkward_penalty += 1.5

                    bigram_count += 1

        if bigram_count == 0:
            return 0

        return awkward_penalty / bigram_count

    # ============= COMPOSITE METRIC =============

    def composite_score(self, keyboard: List[List[str]], texts: List[str],
                        weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted composite score combining all metrics.
        Lower is better.

        Default weights can be adjusted based on optimization goals.
        """
        if weights is None:
            weights = {
                'typing_time': 0.25,
                'finger_stress': 0.20,
                'bigram_time': 0.20,
                'home_row': 0.10,
                'hand_balance': 0.10,
                'trigram_roll': 0.10,
                'awkward_bigram': 0.05
            }

        scores = {
            'typing_time': self.normalized_typing_time(keyboard, texts),
            'finger_stress': self.finger_stress_index(keyboard, texts),
            'bigram_time': self.bigram_transition_time(keyboard, texts),
            'home_row': self.home_row_usage(keyboard, texts),  # Already negative
            'hand_balance': self.hand_balance(keyboard, texts),
            'trigram_roll': self.trigram_roll_efficiency(keyboard, texts),
            'awkward_bigram': self.awkward_bigram_penalty(keyboard, texts)
        }

        # Normalize scores (optional, helps with weight interpretation)
        composite = sum(scores[metric] * weights[metric] for metric in scores)

        return composite


# ============= USAGE EXAMPLE =============

def evaluate_keyboard(keyboard: List[List[str]], test_texts: List[str]) -> Dict[str, float]:
    """
    Evaluate a keyboard layout using all metrics.
    """
    metrics = KeyboardMetrics()

    results = {
        'Normalized Typing Time': metrics.normalized_typing_time(keyboard, test_texts),
        'Finger Stress Index': metrics.finger_stress_index(keyboard, test_texts),
        'Bigram Transition Time': metrics.bigram_transition_time(keyboard, test_texts),
        'Home Row Usage': -metrics.home_row_usage(keyboard, test_texts),  # Convert back to positive
        'Hand Balance': metrics.hand_balance(keyboard, test_texts),
        'Trigram Roll Efficiency': metrics.trigram_roll_efficiency(keyboard, test_texts),
        'Awkward Bigram Penalty': metrics.awkward_bigram_penalty(keyboard, test_texts),
        'Composite Score': metrics.composite_score(keyboard, test_texts)
    }

    return results


# Example usage
if __name__ == "__main__":
    qwerty = [
        ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'],
        ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
    ]

    dvorak = [
        [';', ',', '.', 'p', 'y', 'f', 'g', 'c', 'r', 'l'],
        ['a', 'o', 'e', 'u', 'i', 'd', 'h', 't', 'n', 's'],
        ['q', 'j', 'k', 'x', 'b', 'm', 'w', 'v', 'z', '/']
    ]

    worst_layout_ever = [
        ['t', 'n', 'w', 'm', 'l', 'c', 'b', 'p', 'r', 'h'],
        ['s', 'g', 'x', 'j', 'f', 'k', 'q', 'z', 'v', ';'],
        ['e', 'a', 'd', 'i', 'o', 'y', 'u', ',', ',', '/']
    ]

    test_texts = [
        "the quick brown fox jumps over the lazy dog",
        "python programming is fun and useful",
        "machine learning algorithms optimize keyboard layouts"
    ]

    print("QWERTY Evaluation:")
    qwerty_scores = evaluate_keyboard(qwerty, test_texts)
    for metric, score in qwerty_scores.items():
        print(f"  {metric}: {score:.4f}")

    print("\nDvorak Evaluation:")
    dvorak_scores = evaluate_keyboard(dvorak, test_texts)
    for metric, score in dvorak_scores.items():
        print(f"  {metric}: {score:.4f}")

    print("\nWORST Evaluation:")
    worst_scores = evaluate_keyboard(worst_layout_ever, test_texts)
    for metric, score in worst_scores.items():
        print(f"  {metric}: {score:.4f}")