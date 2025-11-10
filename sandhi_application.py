#!/usr/bin/env python3
"""
Sanskrit Sandhi Applier

This script applies sandhi rules (both vowel and consonant) to Sanskrit text
where sandhi points are marked with '+' signs. Rules are read from an external
sandhi_logic.txt file.

Major improvements in v3:
- Dictionary-based rule lookup for O(1) performance
- Compiled regex patterns for efficiency
- Simplified text processing logic
- Better error handling
- Optional quick testing mode
"""

import re
import sys
from typing import Tuple, Optional, List, Set
from dataclasses import dataclass


@dataclass
class SandhiChange:
    """Records a single sandhi transformation."""
    original: str
    result: str
    rule_applied: str
    
    def __str__(self):
        return f"{self.original} -> {self.result}"


class SandhiApplier:
    """Applies sandhi rules to Sanskrit text based on external rules file."""
    
    # Constants
    MAX_PASSES = 50
    TWO_CHAR_VOWELS = ['ai', 'au']
    TWO_CHAR_CONSONANTS = ['kh', 'gh', 'ch', 'jh', 'ṭh', 'ḍh', 'th', 'dh']
    SINGLE_CHAR_VOWELS = ['a', 'ā', 'i', 'ī', 'u', 'ū', 'e', 'o', 'ṛ', 'ṝ', 'ḷ', 'ḹ']
    SPECIAL_CONSONANTS = {'ṅ', 'n', 's'}
    
    def __init__(self, rules_file: str = "sandhi_logic.txt"):
        """Initialize the sandhi applier and load rules."""
        # All vowels for checking
        self.all_vowels = self.TWO_CHAR_VOWELS + self.SINGLE_CHAR_VOWELS
        
        # Compile regex patterns once
        self.bracket_pattern = re.compile(r'\[([^\]]+)\]')
        self.chain_pattern = re.compile(r'[^\s]+\+[^\s]+')
        
        # Load rules from file - now using dict for O(1) lookup
        self.rules_dict = {}  # (first, second) -> result
        self.exceptional_words = set()  # Words that use full-word matching
        self.exceptional_words_sorted = []  # Sorted by length for matching
        self.changes_log: List[SandhiChange] = []
        self.warnings: List[str] = []
        self.load_rules(rules_file)
        
        # Log of changes
        self.changes_log: List[SandhiChange] = []
        self.warnings: List[str] = []
    
    def load_rules(self, filename: str):
        """Load sandhi rules from external file into dictionary."""
        try:
            in_exceptions_section = False
            rules_count = 0
            
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Check for Exceptions section marker
                    if line.startswith('# Exceptions') or line.startswith('#Exceptions'):
                        in_exceptions_section = True
                        continue
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse rule line
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 3:
                        self.warnings.append(f"Line {line_num}: Invalid rule format (expected 3 parts): {line}")
                        continue
                    
                    first_pattern, second_pattern, result_pattern = parts
                    
                    # Track exceptional words
                    if in_exceptions_section and not first_pattern.startswith('['):
                        self.exceptional_words.add(first_pattern)
                    
                    # Parse patterns and add to dictionary
                    first_options = self._parse_pattern(first_pattern)
                    second_options = self._parse_pattern(second_pattern)
                    result_options = self._parse_pattern(result_pattern)
                    
                    # Store all rule combinations in dictionary
                    rules_count += self._add_rule_combinations(
                        first_options, second_options, result_options
                    )
            
            # Sort exceptional words by length (longest first) for matching
            self.exceptional_words_sorted = sorted(self.exceptional_words, key=len, reverse=True)
            
            print(f"Loaded {rules_count} sandhi rules from {filename}")
            if self.exceptional_words:
                print(f"Found {len(self.exceptional_words)} exceptional words: {sorted(self.exceptional_words)}")
            if self.warnings:
                print(f"\n⚠ {len(self.warnings)} warnings during rule loading:")
                for warning in self.warnings[:5]:  # Show first 5
                    print(f"  {warning}")
                if len(self.warnings) > 5:
                    print(f"  ... and {len(self.warnings) - 5} more")
        
        except FileNotFoundError:
            print(f"Error: Rules file '{filename}' not found!")
            sys.exit(1)
        except UnicodeDecodeError:
            print(f"Error: Rules file '{filename}' is not valid UTF-8!")
            sys.exit(1)
    
    def _add_rule_combinations(self, first_options: List[str], 
                               second_options: List[str], 
                               result_options: List[str]) -> int:
        """Add all combinations of rule options to dictionary. Returns count added."""
        count = 0
        
        if len(result_options) > 1:
            # Match result options with second options
            for first in first_options:
                for i, second in enumerate(second_options):
                    result_idx = i % len(result_options)
                    self.rules_dict[(first, second)] = result_options[result_idx]
                    count += 1
        else:
            # Standard expansion - all combinations
            for first in first_options:
                for second in second_options:
                    for result in result_options:
                        self.rules_dict[(first, second)] = result
                        count += 1
        
        return count
    
    def _parse_pattern(self, pattern: str) -> List[str]:
        """
        Parse a pattern like '[a|ā]' or 'k' or '[ā|ī]n' into a list of options.
        Special case: [END] is treated as a literal, not a bracket pattern.
        """
        if pattern == '[END]':
            return ['[END]']
        
        match = self.bracket_pattern.search(pattern)
        if not match:
            return [pattern]
        
        # Extract options and reconstruct with prefix/suffix
        options = [opt.strip() for opt in match.group(1).split('|')]
        prefix = pattern[:match.start()]
        suffix = pattern[match.end():]
        
        return [prefix + opt + suffix for opt in options]
    
    def get_ending(self, word: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the ending character(s) of a word.
        For exceptional words, returns the entire exceptional word.
        For special consonants (ṅ, n, s), also return the preceding vowel.
        Returns: (ending, preceding_vowel)
        """
        if not word:
            return (None, None)
        
        # Check exceptional words first (already sorted by length, longest first)
        exceptional = self._check_exceptional_ending(word)
        if exceptional:
            return (exceptional, None)
        
        # Check vowel+consonant patterns (like 'is', 'us', 'ās')
        vowel_consonant = self._check_vowel_consonant_pattern(word)
        if vowel_consonant:
            return vowel_consonant
        
        # Check two-character vowels
        for vowel in self.TWO_CHAR_VOWELS:
            if word.endswith(vowel):
                return (vowel, None)
        
        # Check two-character consonants
        for consonant in self.TWO_CHAR_CONSONANTS:
            if word.endswith(consonant):
                return (consonant, None)
        
        # Single character ending
        last_char = word[-1]
        
        # Special handling for ṅ, n, s - need preceding vowel
        if last_char in self.SPECIAL_CONSONANTS and len(word) >= 2:
            preceding_vowel = self._get_preceding_vowel(word[:-1])
            return (last_char, preceding_vowel)
        
        return (last_char, None)
    
    def _check_exceptional_ending(self, word: str) -> Optional[str]:
        """Check if word ends with an exceptional word."""
        for exceptional_word in self.exceptional_words_sorted:
            if word == exceptional_word:
                return exceptional_word
            if word.endswith(' ' + exceptional_word):
                return exceptional_word
        return None
    
    def _check_vowel_consonant_pattern(self, word: str) -> Optional[Tuple[str, None]]:
        """Check for vowel+consonant patterns like 'is', 'us', 'ās'."""
        if len(word) >= 2:
            last_two = word[-2:]
            if last_two[-1] in self.SPECIAL_CONSONANTS:
                return (last_two, None)
        return None
    
    def get_beginning(self, word: str) -> Optional[str]:
        """
        Get the beginning character(s) of a word.
        Returns [END] if word is empty or starts with non-alphabetic character.
        """
        if not word or not word[0].isalpha():
            return '[END]'
        
        # Check two-character vowels
        for vowel in self.TWO_CHAR_VOWELS:
            if word.startswith(vowel):
                return vowel
        
        # Check two-character consonants
        for consonant in self.TWO_CHAR_CONSONANTS:
            if word.startswith(consonant):
                return consonant
        
        return word[0]
    
    def _get_preceding_vowel(self, word: str) -> Optional[str]:
        """Get the last vowel from a word (for special n/s rules)."""
        if not word:
            return None
        
        # Check two-character vowels first
        for vowel in self.TWO_CHAR_VOWELS:
            if word.endswith(vowel):
                return vowel
        
        # Check single-character vowels
        if word[-1] in self.SINGLE_CHAR_VOWELS:
            return word[-1]
        
        # Recurse if last character is not a vowel
        return self._get_preceding_vowel(word[:-1]) if len(word) > 1 else None
    
    def find_matching_rule(self, first: str, second: str, 
                          preceding_vowel: Optional[str] = None) -> Optional[str]:
        """
        Find a matching sandhi rule using dictionary lookup (O(1)).
        For special cases with preceding vowel, check combined pattern first.
        """
        # Try combined pattern first if we have a preceding vowel
        if preceding_vowel:
            combined = preceding_vowel + first
            result = self.rules_dict.get((combined, second))
            if result is not None:
                return result
        
        # Standard lookup
        return self.rules_dict.get((first, second))
    
    def _expand_result_pattern(self, result: str, matched_char: str) -> str:
        """
        Expand [x|y|z] notation in result by substituting with the matched character.
        """
        if '[END]' in result:
            return result.replace('[END]', '').strip()
        
        return self.bracket_pattern.sub(matched_char, result)
    
    def apply_sandhi_at_junction(self, before_word: str, after_word: str) -> Tuple[str, str, bool]:
        """
        Apply sandhi rule at a junction between two words.
        Returns: (modified_before, modified_after, success)
        """
        if not before_word or not after_word:
            return (before_word, after_word, False)
        
        original_before = before_word
        original_after = after_word
        
        # Get the ending and beginning
        ending, preceding_vowel = self.get_ending(before_word)
        beginning = self.get_beginning(after_word)
        
        if ending is None or beginning is None:
            return (before_word, after_word, False)
        
        # Find matching rule
        result = self.find_matching_rule(ending, beginning, preceding_vowel)
        
        if result is None:
            # No rule found - log and return failure
            self.changes_log.append(SandhiChange(
                original=f"{original_before}+{original_after}",
                result=f"NO RULE FOUND ({ending}+{beginning})",
                rule_applied="none"
            ))
            return (before_word, after_word, False)
        
        # Expand result pattern
        result = self._expand_result_pattern(result, beginning)
        
        # Apply the transformation
        if preceding_vowel and len(ending) == 1 and ending in self.SPECIAL_CONSONANTS:
            before_prefix = before_word[:-(len(preceding_vowel) + len(ending))]
        else:
            before_prefix = before_word[:-len(ending)]
        
        # Handle [END] case
        after_suffix = after_word if beginning == '[END]' else after_word[len(beginning):]
        
        # Construct result
        modified_before = before_prefix + result
        modified_after = after_suffix
        
        # Log the change
        self.changes_log.append(SandhiChange(
            original=f"{original_before}+{original_after}",
            result=modified_before + modified_after,
            rule_applied=f"{ending}+{beginning}→{result}"
        ))
        
        return (modified_before.strip(), modified_after.strip(), True)
    
    def process_text(self, text: str) -> str:
        """Process entire text and apply sandhi rules."""
        self.changes_log = []
        
        changes_made = True
        pass_count = 0
        
        while changes_made and pass_count < self.MAX_PASSES:
            changes_made = False
            pass_count += 1
            
            # Find all word chains
            matches = list(self.chain_pattern.finditer(text))
            
            # Process from right to left to avoid position shifts
            for match in reversed(matches):
                chain = match.group(0)
                original_chain = chain
                words = chain.split('+')
                
                # Apply sandhi between adjacent words
                i = 0
                while i < len(words) - 1:
                    before, after = words[i], words[i + 1]
                    modified_before, modified_after, success = self.apply_sandhi_at_junction(before, after)
                    
                    if success:
                        # Merge the words
                        words[i] = modified_before + modified_after
                        words.pop(i + 1)
                        changes_made = True
                        # Don't increment i - check this position again
                    else:
                        # Keep the + and move on
                        i += 1
                
                # Reconstruct and replace if changed
                new_chain = '+'.join(words)
                if new_chain != original_chain:
                    text = text[:match.start()] + new_chain + text[match.end():]
                    changes_made = True
        
        if pass_count >= self.MAX_PASSES:
            print(f"⚠ Warning: Stopped after {self.MAX_PASSES} passes")
        
        return text
    
    def save_log(self, filename: str = "log.txt"):
        """Save the changes log to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            for change in self.changes_log:
                f.write(str(change) + '\n\n')
        print(f"Log saved to {filename} ({len(self.changes_log)} changes)")


def run_quick_tests():
    """Quick sanity checks to catch obvious breaks."""
    print("\n🧪 Running quick tests...\n")
    
    try:
        applier = SandhiApplier()
        
        # Test 1: Exception word bug fix
        result = applier.process_text("tattvatas+punar+ākāśe")
        assert "NO RULE FOUND (r+ā)" not in str(applier.changes_log), \
            "❌ Exception word bug: 'punar' not recognized in compound"
        print("✓ Test 1 passed: Exception words in compounds")
        
        # Test 2: Basic vowel sandhi
        applier2 = SandhiApplier()
        result = applier2.process_text("deva+atra")
        assert "deva+atra" not in result, \
            "❌ Basic vowel sandhi failed"
        print("✓ Test 2 passed: Basic vowel sandhi")
        
        # Test 3: Rules loaded correctly
        assert len(applier.rules_dict) > 0, \
            "❌ No rules loaded"
        print(f"✓ Test 3 passed: {len(applier.rules_dict)} rules loaded")
        
        print("\n✅ All quick tests passed!\n")
        return True
        
    except AssertionError as e:
        print(f"\n{e}\n")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}\n")
        return False


def main():
    """Main function to process Sanskrit text."""
    
    # Check for test mode
    if "--test" in sys.argv:
        success = run_quick_tests()
        sys.exit(0 if success else 1)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python sandhi_applier_v3.py <input_file> [rules_file]")
        print("       python sandhi_applier_v3.py --test")
        print("\nExample: python sandhi_applier_v3.py mytext.txt")
        print("         python sandhi_applier_v3.py mytext.txt custom_rules.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    rules_file = sys.argv[2] if len(sys.argv) > 2 else "sandhi_logic.txt"
    
    # Read input text
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: Input file '{input_file}' is not valid UTF-8!")
        sys.exit(1)
    
    print(f"Processing file: {input_file}")
    print(f"Using rules from: {rules_file}")
    print()
    
    # Create sandhi processor
    processor = SandhiApplier(rules_file)
    
    # Process the text
    result = processor.process_text(text)
    
    # Save the log
    processor.save_log("log.txt")
    
    # Determine output filename
    output_file = input_file.rsplit('.', 1)[0] + '_sandhi.txt'
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"\nOutput saved to: {output_file}")
    print(f"Total changes: {len(processor.changes_log)}")


if __name__ == "__main__":
    main()
