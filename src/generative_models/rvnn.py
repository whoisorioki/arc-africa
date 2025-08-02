"""
Implements a Recursive Neural Network (RvNN) for grammar-guided program generation.

This module provides the foundational architecture for Stage II of the Phase 4 plan.
The RvNN is designed to generate programs by recursively expanding production rules
from a given Context-Free Grammar (CFG), ensuring all outputs are syntactically valid.
"""

import torch
import torch.nn as nn
from lark import Lark, Tree
from typing import List, Dict

from src.analysis.arc_dsl_grammar import ARC_DSL_GRAMMAR


class Grammar:
    """A helper class to parse and manage the DSL grammar."""

    def __init__(self, grammar_str: str):
        """
        Initializes the Grammar helper.

        Args:
            grammar_str (str): The string representation of the Context-Free Grammar.
        """
        self.grammar_str = grammar_str
        self.parser = Lark(
            grammar_str, start="program", keep_all_tokens=True
        )  # Use default parser like working direct test
        self.productions = self._get_productions()
        self.prod_to_idx = {prod: i for i, prod in enumerate(self.productions)}
        self.idx_to_prod = {i: prod for i, prod in enumerate(self.productions)}
        self.vocab_size = len(self.productions)

    def _get_productions(self) -> List[str]:
        """Extracts all unique production rules from the grammar."""
        productions = set()
        for rule in self.parser.rules:
            origin = rule.origin.name

            # Using standard .name attributes for robust extraction
            expansion = " ".join(sym.name for sym in rule.expansion)
            productions.add(f"{origin} -> {expansion}")

        # Add productions for aliased rules manually if needed
        # Example: 'statement_list_recursive' alias
        productions.add("statement_list -> statement_list NEWLINE statement")

        return sorted(list(productions))

    def debug_print_productions(self, n: int = 20):
        """Prints the first N production rules for debugging."""
        print(
            f"--- First {min(n, self.vocab_size)} productions in grammar vocabulary (EXPECTED) ---"
        )
        for i in range(min(n, self.vocab_size)):
            print(f"  {i}: '{self.productions[i]}'")
        print("-------------------------------------------------")

    def parse(self, code: str) -> Tree:
        """Parses a string of code into an AST."""
        return self.parser.parse(code)

    def get_production_index(self, production_str: str) -> int:
        """Gets the index for a given production rule string."""
        return self.prod_to_idx[production_str]

    def get_production_from_index(self, index: int) -> str:
        """Gets the production rule string from an index."""
        return self.idx_to_prod[index]


class RvNNGenerator(nn.Module):
    """
    A Recursive Neural Network for generating programs from a grammar.

    This model learns to generate an Abstract Syntax Tree (AST) top-down,
    starting from the root and recursively applying production rules.
    """

    def __init__(self, grammar: Grammar, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.grammar = grammar
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for grammar production rules
        self.embedding = nn.Embedding(grammar.vocab_size, embedding_dim)

        # The core recursive cell (using an LSTM cell for state management)
        # It takes the parent's state and the current production rule as input
        self.recursive_cell = nn.LSTMCell(embedding_dim, hidden_dim)

        # Output layer to predict the next production rule
        self.output_layer = nn.Linear(hidden_dim, grammar.vocab_size)

    def forward(self, parent_hidden_state, production_idx):
        """
        Performs one step of the recursive generation.

        Given the hidden state of the parent node in the AST and the production
        rule that generated the current node, this function predicts the next
        production rule to apply.

        Args:
            parent_hidden_state (Tuple[torch.Tensor, torch.Tensor]): The (h, c) state from the parent's LSTM cell.
            production_idx (torch.Tensor): The index of the production rule to be embedded.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - The output logits for the next production rule.
                - The new hidden state (h, c) for the current node.
        """
        # Embed the current production rule
        embedded_prod = self.embedding(production_idx)

        # Get the new hidden state from the recursive cell
        h_next, c_next = self.recursive_cell(embedded_prod, parent_hidden_state)

        # Predict the output logits
        logits = self.output_layer(h_next)

        return logits, (h_next, c_next)


if __name__ == "__main__":
    print("--- Initializing Generative Model Components ---")

    # 1. Initialize the grammar
    grammar = Grammar(ARC_DSL_GRAMMAR)
    print(f"✅ Grammar loaded successfully.")
    print(f"   - Vocabulary size (number of production rules): {grammar.vocab_size}")

    # 2. Initialize the RvNN Generator
    embedding_dim = 128
    hidden_dim = 256
    model = RvNNGenerator(grammar, embedding_dim, hidden_dim)
    print(f"✅ RvNN Generator initialized.")
    print(f"   - Embedding Dimension: {embedding_dim}")
    print(f"   - Hidden Dimension: {hidden_dim}")

    # 3. Demonstrate a single forward pass
    print("\n--- Demonstrating a single generation step ---")
    # Initial state for the root of the AST (e.g., for the <program> node)
    initial_h = torch.zeros(1, hidden_dim)
    initial_c = torch.zeros(1, hidden_dim)
    initial_state = (initial_h, initial_c)

    # Let's assume the first production rule we apply is for <program> -> <statement_list>
    # (The actual training process would learn this mapping)
    start_prod_str = "program -> statement_list"
    start_prod_idx = torch.tensor(
        [grammar.get_production_index(start_prod_str)], dtype=torch.long
    )

    # Get the model's prediction for the next rule
    logits, new_state = model(initial_state, start_prod_idx)

    print(f"   - Input production: '{start_prod_str}' (Index: {start_prod_idx.item()})")
    print(f"   - Output logits shape: {logits.shape}")
    print(
        f"   - The model now predicts a distribution over all {grammar.vocab_size} rules for the next step."
    )

    # The training process would involve teaching the model to pick the correct
    # next rule from this distribution to build a valid and meaningful program.
