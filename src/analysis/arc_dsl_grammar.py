"""
Defines the Context-Free Grammar (CFG) for the ARC Domain-Specific Language.

This grammar formally specifies the syntactic structure of valid programs in our DSL.
It serves as a blueprint for the generative models in Phase 4, ensuring that
all synthesized programs are syntactically correct by construction. The grammar is
defined in a standard format that can be used by various parsing and generation tools.
"""

# The grammar is defined in a format compatible with the Lark parser.
# Non-terminals are lowercase, Terminals are uppercase.

ARC_DSL_GRAMMAR = r"""
    ?program: statement_list

    ?statement_list: statement
                   | statement_list "\n" statement -> statement_list_recursive

    ?statement: assignment
              | function_call
              | conditional_statement

    assignment: VARIABLE EQ expression

    ?expression: function_call
               | variable
               | literal

    function_call: PRIMITIVE_NAME LPAR [arg_list] RPAR

    ?arg_list: expression ("," expression)*

    conditional_statement: "conditional_transform" LPAR condition "," function_call "," function_call RPAR
    
    ?condition: expression comparison_op expression

    comparison_op: GT | LT | EQEQ | NE | GE | LE

    variable: VARIABLE

    ?literal: ESCAPED_STRING
            | SIGNED_NUMBER
            | COLOR
            | "input"

    // Define explicit Terminals to remove all ambiguity
    PRIMITIVE_NAME: "colorfilter" | "fill" | "rotate90" | "horizontal_mirror" | "vertical_mirror" | "replace_color" | "compose" | "chain" | "find_objects" | "select_largest_object" | "select_smallest_object" | "count_objects" | "find_symmetry_axis" | "complete_symmetry" | "find_pattern_repetition" | "align_objects" | "crop" | "remove" | "move" | "segment_grid" | "get_color" | "conditional_transform"
    COLOR: "'red'" | "'blue'" | "'green'" | "'yellow'" | "'black'" | "'cyan'" | "'magenta'" | "'white'" | "'gray'"
    VARIABLE: /[a-zA-Z_][a-zA-Z0-9_]*/
    
    // Comparison operators
    GT: ">"
    LT: "<"
    EQEQ: "=="
    NE: "!="
    GE: ">="
    LE: "<="
    EQ: "="

    // Structural tokens
    LPAR: "("
    RPAR: ")"
    COMMA: ","

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

if __name__ == "__main__":
    print("--- ARC DSL Context-Free Grammar ---")
    print(ARC_DSL_GRAMMAR)
    print(
        "\nThis grammar defines the syntactic rules for generating valid ARC programs."
    )
    print("It will be used by the Phase 4 generative models to ensure correctness.")

    # In a real application, this grammar would be loaded by a parser generator
    # like Lark or ANTLR to create a parser object.
    # from lark import Lark
    # try:
    #     parser = Lark(ARC_DSL_GRAMMAR, start='program')
    #     print("\n✅ Grammar is syntactically valid and can be parsed by Lark.")
    # except Exception as e:
    #     print(f"\n❌ Error parsing grammar: {e}")
