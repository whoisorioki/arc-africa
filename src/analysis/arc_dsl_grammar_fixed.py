"""
Fixed Context-Free Grammar (CFG) for the ARC Domain-Specific Language.

This grammar resolves the parsing ambiguity issues in the original grammar.
"""

ARC_DSL_GRAMMAR_FIXED = r"""
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
    // Note: PRIMITIVE_NAME must come before VARIABLE to avoid ambiguity
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
    print("--- Fixed ARC DSL Context-Free Grammar ---")
    print(ARC_DSL_GRAMMAR_FIXED)
    
    # Test the grammar
    from lark import Lark
    try:
        parser = Lark(ARC_DSL_GRAMMAR_FIXED, start='program')
        print("\n✅ Fixed grammar is syntactically valid and can be parsed by Lark.")
        
        # Test with a simple program
        test_program = 'colorfilter("blue")'
        tree = parser.parse(test_program)
        print(f"✅ Test program '{test_program}' parsed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error with fixed grammar: {e}") 