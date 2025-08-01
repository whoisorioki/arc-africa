from lark import Lark
from src.analysis.arc_dsl_grammar import ARC_DSL_GRAMMAR

parser = Lark(ARC_DSL_GRAMMAR, start="program")
print("Testing programs...")
programs = ['colorfilter("blue")', 'rotate90("cw")', 'fill("red")']
[parser.parse(p) for p in programs]
print("All tests passed!")
