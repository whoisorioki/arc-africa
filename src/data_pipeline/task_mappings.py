import re
import sys
import os
import importlib.util


# Helper function to load modules from specific paths
def load_module_from_path(module_name, file_path):
    """Load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the external modules
external_re_arc_path = os.path.join(
    os.path.dirname(__file__), "..", "external", "re_arc"
)
# Update path references
arc_dsl_path = os.path.join(os.path.dirname(__file__), "..", "external", "arc_dsl")

# Load dependencies first
dsl_re_arc = load_module_from_path(
    "dsl_re_arc", os.path.join(external_re_arc_path, "dsl.py")
)
utils_re_arc = load_module_from_path(
    "utils_re_arc", os.path.join(external_re_arc_path, "utils.py")
)
dsl_arc_dsl = load_module_from_path("dsl_arc_dsl", os.path.join(arc_dsl_path, "dsl.py"))

# Add the modules to sys.modules so they can be imported
sys.modules["dsl"] = dsl_re_arc
sys.modules["utils"] = utils_re_arc

# Now load the main modules
generators = load_module_from_path(
    "generators", os.path.join(external_re_arc_path, "generators.py")
)
solvers = load_module_from_path("solvers", os.path.join(arc_dsl_path, "solvers.py"))


# Helper to extract task ID from function name (e.g., generate_00d62c1b or solve_00d62c1b)
def extract_task_id(fn_name):
    match = re.search(r"_(\w{8})$", fn_name)
    return match.group(1) if match else None


# Build generator map
gen_fn_map = {}
for name in dir(generators):
    if name.startswith("generate_"):
        fn = getattr(generators, name)
        task_id = extract_task_id(name)
        if task_id:
            gen_fn_map[task_id] = fn

# Build solver map
solver_fn_map = {}
for name in dir(solvers):
    if name.startswith("solve_"):
        fn = getattr(solvers, name)
        task_id = extract_task_id(name)
        if task_id:
            solver_fn_map[task_id] = fn

GENERATOR_MAP = gen_fn_map
SOLVER_MAP = solver_fn_map
