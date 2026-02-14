
import sys
import logging
import traceback
from dotenv import load_dotenv

# Ensure we can import from benchwarmer package
sys.path.insert(0, ".")

# Load env vars
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

from benchwarmer.agents.intake import IntakeAgent
from benchwarmer.engine.runner import BenchmarkRunner
# Import the toy algorithms from server.py (need to redefine them or import if server was a module, 
# but server.py is a script. I'll clone them here for simplicity/fidelity).

from benchwarmer.algorithms.base import AlgorithmWrapper

class GreedyVertexCover(AlgorithmWrapper):
    name = "greedy_vc"
    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        covered: set[int] = set()
        cover: list[int] = []
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "greedy"}}

class RandomVertexCover(AlgorithmWrapper):
    name = "random_vc"
    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        import random
        covered: set[int] = set()
        cover: list[int] = []
        edges = list(instance["edges"])
        random.shuffle(edges)
        for edge in edges:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                chosen = random.choice([u, v])
                cover.append(chosen)
                covered.add(chosen)
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "random"}}

def main():
    try:
        query = "Evaluate max cut"
        print(f"Processing query: {query}")

        agent = IntakeAgent()
        print("Running IntakeAgent...")
        config = agent.run(query, interactive=False)
        print("Config generated:")
        print(config)

        runner = BenchmarkRunner(config)
        
        # Emulate server.py logic
        if config.problem_class == "minimum_vertex_cover":
            runner.register_algorithm(GreedyVertexCover())
            runner.register_algorithm(RandomVertexCover())
        elif config.problem_class == "maximum_cut":
            print("Problem class is maximum_cut. Registering VC algos as placeholders.")
            runner.register_algorithm(GreedyVertexCover())
            runner.register_algorithm(RandomVertexCover())
        else:
             print(f"Unknown problem class {config.problem_class}")
             runner.register_algorithm(GreedyVertexCover())
             runner.register_algorithm(RandomVertexCover())

        print("Running benchmark...")
        df = runner.run()
        print("Benchmark complete.")
        print(df)
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
