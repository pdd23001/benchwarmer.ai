#!/usr/bin/env python3
"""
Benchwarmer.AI — Full end-to-end CLI.

1. Intake Agent: NL description → BenchmarkConfig (proposed)
2. Instance selection: generator (confirm/modify params) OR custom JSON
3. Execution Engine: runs benchmarks → Results DataFrame
4. Plot Agent: interactive analysis & visualization loop

Usage
-----
    ANTHROPIC_API_KEY=sk-... python scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



def _legacy_main() -> None:
    """Original step-by-step pipeline (used with --no-orchestrator)."""
    import argparse
    from benchwarmer.utils.loader import load_algorithm_from_file

    parser = argparse.ArgumentParser(description="Benchwarmer.AI")
    parser.add_argument("--custom", "-c", type=str, help="Path to custom algorithm file to benchmark against.")
    parser.add_argument("--papers", "-p", nargs="+", type=str, help="PDF papers to analyze for algorithm extraction.")
    parser.add_argument("--mode", "-m", type=str, default="local", choices=["local", "modal"], help="Execution mode: 'local' (default) or 'modal' (remote sandbox).")
    parser.add_argument("--intake-backend", type=str, default="claude", choices=["claude", "nemotron"], help="LLM backend for Intake Agent.")
    parser.add_argument("--nemotron-url", type=str, default="https://integrate.api.nvidia.com/v1", help="Base URL for Nemotron/OpenAI-compatible backend.")
    parser.add_argument("--nemotron-model", type=str, default="nvidia/nemotron-3-nano-30b-a3b", help="Model name for Nemotron backend.")
    parser.add_argument("--no-orchestrator", action="store_true", help="Use legacy step-by-step flow.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  🏋️  Benchwarmer.AI — Algorithm Benchmarking Platform")
    print("=" * 60)
    print()

    # 0. Load custom algorithm if provided
    init_algorithms = None
    if args.custom:
        try:
            print(f"🔹 Loading custom algorithm from: {args.custom}")
            custom_algo = load_algorithm_from_file(args.custom)
            init_algorithms = [custom_algo]
            print(f"✅ Loaded: {custom_algo.name}")
            print("🔹 Default baselines disabled.")
        except Exception as e:
            print(f"❌ Failed to load custom algorithm: {e}")
            sys.exit(1)

    # ── Step 1: Get problem description ──────────────────────
    print("Describe your optimization problem in natural language.")
    print("(Type your description, then press Enter twice to submit)\n")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            lines.pop()
            break
        lines.append(line)

    description = "\n".join(lines).strip()
    if not description:
        print("No description provided. Exiting.")
        return

    print(f"\n📩 Received description ({len(description)} chars)")
    print("-" * 60)

    # ── Step 2: Intake Agent → BenchmarkConfig (proposal) ────
    from benchwarmer.agents.intake import IntakeAgent
    from benchwarmer.agents.backends import ClaudeBackend, OpenAIBackend

    backend = None
    if args.intake_backend == "nemotron":
        print(f"🔹 Using Nemotron backend: {args.nemotron_model} at {args.nemotron_url}")
        try:
            backend = OpenAIBackend(
                base_url=args.nemotron_url,
                model=args.nemotron_model,
            )
        except ImportError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        # Default to Claude
        try:
            backend = ClaudeBackend()
        except ImportError as e:
            print(f"❌ {e}")
            sys.exit(1)

    agent = IntakeAgent(backend=backend)

    # Validate PDF paths if provided
    pdf_paths = None
    if args.papers:
        pdf_paths = []
        for p in args.papers:
            if os.path.exists(p):
                pdf_paths.append(p)
                print(f"  📄 {os.path.basename(p)}")
            else:
                print(f"  ❌ Not found: {p}")
        if not pdf_paths:
            pdf_paths = None

    result = agent.run(description, pdf_paths=pdf_paths)
    config = result.config
    algo_specs = result.algorithms

    print("\n✅ IntakeResult received!")
    if algo_specs:
        print(f"   📋 {len(algo_specs)} algorithm(s) extracted from papers:")
        for spec in algo_specs:
            print(f"      • {spec.name}: {spec.approach} (source: {spec.source})")
    print("-" * 60)

    # ── Step 3: Instance source selection ────────────────────
    config = _instance_selection(config)

    # ── Step 3.5: Create sandbox pool for modal mode ─────────
    pool = None
    if args.mode == "modal":
        from benchwarmer.engine.sandbox_pool import SandboxPool
        pool = SandboxPool()
        print("\n🔧 Modal mode: sandboxes will be created for each algorithm")
        print("   and reused for benchmarking.\n")

    # ── Step 3.6: Select & code extracted algorithms ────────────
    auto_coded = []
    if algo_specs:
        print()
        print("=" * 60)
        print("  📄 Algorithms Extracted from Papers")
        print("=" * 60)
        print()

        for i, spec in enumerate(algo_specs, 1):
            print(f"  [{i}] {spec.name}")
            print(f"      Approach: {spec.approach}")
            print(f"      Complexity: {spec.complexity}")
            print(f"      Source: {spec.source}")
            if spec.key_steps:
                print(f"      Steps: {spec.key_steps[0]}", end="")
                if len(spec.key_steps) > 1:
                    print(f" (+{len(spec.key_steps)-1} more)")
                else:
                    print()
            print()

        print("  Which algorithms would you like to code?")
        print("  Enter numbers (e.g. 1,3), 'all', or 'none':")
        choice = input("\n  ➤ ").strip().lower()

        # Parse selection
        selected_specs = []
        if choice == "all":
            selected_specs = list(algo_specs)
        elif choice in ("none", "n", "0", ""):
            selected_specs = []
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(",")]
                for idx in indices:
                    if 1 <= idx <= len(algo_specs):
                        selected_specs.append(algo_specs[idx - 1])
                    else:
                        print(f"  ⚠️  Skipping invalid index: {idx}")
            except ValueError:
                print("  ⚠️  Could not parse selection. Skipping all.")

        if selected_specs:
            from benchwarmer.agents.implementation import ImplementationAgent
            impl_agent = ImplementationAgent()

            print(f"\n  🤖 Coding {len(selected_specs)} selected algorithm(s)…\n")

            for spec in selected_specs:
                print(f"  Coding '{spec.name}' ({spec.source})…")
                spec_prompt = (
                    f"Implement this algorithm: {spec.name}\n\n"
                    f"Approach: {spec.approach}\n"
                    f"Complexity: {spec.complexity}\n\n"
                    f"Key steps:\n"
                )
                for i, step in enumerate(spec.key_steps, 1):
                    spec_prompt += f"  {i}. {step}\n"

                result = impl_agent.generate(
                    description=spec_prompt,
                    problem_class=config.problem_class,
                    pool=pool,
                )

                if result["success"]:
                    print(f"  ✅ {result['name']} generated and tested!")
                    auto_coded.append(result["algorithm"])
                else:
                    print(f"  ❌ Failed: {result['error']}")
                    print(f"     You can try implementing it manually below.")
        else:
            print("\n  Skipping paper algorithms.")

    # ── Step 4: Register algorithms ─────────────────────────
    # Merge: custom + auto-coded + builtins
    combined_init = []
    if init_algorithms:
        combined_init.extend(init_algorithms)
    if auto_coded:
        combined_init.extend(auto_coded)

    algorithms = _algorithm_registration(
        config,
        initial_algorithms=combined_init if combined_init else None,
        pool=pool,
    )

    if not algorithms:
        print("⚠️  No algorithms registered. Exiting.")
        if pool:
            pool.teardown_all_sync()
        return

    from benchwarmer.engine.runner import BenchmarkRunner

    runner = BenchmarkRunner(config)
    for algo in algorithms:
        runner.register_algorithm(algo)

    print(f"\n🚀 Running benchmark (mode={args.mode})…")
    try:
        df = runner.run(execution_mode=args.mode, sandbox_pool=pool)
    finally:
        # Tear down all sandboxes after benchmarking
        if pool:
            pool.teardown_all_sync()

    print(f"\n✅ Benchmark complete — {len(df)} result rows")
    print("-" * 60)

    _print_summary(df)

    # ── Step 5: Interactive analysis loop ────────────────────
    _analysis_loop(df, config)


# ──────────────────────────────────────────────────────────────
# Instance selection: generator vs custom
# ──────────────────────────────────────────────────────────────

def _instance_selection(config):
    """Ask user whether to use generators, custom instances, or a suite."""
    from benchwarmer.config import BenchmarkConfig, InstanceConfig, GeneratorConfig

    print()
    print("How would you like to provide problem instances?")
    print("  1) generator  — use the proposed generators (you can modify params)")
    print("  2) custom     — load your own instances from a JSON file")
    print("  3) suite      — pull from a benchmarking suite (DIMACS, Biq Mac, SNAP)")
    print()

    while True:
        choice = input("Choose [1/generator, 2/custom, 3/suite]: ").strip().lower()
        if choice in ("1", "generator", "g", "gen"):
            return _generator_flow(config)
        elif choice in ("2", "custom", "c"):
            return _custom_flow(config)
        elif choice in ("3", "suite", "s"):
            return _suite_flow(config)
        else:
            print("   Please enter 1, 2, or 3.")


def _generator_flow(config):
    """Show proposed generators, let user confirm or modify."""
    from benchwarmer.config import GeneratorConfig

    generators = config.instance_config.generators

    while True:
        # Display generator table
        print("\n📋 Proposed Generators:")
        print("-" * 70)
        print(f"  {'#':<4} {'Type':<20} {'Params':<25} {'Sizes':<20} {'Count'}")
        print("-" * 70)
        for i, g in enumerate(generators):
            params_str = json.dumps(g.params) if g.params else "{}"
            sizes_str = str(g.sizes)
            print(f"  {i:<4} {g.type:<20} {params_str:<25} {sizes_str:<20} {g.count_per_size}")
            if g.why:
                print(f"       ↳ {g.why}")
        print("-" * 70)

        total_instances = sum(
            len(g.sizes) * g.count_per_size for g in generators
        )
        runs_per = config.execution_config.runs_per_config
        print(f"\n  Total instances: {total_instances}")
        print(f"  Runs per instance: {runs_per}")
        print(f"  Timeout: {config.execution_config.timeout_seconds}s")
        print()

        action = input("Accept this configuration? (yes/modify/add/remove/config): ").strip().lower()

        if action in ("yes", "y", ""):
            break
        elif action in ("modify", "m"):
            _modify_generator(generators)
        elif action in ("add", "a"):
            _add_generator(generators)
        elif action in ("remove", "r", "delete"):
            _remove_generator(generators)
        elif action in ("config", "c", "cfg"):
            ec = config.execution_config
            print(f"\n   Current runs_per_instance: {ec.runs_per_config}")
            print(f"   Current timeout: {ec.timeout_seconds}s")
            print(f"   Current memory_limit: {ec.memory_limit_mb} MB\n")
            new_runs = input("   Runs per instance (or Enter to keep): ").strip()
            if new_runs:
                try:
                    ec.runs_per_config = int(new_runs)
                except ValueError:
                    print("   ⚠️  Invalid number.")
            new_timeout = input("   Timeout seconds (or Enter to keep): ").strip()
            if new_timeout:
                try:
                    ec.timeout_seconds = float(new_timeout)
                except ValueError:
                    print("   ⚠️  Invalid number.")
            print(f"   ✅ Execution config updated.")
        else:
            print("   Please enter 'yes', 'modify', 'add', 'remove', or 'config'.")

    print("✅ Generator configuration confirmed!")
    return config


def _nl_to_json(prompt: str) -> str | None:
    """Quick LLM call to interpret natural language into JSON/value."""
    try:
        from benchwarmer.agents.backends import ClaudeBackend
        backend = ClaudeBackend()
        response = backend.generate(
            messages=[{"role": "user", "content": prompt}],
            system="You are a concise helper. Return ONLY the requested value — no explanation, no markdown, no code fences.",
            max_tokens=256,
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text.strip()
    except Exception as e:
        logging.getLogger(__name__).debug("NL interpretation failed: %s", e)
        return None


def _modify_generator(generators: list):
    """Let user modify params of a specific generator. Accepts JSON or natural language."""
    from benchwarmer.config import GeneratorConfig

    if not generators:
        print("   No generators to modify.")
        return

    try:
        idx = int(input(f"   Which generator # to modify? (0-{len(generators)-1}): ").strip())
        if idx < 0 or idx >= len(generators):
            print("   Invalid index.")
            return
    except ValueError:
        print("   Invalid number.")
        return

    g = generators[idx]
    print(f"\n   Editing generator #{idx}: {g.type}")
    print(f"   Current params: {json.dumps(g.params)}")
    print(f"   Current sizes:  {g.sizes}")
    print(f"   Current count:  {g.count_per_size}")
    print()
    print("   💡 You can type JSON or natural language (e.g. 'make it denser')")
    print()

    # Modify params
    new_params = input("   New params (JSON/NL, or Enter to keep): ").strip()
    if new_params:
        try:
            g.params = json.loads(new_params)
        except json.JSONDecodeError:
            # Treat as natural language → ask LLM to interpret
            interpreted = _nl_to_json(
                f"Generator type: {g.type}\n"
                f"Current params: {json.dumps(g.params)}\n"
                f"User request: {new_params}\n\n"
                f"Return ONLY valid JSON for the updated params dict. "
                f"No explanation, just the JSON object."
            )
            if interpreted:
                try:
                    g.params = json.loads(interpreted)
                    print(f"   ✨ Interpreted: {json.dumps(g.params)}")
                except json.JSONDecodeError:
                    print("   ⚠️  Could not interpret — keeping existing params.")
            else:
                print("   ⚠️  Could not interpret — keeping existing params.")

    # Modify sizes
    new_sizes = input("   New sizes (numbers/NL, or Enter to keep): ").strip()
    if new_sizes:
        try:
            g.sizes = [int(s.strip()) for s in new_sizes.split(",")]
        except ValueError:
            # Treat as NL
            interpreted = _nl_to_json(
                f"Current sizes: {g.sizes}\n"
                f"User request: {new_sizes}\n\n"
                f"Return ONLY a JSON array of integer sizes. No explanation."
            )
            if interpreted:
                try:
                    g.sizes = json.loads(interpreted)
                    print(f"   ✨ Interpreted: {g.sizes}")
                except (json.JSONDecodeError, TypeError):
                    print("   ⚠️  Could not interpret — keeping existing sizes.")
            else:
                print("   ⚠️  Could not interpret — keeping existing sizes.")

    # Modify count
    new_count = input("   New count_per_size (number/NL, or Enter to keep): ").strip()
    if new_count:
        try:
            g.count_per_size = int(new_count)
        except ValueError:
            interpreted = _nl_to_json(
                f"Current count_per_size: {g.count_per_size}\n"
                f"User request: {new_count}\n\n"
                f"Return ONLY a single integer. No explanation."
            )
            if interpreted:
                try:
                    g.count_per_size = int(interpreted.strip())
                    print(f"   ✨ Interpreted: {g.count_per_size}")
                except ValueError:
                    print("   ⚠️  Could not interpret — keeping existing count.")
            else:
                print("   ⚠️  Could not interpret — keeping existing count.")

    print(f"   ✅ Generator #{idx} updated.")


def _add_generator(generators: list):
    """Add a new generator to the config."""
    from benchwarmer.config import GeneratorConfig
    from benchwarmer.generators import list_generators

    available = list_generators()
    print(f"\n   Available generators: {', '.join(available)}")

    gen_type = input("   Generator type: ").strip()
    if gen_type not in available:
        print(f"   ⚠️  Unknown generator '{gen_type}'. Available: {', '.join(available)}")
        return

    params_str = input("   Params (JSON, e.g. {\"p\": 0.3}): ").strip() or "{}"
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        print("   ⚠️  Invalid JSON — using empty params.")
        params = {}

    sizes_str = input("   Sizes (comma-separated, e.g. 50,100,200): ").strip() or "50,100,200"
    try:
        sizes = [int(s.strip()) for s in sizes_str.split(",")]
    except ValueError:
        print("   ⚠️  Invalid sizes — using [50, 100, 200].")
        sizes = [50, 100, 200]

    count_str = input("   Count per size (default 3): ").strip() or "3"
    try:
        count = int(count_str)
    except ValueError:
        count = 3

    new_gen = GeneratorConfig(type=gen_type, params=params, sizes=sizes, count_per_size=count)
    generators.append(new_gen)
    print(f"   ✅ Added generator: {gen_type}")


def _remove_generator(generators: list):
    """Remove a generator from the config."""
    if not generators:
        print("   No generators to remove.")
        return

    try:
        idx = int(input(f"   Which generator # to remove? (0-{len(generators)-1}): ").strip())
        if idx < 0 or idx >= len(generators):
            print("   Invalid index.")
            return
    except ValueError:
        print("   Invalid number.")
        return

    removed = generators.pop(idx)
    print(f"   ✅ Removed generator: {removed.type}")

    if not generators:
        print("   ⚠️  No generators left! Add at least one, or switch to custom instances.")


def _custom_flow(config):
    """Load custom instances from a JSON file."""
    from benchwarmer.utils.instance_loader import load_instances

    print("\n📂 Custom Instance Loading")
    print("   Your JSON file should contain one or more graph instances.")
    print("   Each instance needs 'nodes' and 'edges' keys.")
    print("   Example format:")
    print('   {"nodes": [0, 1, 2], "edges": [{"source": 0, "target": 1}]}')
    print()

    all_instances = []
    while True:
        path = input("   Path to JSON file (or 'done' to finish): ").strip()

        if path.lower() in ("done", "d", ""):
            if all_instances:
                break
            else:
                print("   ⚠️  No instances loaded yet. Provide at least one file.")
                continue

        try:
            instances = load_instances(path)
            all_instances.extend(instances)
            print(f"   ✅ Loaded {len(instances)} instance(s) from {path}")
            for inst in instances:
                n_nodes = len(inst.get("nodes", []))
                n_edges = len(inst.get("edges", []))
                name = inst.get("instance_name", "?")
                print(f"      • {name}: {n_nodes} nodes, {n_edges} edges")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"   ❌ Error: {e}")

    # Clear generators (custom-only mode) and inject custom instances
    config.instance_config.generators = []
    config.instance_config.custom_instances = all_instances

    print(f"\n✅ {len(all_instances)} custom instance(s) loaded!")
    return config

# ──────────────────────────────────────────────────────────────
# Suite flow — pull from benchmarking suites
# ──────────────────────────────────────────────────────────────

def _suite_flow(config):
    """Browse and download instances from benchmark suites."""
    from benchwarmer.utils.benchmark_suites import (
        list_suites, list_instances, fetch_instance,
    )

    print("\n📦 Benchmark Suite Browser")
    print("=" * 60)

    # Show suites relevant to this problem class
    suites = list_suites(problem_class=config.problem_class)

    if not suites:
        # Fall back to showing all suites
        print(f"   No suites specifically for '{config.problem_class}', showing all:")
        suites = list_suites()

    if not suites:
        print("   ⚠️  No benchmark suites available.")
        return config

    print()
    for i, suite in enumerate(suites):
        print(f"  {i + 1}) {suite['name']}")
        print(f"     {suite['description']}")
        print(f"     {suite['instance_count']} instances | problems: {', '.join(suite['problem_classes'])}")
        print()

    # Pick a suite
    while True:
        try:
            choice = input(f"Select suite (1-{len(suites)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(suites):
                selected_suite = suites[idx]["key"]
                break
            print(f"   Please enter 1-{len(suites)}")
        except ValueError:
            print("   Enter a number.")

    # Show instances in that suite
    instances = list_instances(selected_suite)
    print(f"\n📋 Available instances in {suites[idx]['name']}:")
    print("-" * 65)
    print(f"  {'#':<4} {'Name':<25} {'Nodes':<10} {'Description'}")
    print("-" * 65)
    for i, inst in enumerate(instances):
        print(f"  {i:<4} {inst['name']:<25} {inst.get('nodes', '?'):<10} {inst.get('description', '')}")
    print("-" * 65)
    print()
    print("Select instances:")
    print("  • Enter numbers separated by commas (e.g. 0,1,3)")
    print("  • Enter a range (e.g. 0-4)")
    print("  • Enter 'all' for all instances")
    print()

    while True:
        selection = input("Instances: ").strip().lower()
        if not selection:
            continue

        selected_indices = set()

        if selection == "all":
            selected_indices = set(range(len(instances)))
        else:
            for part in selection.split(","):
                part = part.strip()
                if "-" in part:
                    try:
                        start, end = part.split("-", 1)
                        selected_indices.update(range(int(start), int(end) + 1))
                    except ValueError:
                        print(f"   Invalid range: {part}")
                        continue
                else:
                    try:
                        selected_indices.add(int(part))
                    except ValueError:
                        print(f"   Invalid number: {part}")
                        continue

        # Validate indices
        valid = {i for i in selected_indices if 0 <= i < len(instances)}
        if not valid:
            print("   No valid instances selected. Try again.")
            continue

        if len(valid) < len(selected_indices):
            print(f"   ⚠️  Skipped {len(selected_indices) - len(valid)} out-of-range index(es).")

        break

    # Download and parse selected instances
    print(f"\n⬇️  Downloading {len(valid)} instance(s)…")
    loaded_instances = []

    for i in sorted(valid):
        inst = instances[i]
        print(f"   Fetching {inst['name']}…", end=" ", flush=True)
        try:
            parsed = fetch_instance(selected_suite, inst["name"])
            loaded_instances.append(parsed)
            print(f"✅ ({len(parsed['nodes'])} nodes, {len(parsed['edges'])} edges)")
        except Exception as e:
            print(f"❌ {e}")

    if not loaded_instances:
        print("   ⚠️  No instances loaded. Falling back to generators.")
        return config

    # Inject into config
    config.instance_config.generators = []
    config.instance_config.custom_instances = loaded_instances

    print(f"\n✅ {len(loaded_instances)} benchmark instance(s) loaded!")
    return config


# ──────────────────────────────────────────────────────────────
# Algorithm registration (Implementation Agent + built-in baselines)
# ──────────────────────────────────────────────────────────────

def _algorithm_registration(config, initial_algorithms=None, pool=None) -> list:
    """Interactive loop to register algorithms — built-in and LLM-generated."""
    if initial_algorithms is not None:
        algorithms = initial_algorithms
    else:
        algorithms = _get_builtin_algorithms(config.problem_class)

    print()
    print("=" * 60)
    print("  🧠 Algorithm Registration")
    print("=" * 60)
    print()

    if algorithms:
        print(f"Built-in baselines for '{config.problem_class}':")
        for algo in algorithms:
            print(f"   • {algo.name}")
        print()

    print("You can now describe additional algorithms in natural language.")
    print("The AI will implement them as working code and register them.")
    print()
    print("Examples:")
    if config.problem_class == "maximum_cut":
        print('  • "local search that flips nodes improving the cut"')
        print('  • "simulated annealing with temperature cooling"')
    elif config.problem_class == "minimum_vertex_cover":
        print('  • "2-approximation: pick both endpoints of each uncovered edge"')
        print('  • "LP relaxation rounding for vertex cover"')
    else:
        print('  • "greedy heuristic"')
        print('  • "randomized local search"')
    print()
    print("Commands:")
    print("  run       — start the benchmark with current algorithms")
    print("  list      — show registered algorithms")
    print("  remove #  — remove algorithm by number")
    print("  load <p>  — load 1+ PDF papers (e.g. `load paper.pdf`)")
    print()

    from benchwarmer.agents.implementation import ImplementationAgent
    impl_agent = ImplementationAgent()

    while True:
        try:
            user_input = input("🧠 Describe algorithm (or 'run'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("run", "start", "go", "benchmark"):
            if not algorithms:
                print("   ⚠️  No algorithms registered yet. Describe at least one.")
                continue
            break

        if lower in ("list", "ls", "show"):
            if algorithms:
                print("\n   Registered algorithms:")
                for i, algo in enumerate(algorithms):
                    print(f"   {i}. {algo.name}")
                print()
            else:
                print("   No algorithms registered yet.\n")
            continue

        if lower.startswith("remove") or lower.startswith("rm"):
            parts = lower.split()
            if len(parts) < 2:
                print("   Usage: remove <number>")
                continue
            try:
                idx = int(parts[1])
                if 0 <= idx < len(algorithms):
                    removed = algorithms.pop(idx)
                    print(f"   ✅ Removed: {removed.name}")
                else:
                    print(f"   Invalid index. Range: 0-{len(algorithms)-1}")
            except ValueError:
                print("   Invalid number.")
            continue

        # Check for PDF loading command
        is_load = lower.startswith("load ")
        is_paper = lower.startswith("paper ")
        
        if is_load or is_paper:
            import shlex
            try:
                # Parse command line like shell (handles quotes)
                parts = shlex.split(user_input)
                # parts[0] is the command, rest are arguments
                raw_paths = parts[1:]
            except ValueError as e:
                print(f"   ❌ Syntax error in command: {e}")
                continue

            if not raw_paths:
                print("   Usage: load <pdf_path> [pdf_path2 ...]")
                continue

            valid_paths = []
            total_size_mb = 0.0

            for p in raw_paths:
                # Handle potential quotes if shlex didn't catch them (redundant but safe)
                if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                   p = p[1:-1]
                
                if os.path.exists(p):
                    valid_paths.append(p)
                    total_size_mb += os.path.getsize(p) / (1024 * 1024)
                else:
                    print(f"   ❌ File not found: {p}")
            
            if not valid_paths:
                continue
            
            if len(valid_paths) < len(raw_paths):
                print(f"   ⚠️  Proceeding with {len(valid_paths)} valid file(s).")

            # Check total file size (API limits apply)
            if total_size_mb > 30:
                print(f"   ⚠️  Total size is large ({total_size_mb:.1f} MB). API might reject it.")
                if input("   Continue? [y/N]: ").lower() != "y":
                    continue

            print(f"   📄 Ready to process {len(valid_paths)} document(s) (will be sent to LLM).")
            
            # Ask for specific instructions regarding the papers
            print("   ❓ What algorithm from these papers should I implement?")
            specific_prompt = input("   > ").strip()
            if not specific_prompt:
                specific_prompt = "Implement the main algorithm described in these papers."

            print("🤖 Sending docs to agent for implementation...")
            result = impl_agent.generate(
                description=specific_prompt,
                problem_class=config.problem_class,
                pdf_paths=valid_paths,
                pool=pool,
            )
        else:
            # Treat as standard algorithm description → send to Implementation Agent
            print("🤖 Generating algorithm implementation…")
            result = impl_agent.generate(
                description=user_input,
                problem_class=config.problem_class,
                pool=pool,
            )

        if result["success"]:
            print(f"\n✅ Generated: {result['name']}")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)
            print(f"   Smoke test passed: {result['smoke_result']}")

            algorithms.append(result["algorithm"])
            print(f"\n   Registered! ({len(algorithms)} total algorithms)\n")
        else:
            print(f"\n❌ Failed: {result['error']}")
            if result.get("code"):
                print(f"\n   Generated code:")
                print("-" * 40)
                print(result["code"])
                print("-" * 40)
            if result.get("traceback"):
                print(result["traceback"])
            print("   Try a different description.\n")

    print(f"\n📋 Final algorithm list ({len(algorithms)}):")
    for algo in algorithms:
        print(f"   • {algo.name}")

    return algorithms


# ──────────────────────────────────────────────────────────────
# Built-in baseline algorithms
# ──────────────────────────────────────────────────────────────

def _get_builtin_algorithms(problem_class: str) -> list:
    """Return simple baseline algorithms for the given problem class."""
    from benchwarmer.algorithms.base import AlgorithmWrapper
    import random

    algorithms = []

    if problem_class == "maximum_cut":
        class RandomMaxCut(AlgorithmWrapper):
            name = "random_cut"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                partition = [random.choice([0, 1]) for _ in nodes]
                return {"solution": {"partition": partition}, "metadata": {}}

        class GreedyMaxCut(AlgorithmWrapper):
            name = "greedy_cut"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                edges = instance["edges"]
                n = len(nodes)
                # Map node object to 0..n-1 index if needed (though usually they match)
                node_idx = {nd: i for i, nd in enumerate(nodes)}

                # Build weighted adjacency: adj[i] = [(j, weight), ...]
                adj = [[] for _ in range(n)]
                for e in edges:
                    si = node_idx[e["source"]]
                    ti = node_idx[e["target"]]
                    w = e.get("weight", 1.0)
                    adj[si].append((ti, w))
                    adj[ti].append((si, w))

                partition = [0] * n
                # Iterate through nodes and place in the side that maximizes the cut 
                # with respect to already-placed neighbors (or all neighbors, in this simple greedy)
                for i in range(n):
                    weight_to_0 = 0.0
                    weight_to_1 = 0.0
                    
                    for j, w in adj[i]:
                        # Check partition of neighbor j
                        if partition[j] == 0:
                            weight_to_0 += w
                        else:
                            weight_to_1 += w
                    
                    # If neighbor is in 0, we add 'w' to cut if we pick 1.
                    # So weight_to_0 is the gain if we pick 1.
                    # weight_to_1 is the gain if we pick 0.
                    partition[i] = 1 if weight_to_0 > weight_to_1 else 0
                    
                return {"solution": {"partition": partition}, "metadata": {}}

        algorithms = [GreedyMaxCut(), RandomMaxCut()]

    elif problem_class == "minimum_vertex_cover":
        class GreedyVertexCover(AlgorithmWrapper):
            name = "greedy_cover"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                edges = instance["edges"]

                adj: dict[int, set[int]] = {n: set() for n in nodes}
                for e in edges:
                    adj[e["source"]].add(e["target"])
                    adj[e["target"]].add(e["source"])

                cover: set[int] = set()
                degree = {n: len(adj[n]) for n in nodes}

                while True:
                    best_node = max(
                        (n for n in nodes if n not in cover and degree[n] > 0),
                        key=lambda n: degree[n],
                        default=None,
                    )
                    if best_node is None:
                        break

                    cover.add(best_node)
                    for neighbor in adj[best_node]:
                        if neighbor not in cover:
                            degree[neighbor] -= 1
                    degree[best_node] = 0

                return {"solution": {"vertex_cover": sorted(cover)}, "metadata": {}}

        class RandomVertexCover(AlgorithmWrapper):
            name = "random_cover"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                edges = instance["edges"]
                cover: set[int] = set()
                for e in edges:
                    u, v = e["source"], e["target"]
                    if u not in cover and v not in cover:
                        cover.add(random.choice([u, v]))
                return {"solution": {"vertex_cover": sorted(cover)}, "metadata": {}}

        algorithms = [GreedyVertexCover(), RandomVertexCover()]

    return algorithms


# ──────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────

def _print_summary(df) -> None:
    """Print a concise summary table of benchmark results."""
    import pandas as pd

    success_df = df[df["status"] == "success"]
    if success_df.empty:
        print("⚠️  No successful runs to summarize.")
        # Show error breakdown so user can debug
        error_df = df[df["status"] != "success"]
        if not error_df.empty:
            print("\n❌ Error breakdown:")
            for algo, group in error_df.groupby("algorithm_name"):
                errors = group["error_message"].value_counts()
                print(f"\n   {algo}:")
                for msg, count in errors.items():
                    print(f"     [{count}×] {msg[:200]}")
            print()
        return

    summary = (
        success_df
        .groupby("algorithm_name")
        .agg(
            avg_objective=("objective_value", "mean"),
            std_objective=("objective_value", "std"),
            avg_time=("wall_time_seconds", "mean"),
            avg_memory=("peak_memory_mb", "mean"),
            success_rate=("status", "count"),
        )
    )
    total_runs = len(df.groupby("algorithm_name").size())
    summary["success_rate"] = (
        summary["success_rate"] / len(df) * len(summary) * 100
    ).round(1)

    print("\n📊 Summary Table:")
    print(summary.to_string())
    print()


# ──────────────────────────────────────────────────────────────
# Interactive analysis loop
# ──────────────────────────────────────────────────────────────

def _analysis_loop(df, config) -> None:
    """Interactive visualization & analysis with the Plot Agent."""
    from benchwarmer.agents.plot import PlotAgent

    print("=" * 60)
    print("  📈 Interactive Analysis — powered by Plot Agent")
    print("=" * 60)
    print()
    print("Ask for visualizations or analysis in natural language.")
    print("Examples:")
    print('  • "Show a bar chart comparing average objective by algorithm"')
    print('  • "Box plot of wall time by algorithm for each graph size"')
    print('  • "Summary table of results"')
    print('  • "How does objective scale with problem size?"')
    print()
    print("Commands:")
    print("  exit      — quit the analysis loop")
    print("  export    — save the results DataFrame as CSV")
    print()

    plot_agent = PlotAgent()
    plot_agent.set_dataframe(df)
    plot_index = 0

    while True:
        try:
            request = input("📊 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not request:
            continue

        if request.lower() in ("exit", "quit", "q", "done", "stop", "bye"):
            print("👋 Goodbye!")
            break

        # Detect natural-language exit intent
        exit_phrases = [
            "that's enough", "thats enough", "that is enough",
            "no more", "i'm done", "im done", "all done",
            "thank you", "thanks", "thank", "enough",
            "nothing else", "no thanks", "stop",
        ]
        if any(phrase in request.lower() for phrase in exit_phrases):
            print("👋 Thanks for using Benchwarmer.AI! Goodbye!")
            break

        if request.lower() in ("export", "export csv", "export raw"):
            path = "benchmark_results.csv"
            df.to_csv(path, index=False)
            print(f"📁 Exported to {path}")
            continue

        # Generate and execute the visualization
        print("🤖 Generating visualization…")
        result = plot_agent.generate_and_execute(
            user_request=request,
            df=df,
            output_dir="plots",
            plot_index=plot_index,
        )

        if result.get("code"):
            print(f"\n💻 Generated code:")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)

        if result["success"]:
            if result.get("output_path") and os.path.exists(result["output_path"]):
                print(f"✅ Plot saved to: {result['output_path']}")
                plot_index += 1
            elif result.get("message"):
                print(f"\n{result['message']}")
            if result.get("stdout"):
                print(result["stdout"])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            if result.get("traceback"):
                print(result["traceback"])

        print()


def main() -> None:
    """Entry point — uses orchestrator by default, legacy flow with --no-orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchwarmer.AI")
    parser.add_argument("--custom", "-c", type=str, help="Path to custom algorithm file to benchmark against.")
    parser.add_argument("--papers", "-p", nargs="+", type=str, help="PDF papers to analyze for algorithm extraction.")
    parser.add_argument("--mode", "-m", type=str, default="local", choices=["local", "modal"], help="Execution mode.")
    parser.add_argument("--intake-backend", type=str, default="claude", choices=["claude", "nemotron"], help="LLM backend for both Orchestrator and Intake (claude or nemotron).")
    parser.add_argument("--nemotron-url", type=str, default="https://integrate.api.nvidia.com/v1", help="Base URL for Nemotron (when --intake-backend=nemotron).")
    parser.add_argument("--nemotron-model", type=str, default="nvidia/nemotron-3-nano-30b-a3b", help="Model name for Nemotron (when --intake-backend=nemotron).")
    parser.add_argument("--no-orchestrator", action="store_true", help="Use legacy step-by-step flow instead of conversational mode.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    if args.no_orchestrator:
        _legacy_main()
        return

    # ── Conversational Orchestrator Mode ──
    from benchwarmer.agents.orchestrator import OrchestratorAgent

    # Validate PDF paths
    pdf_paths = None
    if args.papers:
        pdf_paths = [p for p in args.papers if os.path.exists(p)]
        if not pdf_paths:
            pdf_paths = None

    orch = OrchestratorAgent(
        execution_mode=args.mode,
        intake_backend=args.intake_backend,
        pdf_paths=pdf_paths,
        custom_algo_path=args.custom,
        nemotron_url=args.nemotron_url,
        nemotron_model=args.nemotron_model,
    )
    orch.run()


if __name__ == "__main__":
    main()
