# Benchwarmer.AI

Algorithm benchmarking platform with AI-powered code generation and natural language interaction.

## Features

- **Natural Language Problem Descriptions** - Describe your optimization problem in plain English
- **PDF Paper Analysis** - Upload algorithm papers and automatically generate implementations
- **Automated Benchmarking** - Run comprehensive benchmarks with multiple graph generators
- **Interactive Visualizations** - View results with interactive charts
- **Secure Execution** - All algorithms run in isolated sandboxes

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Anthropic API key

### Setup

1. **Install Backend Dependencies**
```bash
cd agent-backend
pip install -r requirements.txt
```

2. **Install Frontend Dependencies**
```bash
cd frontend
npm install
```

3. **Configure Environment**
Create `.env` in project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### Run

**Terminal 1 - Backend:**
```bash
./start_backend.sh
# or manually: cd agent-backend && python server.py
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
# or manually: cd frontend && npm run dev
```

Open **http://localhost:3000**

## Usage

### Basic Benchmark
1. Enter a problem description: `"Compare vertex cover algorithms on sparse graphs"`
2. Click "Run Experiment"
3. View results

### With PDF Papers (NEW!)
1. Describe your problem: `"Benchmark maximum cut algorithms"`
2. Click "Add PDF" and upload algorithm paper(s)
3. (Optional) Add implementation instructions
4. Click "Run Experiment"
5. The system will:
   - Analyze the PDF
   - Generate algorithm code
   - Run benchmarks
   - Display comparative results

## Architecture

```
Benchwarmer.AI/
├── agent-backend/          # FastAPI backend
│   ├── server.py           # API endpoints
│   └── benchwarmer/        # Core library
│       ├── agents/         # AI agents (Intake, Implementation, Plot)
│       ├── engine/         # Benchmark execution
│       ├── generators/     # Graph instance generators
│       └── utils/          # Benchmark suites, sandbox
├── frontend/               # Next.js frontend
│   └── src/
│       └── components/lab/ # Experiment UI
└── docs/
    ├── SPEC.md             # Original specification
    └── revised-architecture.md  # Enhanced architecture
```

## API

**POST** `/api/benchmark`
- Content-Type: `multipart/form-data`
- Parameters:
  - `query` (string, required) - Problem description
  - `algorithm_description` (string, optional) - Implementation instructions
  - `pdfs` (File[], optional) - Algorithm papers

## Benchmark Suites

Built-in support for established benchmark libraries:
- **Biq Mac Library** - Max-Cut instances
- **DIMACS** - Classic graph optimization problems
- **SNAP** - Real-world network datasets

## Agents

### Intake Agent
Converts natural language problem descriptions into structured benchmark configurations.

### Implementation Agent
Generates algorithm code from:
- Natural language descriptions
- PDF research papers
- Existing code samples

### Plot Agent (Coming Soon)
Generates custom visualizations based on user requests.

## Contributing

See individual component READMEs for development details.

## License

[Your License Here]
