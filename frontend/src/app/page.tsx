import { Sidebar } from "@/components/layout/Sidebar";
import { ExperimentView } from "@/components/lab/ExperimentView";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 flex flex-col min-h-screen relative overflow-hidden">
        {/* Header / Top Bar (Optional, can be part of main content) */}

        <div className="flex-1 flex flex-col items-center justify-center p-4 md:p-8 max-w-5xl mx-auto w-full">
          <div className="text-center space-y-4 mb-8">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              Analyze Algorithms. <span className="text-muted-foreground">Scientifically.</span>
            </h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Benchwarmer researches, implements, and benchmarks algorithms in a secure sandbox.
            </p>
          </div>

          {/* Experiment Lab */}
          <div className="w-full max-w-4xl mx-auto">
            <ExperimentView />
          </div>
        </div>
      </main>
    </div>
  );
}
