import { Sidebar } from "@/components/layout/Sidebar";
import { ExperimentView } from "@/components/lab/ExperimentView";

export default function Home() {
  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 flex flex-col min-h-screen relative overflow-hidden">
        <div className="flex-1 flex flex-col p-4 md:p-8 max-w-6xl mx-auto w-full">
          {/* Experiment Lab */}
          <ExperimentView />
        </div>
      </main>
    </div>
  );
}
