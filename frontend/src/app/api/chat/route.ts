import { NextRequest, NextResponse } from "next/server";

const BACKEND =
  process.env.NEXT_PUBLIC_BENCHMARK_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const res = await fetch(`${BACKEND}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: body.session_id ?? null,
        message: body.message ?? "",
        execution_mode: body.execution_mode ?? "local",
        llm_backend: body.llm_backend ?? "claude",
        pdfs: body.pdfs ?? null,
        custom_algorithm_file: body.custom_algorithm_file ?? null,
      }),
    });

    // If backend returned an error (non-SSE), forward as JSON
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return NextResponse.json(
        { error: data.detail ?? res.statusText ?? "Backend error" },
        { status: res.status },
      );
    }

    // Forward the SSE stream directly to the client
    return new Response(res.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "Request failed" },
      { status: 500 },
    );
  }
}
