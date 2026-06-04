"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import MemoryInspector from "@/components/MemoryInspector";
import type { Project } from "@/types";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function ProjectMemoryPage({ params }: PageProps) {
  const resolvedParams = use(params);
  const projectId = resolvedParams.id;
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  const apiUrl = rawApiUrl.replace("localhost", "127.0.0.1");

  useEffect(() => {
    async function loadProject() {
      try {
        const res = await fetch(`${apiUrl}/api/v1/projects/${projectId}`);
        if (res.ok) {
          const data = await res.json();
          setProject(data);
        } else {
          setError(`Project ${projectId} not found.`);
        }
      } catch {
        setError("Network error: Failed to connect to project service.");
      } finally {
        setLoading(false);
      }
    }
    loadProject();
  }, [projectId, apiUrl]);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans text-zinc-900">
      {/* Top Header Bar */}
      <header className="border-b border-zinc-200 bg-white px-6 py-4 shadow-2xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-xs font-medium text-blue-600 hover:text-blue-800 transition"
            >
              ← Back to Workspace
            </Link>
            <div className="h-4 w-px bg-zinc-200" />
            <h1 className="text-xl font-bold tracking-tight text-zinc-950">
              {project
                ? `${project.name} — Memory Inspector`
                : "Memory Inspector"}
            </h1>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 p-6">
        {loading ? (
          <div className="p-8 text-center text-xs text-zinc-500">
            Loading project details…
          </div>
        ) : error ? (
          <div className="p-6 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            <p className="font-semibold mb-1">Project Not Found</p>
            <p className="text-xs">{error}</p>
            <Link
              href="/"
              className="mt-3 inline-block text-xs font-medium text-blue-600 hover:underline"
            >
              Return to Home
            </Link>
          </div>
        ) : (
          <MemoryInspector projectId={projectId} apiUrl={apiUrl} />
        )}
      </main>
    </div>
  );
}
