"use client";

import { useState } from "react";
import type { Project } from "@/types";

interface ProjectSelectorProps {
  projects: Project[];
  selectedProject: Project | null;
  onSelectProject: (project: Project) => void;
  onCreateProject: (name: string, description?: string) => Promise<void>;
}

export default function ProjectSelector({
  projects,
  selectedProject,
  onSelectProject,
  onCreateProject,
}: ProjectSelectorProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDesc, setNewProjectDesc] = useState("");
  const [loading, setLoading] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim() || loading) return;
    setLoading(true);
    try {
      await onCreateProject(
        newProjectName.trim(),
        newProjectDesc.trim() || undefined,
      );
      setNewProjectName("");
      setNewProjectDesc("");
      setIsCreating(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <label
          htmlFor="project-select"
          className="text-xs font-medium text-zinc-500"
        >
          Project:
        </label>
        <select
          id="project-select"
          value={selectedProject?.id || ""}
          onChange={(e) => {
            const proj = projects.find((p) => p.id === e.target.value);
            if (proj) onSelectProject(proj);
          }}
          className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-900 focus:border-zinc-900 focus:outline-hidden"
        >
          {projects.length === 0 && <option value="">No projects</option>}
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
            </option>
          ))}
        </select>
      </div>

      <button
        type="button"
        onClick={() => setIsCreating((prev) => !prev)}
        className="rounded-lg border border-zinc-300 bg-white px-2.5 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 cursor-pointer"
      >
        {isCreating ? "Cancel" : "+ New Project"}
      </button>

      {/* Creation popover/form */}
      {isCreating && (
        <form
          onSubmit={handleCreate}
          className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-zinc-50 p-1.5"
        >
          <input
            type="text"
            placeholder="Project name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            required
            className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 focus:border-zinc-900 focus:outline-hidden"
          />
          <button
            type="submit"
            disabled={!newProjectName.trim() || loading}
            className="rounded bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
          >
            Create
          </button>
        </form>
      )}
    </div>
  );
}
