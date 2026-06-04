"use client";

import { useEffect, useState, useCallback } from "react";
import type { Memory, MemorySource, MemoryType, MemoryStatus } from "@/types";

interface MemoryInspectorProps {
  projectId: string | null;
  apiUrl: string;
  onSelectSource?: (source: MemorySource) => void;
  className?: string;
}

export default function MemoryInspector({
  projectId,
  apiUrl,
  className = "",
}: MemoryInspectorProps) {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [typeFilter, setTypeFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  // Modals & form state
  const [isCreating, setIsCreating] = useState(false);
  const [editingMemory, setEditingMemory] = useState<Memory | null>(null);
  const [supersedingMemory, setSupersedingMemory] = useState<Memory | null>(
    null,
  );
  const [expandedHistoryId, setExpandedHistoryId] = useState<string | null>(
    null,
  );

  // Form fields
  const [formType, setFormType] = useState<MemoryType>("DECISION");
  const [formTitle, setFormTitle] = useState("");
  const [formContent, setFormContent] = useState("");
  const [formImportance, setFormImportance] = useState(0.8);
  const [formReason, setFormReason] = useState("");
  const [formConflictError, setFormConflictError] = useState<string | null>(
    null,
  );
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const fetchMemories = useCallback(() => {
    setRefreshTrigger((prev) => prev + 1);
  }, []);

  useEffect(() => {
    let ignore = false;
    async function load() {
      if (!projectId) {
        if (!ignore) {
          setMemories([]);
          setTotal(0);
        }
        return;
      }
      try {
        const params = new URLSearchParams();
        if (typeFilter !== "ALL") params.append("memory_type", typeFilter);
        if (statusFilter !== "ALL") params.append("status", statusFilter);
        if (searchQuery.trim()) params.append("search", searchQuery.trim());
        params.append("limit", "50");

        const res = await fetch(
          `${apiUrl}/api/v1/projects/${projectId}/memories?${params.toString()}`,
        );
        if (!ignore && res.ok) {
          const data = await res.json();
          setMemories(data.items || []);
          setTotal(data.total || 0);
        } else if (!ignore) {
          const err = await res.json().catch(() => ({}));
          setError(err.detail || "Failed to load project memories.");
        }
      } catch {
        if (!ignore) {
          setError("Network error: Failed to connect to memory service.");
        }
      } finally {
        if (!ignore) {
          setIsLoading(false);
        }
      }
    }
    load();
    return () => {
      ignore = true;
    };
  }, [
    projectId,
    apiUrl,
    typeFilter,
    statusFilter,
    searchQuery,
    refreshTrigger,
  ]);

  const handleTogglePin = async (mem: Memory) => {
    if (!projectId) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories/${mem.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            is_pinned: !mem.is_pinned,
            version: mem.version,
            reason: mem.is_pinned ? "Unpinned by user" : "Pinned by user",
          }),
        },
      );
      if (res.ok) {
        fetchMemories();
      } else if (res.status === 409) {
        setError("Concurrent edit conflict while toggling pin. Refreshed.");
        fetchMemories();
      }
    } catch {
      setError("Network error while updating pin status.");
    }
  };

  const handleArchive = async (mem: Memory) => {
    if (!projectId) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories/${mem.id}?hard_delete=false&reason=Archived+or+marked+incorrect`,
        { method: "DELETE" },
      );
      if (res.ok) {
        fetchMemories();
      }
    } catch {
      setError("Network error while archiving memory.");
    }
  };

  const handleDelete = async (mem: Memory) => {
    if (!projectId) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories/${mem.id}?hard_delete=true`,
        { method: "DELETE" },
      );
      if (res.ok) {
        fetchMemories();
      }
    } catch {
      setError("Network error while deleting memory.");
    }
  };

  const openCreateModal = () => {
    setFormType("DECISION");
    setFormTitle("");
    setFormContent("");
    setFormImportance(0.8);
    setFormReason("");
    setFormConflictError(null);
    setIsCreating(true);
  };

  const openEditModal = (mem: Memory) => {
    setEditingMemory(mem);
    setFormType(mem.memory_type);
    setFormTitle(mem.title);
    setFormContent(mem.content);
    setFormImportance(mem.importance);
    setFormReason("");
    setFormConflictError(null);
  };

  const openSupersedeModal = (mem: Memory) => {
    setSupersedingMemory(mem);
    setFormType(mem.memory_type);
    setFormTitle(`Updated: ${mem.title}`);
    setFormContent(mem.content);
    setFormImportance(mem.importance);
    setFormReason(`Supersedes decision v${mem.version}`);
    setFormConflictError(null);
  };

  const handleSaveCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!projectId || !formTitle.trim() || !formContent.trim()) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            memory_type: formType,
            title: formTitle.trim(),
            content: formContent.trim(),
            importance: formImportance,
            confidence: 1.0,
            is_pinned: false,
            sources: [],
          }),
        },
      );
      if (res.ok) {
        setIsCreating(false);
        fetchMemories();
      } else {
        const data = await res.json().catch(() => ({}));
        setFormConflictError(data.detail || "Failed to create memory.");
      }
    } catch {
      setFormConflictError("Network error while creating memory.");
    }
  };

  const handleSaveEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (
      !projectId ||
      !editingMemory ||
      !formTitle.trim() ||
      !formContent.trim()
    )
      return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories/${editingMemory.id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: formTitle.trim(),
            content: formContent.trim(),
            importance: formImportance,
            version: editingMemory.version,
            reason: formReason.trim() || "User manual edit",
          }),
        },
      );
      if (res.ok) {
        setEditingMemory(null);
        fetchMemories();
      } else if (res.status === 409) {
        setFormConflictError(
          "Stale edit warning: this memory was modified concurrently. Please reload and re-apply.",
        );
      } else {
        const data = await res.json().catch(() => ({}));
        setFormConflictError(data.detail || "Failed to save edit.");
      }
    } catch {
      setFormConflictError("Network error while updating memory.");
    }
  };

  const handleSaveSupersede = async (e: React.FormEvent) => {
    e.preventDefault();
    if (
      !projectId ||
      !supersedingMemory ||
      !formTitle.trim() ||
      !formContent.trim()
    )
      return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${projectId}/memories/${supersedingMemory.id}/supersede`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            memory_type: formType,
            title: formTitle.trim(),
            content: formContent.trim(),
            importance: formImportance,
            confidence: 1.0,
            is_pinned: supersedingMemory.is_pinned,
            sources: [],
          }),
        },
      );
      if (res.ok) {
        setSupersedingMemory(null);
        fetchMemories();
      } else {
        const data = await res.json().catch(() => ({}));
        setFormConflictError(data.detail || "Failed to supersede memory.");
      }
    } catch {
      setFormConflictError("Network error while superseding memory.");
    }
  };

  const getTypeBadgeClass = (type: MemoryType) => {
    switch (type) {
      case "DECISION":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "PREFERENCE":
        return "bg-purple-100 text-purple-800 border-purple-200";
      case "TERMINOLOGY":
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
      case "PAPER_FACT":
        return "bg-amber-100 text-amber-800 border-amber-200";
      default:
        return "bg-zinc-100 text-zinc-800 border-zinc-200";
    }
  };

  const getStatusBadgeClass = (status: MemoryStatus) => {
    switch (status) {
      case "ACTIVE":
        return "bg-green-100 text-green-800 border-green-200";
      case "SUPERSEDED":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "ARCHIVED":
        return "bg-zinc-200 text-zinc-700 border-zinc-300";
      case "DISPUTED":
        return "bg-red-100 text-red-800 border-red-200";
      default:
        return "bg-zinc-100 text-zinc-800 border-zinc-200";
    }
  };

  return (
    <div
      data-testid="memory-inspector"
      className={`flex flex-col bg-white border border-zinc-200 rounded-lg shadow-2xs overflow-hidden ${className}`}
    >
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 border-b border-zinc-200 bg-zinc-50">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold text-zinc-900">
              Project Research Memory
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-200 text-zinc-700 font-mono">
              {total} {total === 1 ? "entry" : "entries"}
            </span>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            Durable repository of project decisions, conventions, and user
            preferences.
          </p>
        </div>

        <button
          type="button"
          onClick={openCreateModal}
          disabled={!projectId}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition"
        >
          <span>+ Add Decision / Preference</span>
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center gap-3 p-3 border-b border-zinc-100 bg-white">
        {/* Search */}
        <div className="flex-1 min-w-[200px]">
          <input
            type="text"
            placeholder="Search memories..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md focus:outline-hidden focus:ring-1 focus:ring-blue-500"
          />
        </div>

        {/* Type Filter */}
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="text-xs px-2.5 py-1.5 border border-zinc-300 rounded-md bg-white text-zinc-700 focus:outline-hidden focus:ring-1 focus:ring-blue-500"
        >
          <option value="ALL">All Types</option>
          <option value="DECISION">Decisions</option>
          <option value="PREFERENCE">Preferences</option>
          <option value="TERMINOLOGY">Terminology</option>
          <option value="PAPER_FACT">Paper Facts</option>
          <option value="PROCEDURAL">Procedural</option>
        </select>

        {/* Status Filter */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="text-xs px-2.5 py-1.5 border border-zinc-300 rounded-md bg-white text-zinc-700 focus:outline-hidden focus:ring-1 focus:ring-blue-500"
        >
          <option value="ALL">All Statuses</option>
          <option value="ACTIVE">Active Only</option>
          <option value="SUPERSEDED">Superseded</option>
          <option value="ARCHIVED">Archived</option>
        </select>
      </div>

      {/* Error alert */}
      {error && (
        <div className="p-3 bg-red-50 border-b border-red-200 text-xs text-red-700 flex justify-between items-center">
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700 font-bold ml-2"
          >
            ×
          </button>
        </div>
      )}

      {/* Memory List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 max-h-[500px]">
        {isLoading && memories.length === 0 ? (
          <div className="p-8 text-center text-xs text-zinc-500">
            Loading memories…
          </div>
        ) : memories.length === 0 ? (
          <div className="p-8 text-center text-xs text-zinc-500 border border-dashed border-zinc-200 rounded-lg">
            No memories found matching the selected filters.
          </div>
        ) : (
          memories.map((mem) => {
            const isExpanded = expandedHistoryId === mem.id;
            return (
              <div
                key={mem.id}
                data-testid={`memory-card-${mem.id}`}
                className={`p-3.5 border rounded-lg transition ${
                  mem.status === "ACTIVE"
                    ? "bg-white border-zinc-200 shadow-2xs hover:border-zinc-300"
                    : "bg-zinc-50 border-zinc-200 opacity-80"
                }`}
              >
                {/* Header row */}
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Pin button */}
                    <button
                      type="button"
                      onClick={() => handleTogglePin(mem)}
                      title={mem.is_pinned ? "Unpin memory" : "Pin memory"}
                      className={`text-sm transition ${
                        mem.is_pinned
                          ? "text-amber-500 hover:text-amber-600"
                          : "text-zinc-300 hover:text-zinc-500"
                      }`}
                    >
                      ★
                    </button>

                    <h3 className="text-sm font-medium text-zinc-900">
                      {mem.title}
                    </h3>

                    {/* Type Badge */}
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${getTypeBadgeClass(
                        mem.memory_type,
                      )}`}
                    >
                      {mem.memory_type}
                    </span>

                    {/* Status Badge */}
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${getStatusBadgeClass(
                        mem.status,
                      )}`}
                    >
                      {mem.status}
                    </span>

                    {/* Version */}
                    <span className="text-[10px] px-1.5 py-0.5 rounded-sm bg-zinc-100 text-zinc-600 font-mono">
                      v{mem.version}
                    </span>
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center gap-1.5">
                    {mem.status === "ACTIVE" && (
                      <button
                        type="button"
                        onClick={() => openSupersedeModal(mem)}
                        className="text-xs px-2 py-1 rounded-sm bg-blue-50 text-blue-700 hover:bg-blue-100 transition"
                      >
                        Supersede
                      </button>
                    )}

                    <button
                      type="button"
                      onClick={() => openEditModal(mem)}
                      className="text-xs px-2 py-1 rounded-sm bg-zinc-100 text-zinc-700 hover:bg-zinc-200 transition"
                    >
                      Edit
                    </button>

                    {mem.status === "ACTIVE" ? (
                      <button
                        type="button"
                        onClick={() => handleArchive(mem)}
                        className="text-xs px-2 py-1 rounded-sm bg-orange-50 text-orange-700 hover:bg-orange-100 transition"
                      >
                        Archive
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => handleDelete(mem)}
                        className="text-xs px-2 py-1 rounded-sm bg-red-50 text-red-700 hover:bg-red-100 transition"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                {/* Content */}
                <p className="text-xs text-zinc-700 whitespace-pre-wrap leading-relaxed mb-2.5">
                  {mem.content}
                </p>

                {/* Superseded indicator */}
                {mem.superseded_by_id && (
                  <div className="text-[11px] p-1.5 rounded-sm bg-orange-50 text-orange-800 mb-2 border border-orange-200">
                    Superseded by memory ID:{" "}
                    <code className="font-mono">{mem.superseded_by_id}</code>
                  </div>
                )}

                {/* Provenance and Sources */}
                {mem.sources && mem.sources.length > 0 && (
                  <div className="text-[11px] text-zinc-500 mb-2 space-y-1">
                    <span className="font-medium text-zinc-600">
                      Provenance:
                    </span>
                    {mem.sources.map((s) => (
                      <div
                        key={s.id}
                        className="flex items-center gap-1 pl-2 text-zinc-600"
                      >
                        <span className="font-semibold text-zinc-700">
                          {s.source_type === "MESSAGE"
                            ? "Conversation Message"
                            : "Paper Quote"}
                          :
                        </span>
                        {s.quote_text && (
                          <span className="italic">
                            &quot;{s.quote_text.slice(0, 80)}
                            {s.quote_text.length > 80 ? "…" : ""}&quot;
                          </span>
                        )}
                        {s.page_number && (
                          <span className="font-mono text-zinc-500">
                            (Page {s.page_number})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Revision history toggle */}
                {mem.history && mem.history.length > 0 && (
                  <div className="pt-2 border-t border-zinc-100">
                    <button
                      type="button"
                      onClick={() =>
                        setExpandedHistoryId(isExpanded ? null : mem.id)
                      }
                      className="text-[11px] text-zinc-500 hover:text-zinc-800 font-medium flex items-center gap-1"
                    >
                      <span>
                        {isExpanded ? "▼ Hide" : "▶ Show"} Audit History
                      </span>
                      <span className="text-zinc-400">
                        ({mem.history.length})
                      </span>
                    </button>

                    {isExpanded && (
                      <div className="mt-2 pl-3 border-l-2 border-zinc-200 space-y-1.5 text-[11px]">
                        {mem.history.map((h) => (
                          <div key={h.id} className="text-zinc-600">
                            <span className="font-semibold text-zinc-800">
                              {h.action}
                            </span>
                            <span className="text-zinc-400 ml-1">
                              ({new Date(h.created_at).toLocaleString()})
                            </span>
                            {h.reason && (
                              <span className="block text-zinc-500 italic pl-2">
                                Reason: {h.reason}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Create Modal */}
      {isCreating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <form
            onSubmit={handleSaveCreate}
            className="w-full max-w-md bg-white rounded-lg shadow-lg border border-zinc-200 p-5 space-y-4"
          >
            <div className="flex justify-between items-center border-b border-zinc-100 pb-3">
              <h3 className="text-sm font-semibold text-zinc-900">
                Add Project Memory
              </h3>
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="text-zinc-400 hover:text-zinc-600 text-lg"
              >
                ×
              </button>
            </div>

            {formConflictError && (
              <div className="p-2.5 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                {formConflictError}
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Memory Type
              </label>
              <select
                value={formType}
                onChange={(e) => setFormType(e.target.value as MemoryType)}
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              >
                <option value="DECISION">Project Decision</option>
                <option value="PREFERENCE">User Preference</option>
                <option value="TERMINOLOGY">Terminology / Definition</option>
                <option value="PROCEDURAL">Procedural Convention</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Title
              </label>
              <input
                type="text"
                required
                value={formTitle}
                onChange={(e) => setFormTitle(e.target.value)}
                placeholder="e.g. Choose AURC over ECE"
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Content
              </label>
              <textarea
                required
                rows={4}
                value={formContent}
                onChange={(e) => setFormContent(e.target.value)}
                placeholder="Detailed rationale or specification..."
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setIsCreating(false)}
                className="px-3 py-1.5 text-xs rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1.5 text-xs rounded-md bg-blue-600 text-white hover:bg-blue-700 font-medium"
              >
                Create Memory
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Edit Modal */}
      {editingMemory && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <form
            onSubmit={handleSaveEdit}
            className="w-full max-w-md bg-white rounded-lg shadow-lg border border-zinc-200 p-5 space-y-4"
          >
            <div className="flex justify-between items-center border-b border-zinc-100 pb-3">
              <h3 className="text-sm font-semibold text-zinc-900">
                Edit Memory (v{editingMemory.version})
              </h3>
              <button
                type="button"
                onClick={() => setEditingMemory(null)}
                className="text-zinc-400 hover:text-zinc-600 text-lg"
              >
                ×
              </button>
            </div>

            {formConflictError && (
              <div className="p-2.5 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                {formConflictError}
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Title
              </label>
              <input
                type="text"
                required
                value={formTitle}
                onChange={(e) => setFormTitle(e.target.value)}
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Content
              </label>
              <textarea
                required
                rows={4}
                value={formContent}
                onChange={(e) => setFormContent(e.target.value)}
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                Edit Reason / Note (Audit Log)
              </label>
              <input
                type="text"
                value={formReason}
                onChange={(e) => setFormReason(e.target.value)}
                placeholder="Why is this change being made?"
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setEditingMemory(null)}
                className="px-3 py-1.5 text-xs rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1.5 text-xs rounded-md bg-blue-600 text-white hover:bg-blue-700 font-medium"
              >
                Save Changes
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Supersede Modal */}
      {supersedingMemory && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <form
            onSubmit={handleSaveSupersede}
            className="w-full max-w-md bg-white rounded-lg shadow-lg border border-zinc-200 p-5 space-y-4"
          >
            <div className="flex justify-between items-center border-b border-zinc-100 pb-3">
              <div>
                <h3 className="text-sm font-semibold text-zinc-900">
                  Supersede Decision
                </h3>
                <p className="text-xs text-zinc-500">
                  The current decision will become SUPERSEDED and link to this
                  new version.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSupersedingMemory(null)}
                className="text-zinc-400 hover:text-zinc-600 text-lg"
              >
                ×
              </button>
            </div>

            {formConflictError && (
              <div className="p-2.5 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                {formConflictError}
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                New Title
              </label>
              <input
                type="text"
                required
                value={formTitle}
                onChange={(e) => setFormTitle(e.target.value)}
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-700 mb-1">
                New Content / Rationale
              </label>
              <textarea
                required
                rows={4}
                value={formContent}
                onChange={(e) => setFormContent(e.target.value)}
                className="w-full text-xs px-3 py-1.5 border border-zinc-300 rounded-md"
              />
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setSupersedingMemory(null)}
                className="px-3 py-1.5 text-xs rounded-md border border-zinc-300 text-zinc-700 hover:bg-zinc-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1.5 text-xs rounded-md bg-orange-600 text-white hover:bg-orange-700 font-medium"
              >
                Supersede Decision
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
