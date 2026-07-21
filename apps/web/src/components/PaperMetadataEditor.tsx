"use client";

import { useState } from "react";
import type { Paper } from "@/types";

interface PaperMetadataEditorProps {
  paper: Paper;
  apiUrl: string;
  onUpdated: (paper: Paper) => void;
}

type EditableMetadata = Pick<
  Paper,
  | "title"
  | "authors"
  | "publication_year"
  | "doi"
  | "arxiv_id"
  | "abstract"
  | "source_url"
>;

const fields: { key: keyof EditableMetadata; label: string }[] = [
  { key: "title", label: "Title" },
  { key: "authors", label: "Authors" },
  { key: "publication_year", label: "Year" },
  { key: "doi", label: "DOI" },
  { key: "arxiv_id", label: "arXiv ID" },
  { key: "source_url", label: "Source URL" },
  { key: "abstract", label: "Abstract" },
];

function authorsText(authors: string[] | null | undefined) {
  return authors?.join(", ") ?? "";
}

function toDraft(paper: Paper): Record<keyof EditableMetadata, string> {
  return {
    title: paper.title ?? "",
    authors: authorsText(paper.authors),
    publication_year: paper.publication_year?.toString() ?? "",
    doi: paper.doi ?? "",
    arxiv_id: paper.arxiv_id ?? "",
    source_url: paper.source_url ?? "",
    abstract: paper.abstract ?? "",
  };
}

function valueText(key: keyof EditableMetadata, paper: Paper) {
  if (key === "authors") return authorsText(paper.authors);
  return String(paper[key] ?? "");
}

export default function PaperMetadataEditor({
  paper,
  apiUrl,
  onUpdated,
}: PaperMetadataEditorProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(() => toDraft(paper));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function saveMetadata(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving) return;
    setSaving(true);
    setError(null);
    const payload: Record<string, string | number | string[] | null> = {};
    for (const field of fields) {
      const value = draft[field.key].trim();
      if (field.key === "authors") {
        payload.authors = value
          ? value
              .split(",")
              .map((author) => author.trim())
              .filter(Boolean)
          : null;
      } else if (field.key === "publication_year") {
        payload.publication_year = value ? Number(value) : null;
      } else {
        payload[field.key] = value || null;
      }
    }

    try {
      const response = await fetch(
        `${apiUrl}/api/v1/papers/${paper.id}?project_id=${paper.project_id}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Could not save paper details.");
      }
      const updated: Paper = await response.json();
      onUpdated(updated);
      setDraft(toDraft(updated));
      setEditing(false);
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Could not save paper details.",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <details className="mt-3 rounded-lg border border-zinc-200 bg-white text-sm">
      <summary className="cursor-pointer px-4 py-3 font-medium text-zinc-800">
        Paper details
        <span className="ml-2 font-normal text-zinc-500">{paper.filename}</span>
      </summary>
      <div className="border-t border-zinc-100 p-4">
        {editing ? (
          <form onSubmit={saveMetadata} className="space-y-3">
            {fields.map(({ key, label }) => (
              <label
                key={key}
                className="grid gap-1 text-xs font-medium text-zinc-700 sm:grid-cols-[8rem_1fr] sm:items-start"
              >
                <span className="pt-2">{label}</span>
                {key === "abstract" ? (
                  <textarea
                    value={draft[key]}
                    onChange={(event) =>
                      setDraft((current) => ({
                        ...current,
                        [key]: event.target.value,
                      }))
                    }
                    rows={4}
                    maxLength={50000}
                    className="rounded border border-zinc-300 px-2 py-1.5 text-sm font-normal"
                  />
                ) : (
                  <input
                    type={key === "publication_year" ? "number" : "text"}
                    min={key === "publication_year" ? 1000 : undefined}
                    max={key === "publication_year" ? 2100 : undefined}
                    maxLength={key === "title" ? 2000 : 4000}
                    value={draft[key]}
                    onChange={(event) =>
                      setDraft((current) => ({
                        ...current,
                        [key]: event.target.value,
                      }))
                    }
                    className="rounded border border-zinc-300 px-2 py-1.5 text-sm font-normal"
                  />
                )}
              </label>
            ))}
            {error && (
              <p role="alert" className="text-xs text-red-700">
                {error}
              </p>
            )}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={saving}
                className="rounded bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save details"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setDraft(toDraft(paper));
                  setEditing(false);
                  setError(null);
                }}
                className="rounded border border-zinc-300 px-3 py-1.5 text-xs"
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <>
            <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-[8rem_1fr]">
              {fields.map(({ key, label }) => {
                const value = valueText(key, paper);
                return (
                  <div key={key} className="contents">
                    <dt className="text-xs font-medium text-zinc-500">
                      {label}
                    </dt>
                    <dd className="break-words text-zinc-800">
                      {value || (
                        <span className="text-zinc-400">Not available</span>
                      )}
                      {paper.metadata_provenance?.[key] && (
                        <span className="ml-2 text-[10px] text-zinc-400">
                          ({paper.metadata_provenance[key]})
                        </span>
                      )}
                    </dd>
                  </div>
                );
              })}
            </dl>
            <button
              type="button"
              onClick={() => setEditing(true)}
              className="mt-3 rounded border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50"
            >
              Edit details
            </button>
          </>
        )}
      </div>
    </details>
  );
}
