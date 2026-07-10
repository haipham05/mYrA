"use client";

import { useCallback, useEffect, useState } from "react";

interface TranslationPanelProps {
  apiUrl: string;
  projectId: string;
  paperId: string;
  paperStatus: string;
  onOpenSource?: (source: SourceMapSegment) => void;
}

interface TranslationSegment {
  ordinal: number;
  source_page_number: number;
  source_quote: string;
  status: string;
}

interface SourceMapSegment extends TranslationSegment {
  source_element_id: string | null;
  anchor_status: string;
  source_char_start: number | null;
  source_char_end: number | null;
  document_sha256: string;
  parser_version: string | null;
}

interface TranslationJob {
  id: string;
  status: string;
  stage?: string | null;
  completed_units?: number | null;
  total_units?: number | null;
  progress?: number | null;
  error_message?: string | null;
  warnings?: string[] | null;
  output_available?: boolean;
  segments?: TranslationSegment[];
}

interface GlossaryEntry {
  english: string;
  vietnamese: string;
}

const ACTIVE_STATUSES = new Set(["PENDING", "QUEUED", "PROCESSING", "RUNNING"]);

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function normalizeJob(value: unknown): TranslationJob | null {
  const job = asRecord(value);
  if (typeof job.id !== "string") return null;
  return {
    id: job.id,
    status: String(job.status ?? "UNKNOWN").toUpperCase(),
    stage: typeof job.stage === "string" ? job.stage : null,
    completed_units:
      typeof job.completed_units === "number" ? job.completed_units : null,
    total_units: typeof job.total_units === "number" ? job.total_units : null,
    progress: typeof job.progress === "number" ? job.progress : null,
    error_message:
      typeof job.error_message === "string"
        ? job.error_message
        : typeof job.error === "string"
          ? job.error
          : null,
    warnings: Array.isArray(job.warnings)
      ? job.warnings.filter(
          (warning): warning is string => typeof warning === "string",
        )
      : [],
    output_available: job.output_available === true,
    segments: Array.isArray(job.segments)
      ? job.segments.flatMap((segment) => {
          const item = asRecord(segment);
          return typeof item.ordinal === "number" &&
            typeof item.source_page_number === "number" &&
            typeof item.source_quote === "string" &&
            typeof item.status === "string"
            ? [
                {
                  ordinal: item.ordinal,
                  source_page_number: item.source_page_number,
                  source_quote: item.source_quote,
                  status: item.status,
                },
              ]
            : [];
        })
      : [],
  };
}

function normalizeGlossary(value: unknown): GlossaryEntry[] {
  const record = asRecord(value);
  const rawEntries = Array.isArray(value)
    ? value
    : Array.isArray(record.entries)
      ? record.entries
      : [];
  return rawEntries.flatMap((entry) => {
    const item = asRecord(entry);
    const english = item.english ?? item.source_term;
    const vietnamese = item.vietnamese ?? item.preferred_translation;
    return typeof english === "string" && typeof vietnamese === "string"
      ? [{ english, vietnamese }]
      : [];
  });
}

async function readError(
  response: Response,
  fallback: string,
): Promise<string> {
  const payload = await response.json().catch(() => ({}));
  const detail = asRecord(payload).detail;
  return typeof detail === "string" ? detail : fallback;
}

export default function TranslationPanel({
  apiUrl,
  projectId,
  paperId,
  paperStatus,
  onOpenSource,
}: TranslationPanelProps) {
  const [jobs, setJobs] = useState<TranslationJob[]>([]);
  const [glossary, setGlossary] = useState<GlossaryEntry[]>([]);
  const [acknowledged, setAcknowledged] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [savingGlossary, setSavingGlossary] = useState(false);
  const [previewJobId, setPreviewJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const baseUrl = `${apiUrl.replace(/\/$/, "")}/api/v1`;
  const paperUrl = `${baseUrl}/papers/${encodeURIComponent(paperId)}/translations?project_id=${encodeURIComponent(projectId)}`;
  const glossaryUrl = `${baseUrl}/projects/${encodeURIComponent(projectId)}/translation-glossary`;
  const projectQuery = `?project_id=${encodeURIComponent(projectId)}`;

  const loadData = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const [jobsResponse, glossaryResponse] = await Promise.all([
          fetch(paperUrl, { signal }),
          fetch(glossaryUrl, { signal }),
        ]);
        if (!jobsResponse.ok) {
          throw new Error(
            await readError(jobsResponse, "Could not load translations."),
          );
        }
        const jobsPayload = await jobsResponse.json();
        const rawJobs = Array.isArray(jobsPayload)
          ? jobsPayload
          : Array.isArray(asRecord(jobsPayload).items)
            ? (asRecord(jobsPayload).items as unknown[])
            : [];
        setJobs(
          rawJobs
            .map(normalizeJob)
            .filter((job): job is TranslationJob => job !== null),
        );

        if (!glossaryResponse.ok) {
          throw new Error(
            await readError(
              glossaryResponse,
              "Could not load the project glossary.",
            ),
          );
        }
        setGlossary(normalizeGlossary(await glossaryResponse.json()));
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError")
          return;
        setError(
          cause instanceof Error
            ? cause.message
            : "Could not load translation data.",
        );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [glossaryUrl, paperUrl],
  );

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void loadData(controller.signal);
    }, 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [loadData]);

  const refreshJob = useCallback(
    async (jobId: string) => {
      try {
        const response = await fetch(
          `${baseUrl}/translations/${encodeURIComponent(jobId)}${projectQuery}`,
        );
        if (!response.ok) return;
        const updated = normalizeJob(await response.json());
        if (updated) {
          setJobs((current) =>
            current.map((job) => (job.id === jobId ? updated : job)),
          );
        }
      } catch {
        // Keep the last known status visible; the next poll can recover.
      }
    },
    [baseUrl, projectQuery],
  );

  useEffect(() => {
    const activeJobs = jobs.filter((job) => ACTIVE_STATUSES.has(job.status));
    if (activeJobs.length === 0) return;
    const timer = window.setInterval(() => {
      activeJobs.forEach((job) => void refreshJob(job.id));
    }, 2500);
    return () => window.clearInterval(timer);
  }, [jobs, refreshJob]);

  const submitTranslation = async () => {
    if (!acknowledged || submitting || paperStatus.toUpperCase() !== "READY")
      return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(paperUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          acknowledge_external_processing: true,
          idempotency_key:
            typeof crypto !== "undefined" && "randomUUID" in crypto
              ? crypto.randomUUID()
              : `${paperId}-${Date.now()}`,
        }),
      });
      if (!response.ok)
        throw new Error(
          await readError(response, "Could not start translation."),
        );
      const job = normalizeJob(await response.json());
      if (!job)
        throw new Error(
          "The translation service returned an invalid job response.",
        );
      setJobs((current) => [
        job,
        ...current.filter((item) => item.id !== job.id),
      ]);
      setAcknowledged(false);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not start translation.",
      );
    } finally {
      setSubmitting(false);
    }
  };

  const runJobAction = async (
    job: TranslationJob,
    action: "cancel" | "retry",
  ) => {
    setError(null);
    try {
      const response = await fetch(
        `${baseUrl}/translations/${encodeURIComponent(job.id)}/${action}${projectQuery}`,
        {
          method: "POST",
        },
      );
      if (!response.ok)
        throw new Error(
          await readError(response, `Could not ${action} translation.`),
        );
      const updated = normalizeJob(await response.json());
      if (updated) {
        setJobs((current) =>
          current.map((item) => (item.id === job.id ? updated : item)),
        );
      } else {
        await refreshJob(job.id);
      }
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : `Could not ${action} translation.`,
      );
    }
  };

  const saveGlossary = async () => {
    setSavingGlossary(true);
    setError(null);
    const entries = glossary
      .map(({ english, vietnamese }) => ({
        english: english.trim(),
        vietnamese: vietnamese.trim(),
      }))
      .filter((entry) => entry.english && entry.vietnamese);
    try {
      const response = await fetch(glossaryUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries }),
      });
      if (!response.ok)
        throw new Error(
          await readError(response, "Could not save the glossary."),
        );
      const saved = await response.json().catch(() => ({ entries }));
      setGlossary(
        normalizeGlossary(saved).length > 0
          ? normalizeGlossary(saved)
          : entries,
      );
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not save the glossary.",
      );
    } finally {
      setSavingGlossary(false);
    }
  };

  const openSource = async (
    job: TranslationJob,
    segment: TranslationSegment,
  ) => {
    if (!onOpenSource) return;
    setError(null);
    try {
      const response = await fetch(
        `${baseUrl}/translations/${encodeURIComponent(job.id)}/source-map${projectQuery}`,
      );
      if (!response.ok) {
        throw new Error(
          await readError(response, "Could not load the source map."),
        );
      }
      const payload = asRecord(await response.json());
      const mappedSegments = Array.isArray(payload.segments)
        ? payload.segments
        : [];
      const mapped = mappedSegments
        .map(asRecord)
        .find((item) => item.ordinal === segment.ordinal);
      if (!mapped) throw new Error("The source page mapping is unavailable.");
      onOpenSource({
        ...segment,
        source_element_id:
          typeof mapped.source_element_id === "string"
            ? mapped.source_element_id
            : null,
        anchor_status:
          typeof mapped.anchor_status === "string"
            ? mapped.anchor_status
            : "page_only",
        source_char_start:
          typeof mapped.source_char_start === "number"
            ? mapped.source_char_start
            : null,
        source_char_end:
          typeof mapped.source_char_end === "number"
            ? mapped.source_char_end
            : null,
        document_sha256:
          typeof mapped.document_sha256 === "string"
            ? mapped.document_sha256
            : "",
        parser_version:
          typeof mapped.parser_version === "string"
            ? mapped.parser_version
            : null,
      });
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not open source page.",
      );
    }
  };

  const paperReady = paperStatus.toUpperCase() === "READY";

  return (
    <section
      className="space-y-5 rounded-xl border border-zinc-200 bg-white p-5 text-zinc-900 shadow-xs"
      aria-labelledby="translation-heading"
    >
      <header>
        <h2 id="translation-heading" className="text-base font-semibold">
          Translate this paper
        </h2>
        <p className="mt-1 text-sm text-zinc-600">
          English → Vietnamese · translated-only PDF
        </p>
      </header>

      <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
        <p className="font-medium">External processing disclosure</p>
        <p className="mt-1">
          Selected paper text will be sent through the PDFMathTranslate service
          to SiliconFlowFree for translation. This is not local-only processing.
          The free service and its availability are operated by third parties.
        </p>
        <label className="mt-3 flex items-start gap-2">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(event) => setAcknowledged(event.target.checked)}
            aria-label="I understand that paper text is processed by an external translation service"
            className="mt-0.5 size-4 accent-amber-700"
          />
          <span>
            I understand and agree to send this paper’s text to the external
            translation service.
          </span>
        </label>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void submitTranslation()}
          disabled={!paperReady || !acknowledged || submitting}
          className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {submitting ? "Starting…" : "Translate to Vietnamese"}
        </button>
        {!paperReady && (
          <p className="text-sm text-zinc-600">
            This paper must finish indexing before it can be translated.
          </p>
        )}
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-md bg-rose-50 p-3 text-sm text-rose-800"
        >
          {error}
        </p>
      )}

      <div aria-live="polite" aria-busy={loading}>
        <h3 className="text-sm font-semibold">Translation versions</h3>
        {loading ? (
          <p className="mt-2 text-sm text-zinc-500">Loading translations…</p>
        ) : jobs.length === 0 ? (
          <p className="mt-2 text-sm text-zinc-500">No translations yet.</p>
        ) : (
          <ul className="mt-2 space-y-3">
            {jobs.map((job) => {
              const active = ACTIVE_STATUSES.has(job.status);
              const progressText =
                job.completed_units !== null &&
                job.completed_units !== undefined &&
                job.total_units !== null &&
                job.total_units !== undefined
                  ? `${job.completed_units} of ${job.total_units} sections`
                  : typeof job.progress === "number"
                    ? `${Math.round(Math.max(0, Math.min(1, job.progress)) * 100)}%`
                    : null;
              return (
                <li
                  key={job.id}
                  className="rounded-lg border border-zinc-200 p-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <p className="font-medium">English → Vietnamese</p>
                      <p className="text-sm text-zinc-600">
                        Status: {job.status.toLowerCase().replaceAll("_", " ")}
                        {job.stage
                          ? ` · ${job.stage.replaceAll("_", " ").toLowerCase()}`
                          : ""}
                      </p>
                      {progressText && (
                        <p className="text-sm text-zinc-600">
                          Progress: {progressText}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2">
                      {job.status === "COMPLETED" && (
                        <>
                          <button
                            type="button"
                            onClick={() =>
                              setPreviewJobId((current) =>
                                current === job.id ? null : job.id,
                              )
                            }
                            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
                          >
                            {previewJobId === job.id
                              ? "Hide preview"
                              : "Preview PDF"}
                          </button>
                          <a
                            href={`${baseUrl}/translations/${encodeURIComponent(job.id)}/pdf${projectQuery}`}
                            className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
                          >
                            Download PDF
                          </a>
                        </>
                      )}
                      {active && (
                        <button
                          type="button"
                          onClick={() => void runJobAction(job, "cancel")}
                          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
                        >
                          Cancel
                        </button>
                      )}
                      {job.status === "FAILED" && (
                        <button
                          type="button"
                          onClick={() => void runJobAction(job, "retry")}
                          className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
                        >
                          Retry
                        </button>
                      )}
                    </div>
                  </div>
                  {job.error_message && (
                    <p className="mt-2 text-sm text-rose-700">
                      {job.error_message}
                    </p>
                  )}
                  {job.warnings?.map((warning, index) => (
                    <p
                      key={`${job.id}-warning-${index}`}
                      className="mt-2 text-sm text-amber-800"
                    >
                      {warning}
                    </p>
                  ))}
                  {job.status === "COMPLETED" && job.segments?.length ? (
                    <ul className="mt-3 space-y-2 border-t border-zinc-200 pt-3">
                      {job.segments.map((segment) => (
                        <li
                          key={`${job.id}-${segment.ordinal}`}
                          className="flex flex-wrap items-start justify-between gap-2 text-sm"
                        >
                          <p className="min-w-0 flex-1 text-zinc-700">
                            Page {segment.source_page_number}: “
                            {segment.source_quote}”
                          </p>
                          <button
                            type="button"
                            onClick={() => void openSource(job, segment)}
                            className="rounded-md border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-50"
                          >
                            Open original page
                          </button>
                        </li>
                      ))}
                      <li className="text-xs text-zinc-500">
                        Exact text highlighting is shown only when the original
                        quote can be matched safely.
                      </li>
                    </ul>
                  ) : null}
                  {previewJobId === job.id && job.status === "COMPLETED" ? (
                    <iframe
                      title="Translated Vietnamese PDF preview"
                      src={`${baseUrl}/translations/${encodeURIComponent(job.id)}/pdf${projectQuery}&inline=true`}
                      loading="lazy"
                      referrerPolicy="no-referrer"
                      className="mt-3 h-[640px] w-full rounded-lg border border-zinc-300 bg-zinc-100"
                    />
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <section
        className="border-t border-zinc-200 pt-4"
        aria-labelledby="glossary-heading"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 id="glossary-heading" className="text-sm font-semibold">
              Project glossary
            </h3>
            <p className="text-sm text-zinc-600">
              Preferred Vietnamese terms appear consistently; the English term
              is retained on first use.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() =>
                setGlossary((current) => [
                  ...current,
                  { english: "", vietnamese: "" },
                ])
              }
              className="rounded-md border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-50"
            >
              Add term
            </button>
            <button
              type="button"
              onClick={() => void saveGlossary()}
              disabled={savingGlossary}
              className="rounded-md bg-zinc-800 px-3 py-1.5 text-sm font-medium text-white hover:bg-zinc-700 disabled:opacity-50"
            >
              {savingGlossary ? "Saving…" : "Save glossary"}
            </button>
          </div>
        </div>
        {glossary.length === 0 ? (
          <p className="mt-3 text-sm text-zinc-500">No glossary terms yet.</p>
        ) : (
          <ul className="mt-3 space-y-2">
            {glossary.map((entry, index) => (
              <li
                key={`glossary-${index}`}
                className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]"
              >
                <label className="text-xs font-medium text-zinc-600">
                  English term
                  <input
                    value={entry.english}
                    onChange={(event) =>
                      setGlossary((current) =>
                        current.map((item, itemIndex) =>
                          itemIndex === index
                            ? { ...item, english: event.target.value }
                            : item,
                        ),
                      )
                    }
                    className="mt-1 w-full rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-900"
                  />
                </label>
                <label className="text-xs font-medium text-zinc-600">
                  Preferred Vietnamese
                  <input
                    value={entry.vietnamese}
                    onChange={(event) =>
                      setGlossary((current) =>
                        current.map((item, itemIndex) =>
                          itemIndex === index
                            ? { ...item, vietnamese: event.target.value }
                            : item,
                        ),
                      )
                    }
                    className="mt-1 w-full rounded-md border border-zinc-300 px-3 py-2 text-sm text-zinc-900"
                  />
                </label>
                <button
                  type="button"
                  aria-label={`Remove glossary term ${entry.english || index + 1}`}
                  onClick={() =>
                    setGlossary((current) =>
                      current.filter((_, itemIndex) => itemIndex !== index),
                    )
                  }
                  className="self-end rounded-md border border-zinc-300 px-3 py-2 text-sm hover:bg-zinc-50"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}
