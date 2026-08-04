"use client";

import { useState } from "react";
import type { AssistantApprovalResponse } from "@/types";

interface DiscoveryImportControlsProps {
  runId: string | null;
  candidate: Record<string, unknown>;
  pdfUrl?: string;
  openAccess?: boolean;
  disabled: boolean;
  onPropose: (
    runId: string,
    candidate: Record<string, unknown>,
  ) => Promise<AssistantApprovalResponse>;
  onDecide: (
    actionId: string,
    approve: boolean,
  ) => Promise<AssistantApprovalResponse>;
}

export default function DiscoveryImportControls({
  runId,
  candidate,
  pdfUrl,
  openAccess,
  disabled,
  onPropose,
  onDecide,
}: DiscoveryImportControlsProps) {
  const [proposal, setProposal] = useState<AssistantApprovalResponse | null>(
    null,
  );
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  if (!runId || !pdfUrl || openAccess !== true) return null;

  const requestProposal = async () => {
    setBusy(true);
    try {
      setProposal(await onPropose(runId, candidate));
      setMessage("Review the PDF source before approving its download.");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Could not prepare this import.",
      );
    } finally {
      setBusy(false);
    }
  };

  const decide = async (approve: boolean) => {
    if (!proposal) return;
    setBusy(true);
    try {
      const updated = await onDecide(proposal.id, approve);
      setProposal(updated);
      const importStatus = updated.import_result?.status;
      setMessage(
        approve
          ? importStatus === "PROCESSING"
            ? "Approved; paper queued for ingestion."
            : importStatus === "READY"
              ? "This paper is already available in your library."
              : importStatus
                ? "Approved; paper import was created."
                : updated.status === "APPROVED"
                  ? "The import was approved."
                  : "The import was approved."
          : "Import rejected; no paper was downloaded.",
      );
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "The import could not be completed.",
      );
    } finally {
      setBusy(false);
    }
  };

  if (!proposal) {
    return (
      <div className="mt-2">
        <button
          type="button"
          disabled={disabled || busy}
          onClick={() => void requestProposal()}
          className="rounded border border-zinc-300 px-2 py-1 text-xs hover:bg-zinc-50 disabled:opacity-50"
        >
          Review import
        </button>
        {message && (
          <p role="status" className="mt-1 text-xs text-amber-800">
            {message}
          </p>
        )}
      </div>
    );
  }

  if (proposal.status !== "PENDING") {
    return (
      <p role="status" className="mt-2 text-xs text-emerald-800">
        {message || "Import proposal resolved."}
      </p>
    );
  }

  const downloadHost = new URL(pdfUrl).host;
  return (
    <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs">
      <p>
        Confirm download from {downloadHost}? This catalog marks the PDF open
        access. After approval, mYrA downloads it and queues normal ingestion.
      </p>
      {message && (
        <p role="status" className="mt-1 text-amber-900">
          {message}
        </p>
      )}
      <div className="mt-2 flex gap-2">
        <button
          type="button"
          disabled={busy}
          onClick={() => void decide(true)}
          className="rounded bg-emerald-700 px-2 py-1 text-white disabled:opacity-50"
        >
          Approve and import
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => void decide(false)}
          className="rounded border border-zinc-300 px-2 py-1 disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  );
}
