"use client";

import { useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { visit } from "unist-util-visit";
import type { PhrasingContent, Root } from "mdast";
import ConversationNavigator from "@/components/ConversationNavigator";
import type {
  AssistantApprovalResponse,
  AssistantIntent,
  AssistantRunResponse,
  Citation,
  Conversation,
  Message,
  SourceSelection,
} from "@/types";

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  routedIntent?: AssistantIntent | null;
  error?: string | null;
  onDismissError?: () => void;
  onSendMessage: (content: string) => Promise<void>;
  onCitationClick: (citation: Citation) => void;
  activeCitation: Citation | null;
  disabled: boolean;
  conversations?: Conversation[];
  activeConversation?: Conversation | null;
  onSelectConversation?: (conv: Conversation) => void;
  onCreateConversation?: (title?: string) => Promise<void>;
  onRenameConversation?: (id: string, newTitle: string) => Promise<void>;
  onArchiveConversation?: (id: string, isArchived: boolean) => Promise<void>;
  onDeleteConversation?: (id: string) => Promise<void>;
  activeRun?: AssistantRunResponse | null;
  pendingAction?: AssistantApprovalResponse | null;
  onCancelRun?: (runId: string) => Promise<void>;
  onResumeRun?: (runId: string, input: string) => Promise<void>;
  onDecideAction?: (actionId: string, approve: boolean) => Promise<void>;
  sourceSelection?: SourceSelection | null;
  onClearSourceSelection?: () => void;
}

export default function ChatPanel({
  messages,
  isLoading,
  routedIntent,
  error,
  onDismissError,
  onSendMessage,
  onCitationClick,
  activeCitation,
  disabled,
  conversations,
  activeConversation,
  onSelectConversation,
  onCreateConversation,
  onRenameConversation,
  onArchiveConversation,
  onDeleteConversation,
  activeRun,
  pendingAction,
  onCancelRun,
  onResumeRun,
  onDecideAction,
  sourceSelection,
  onClearSourceSelection,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [resumeInput, setResumeInput] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || disabled) return;
    const content = input.trim();
    setInput("");
    await onSendMessage(content);
  };

  const renderContent = (content: string, citations: Citation[]) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath, remarkCitationLinks]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
      skipHtml
      urlTransform={(url) =>
        url.startsWith("myra-citation:") ? url : defaultUrlTransform(url)
      }
      components={{
        h1: ({ children }) => (
          <h1 className="my-3 text-lg font-semibold">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="my-3 text-base font-semibold">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="my-2 font-semibold">{children}</h3>
        ),
        p: ({ children }) => (
          <p className="my-2 whitespace-pre-wrap">{children}</p>
        ),
        ul: ({ children }) => (
          <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l-2 border-zinc-300 pl-3 text-zinc-600">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="my-3 overflow-x-auto">
            <table className="min-w-full border-collapse text-left">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-zinc-300 bg-zinc-50 px-2 py-1">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-zinc-300 px-2 py-1 align-top">
            {children}
          </td>
        ),
        code: ({ children }) => (
          <code className="rounded bg-zinc-200 px-1 py-0.5 font-mono text-[0.9em]">
            {children}
          </code>
        ),
        a: ({ href, children }) => {
          const citationMatch = href?.match(/^myra-citation:(\d+)$/);
          if (!citationMatch) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-blue-700 underline"
              >
                {children}
              </a>
            );
          }
          const citationIndex = Number(citationMatch[1]);
          const citation = citations.find(
            (item) => item.citation_index === citationIndex,
          );
          if (!citation) return <>{children}</>;
          const isSelected = activeCitation?.citation_index === citationIndex;
          return (
            <button
              type="button"
              onClick={() => onCitationClick(citation)}
              className={`mx-0.5 inline-flex cursor-pointer items-center rounded px-1.5 py-0.5 text-xs font-semibold ${
                isSelected
                  ? "bg-amber-400 text-amber-950 ring-2 ring-amber-500"
                  : "bg-blue-100 text-blue-800 hover:bg-blue-200"
              }`}
              title={`Click to view Page ${citation.page_number} evidence`}
            >
              {children}
            </button>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xs">
      <div className="border-b border-zinc-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">
          Research Assistant
        </h2>
        <p className="text-xs text-zinc-500">
          Ask naturally to read, compare, verify, or question your research.
          Click any [n] chip to jump to evidence.
        </p>
        {routedIntent && (
          <p className="mt-2 text-xs text-blue-700" role="status">
            Using: {intentLabel(routedIntent)}
          </p>
        )}
      </div>

      {conversations &&
        onSelectConversation &&
        onCreateConversation &&
        onRenameConversation &&
        onDeleteConversation && (
          <ConversationNavigator
            conversations={conversations}
            activeConversation={activeConversation || null}
            onSelectConversation={onSelectConversation}
            onCreateConversation={onCreateConversation}
            onRenameConversation={onRenameConversation}
            onArchiveConversation={onArchiveConversation}
            onDeleteConversation={onDeleteConversation}
            disabled={disabled}
          />
        )}

      {/* Error banner */}
      {error && (
        <div className="flex items-center justify-between border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-800">
          <span>{error}</span>
          {onDismissError && (
            <button
              type="button"
              onClick={onDismissError}
              className="text-red-500 hover:text-red-700 cursor-pointer ml-2"
              title="Dismiss error"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          )}
        </div>
      )}

      {activeRun && (
        <section
          className="border-b border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-700"
          aria-label="Active research run"
        >
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="font-medium">
                {activeRun.action_summary ||
                  activeRun.intent ||
                  "Research request"}
              </p>
              <p role="status">
                {activeRun.status.replaceAll("_", " ")}
                {activeRun.stage ? ` · ${activeRun.stage}` : ""}
              </p>
            </div>
            {(activeRun.status === "QUEUED" ||
              activeRun.status === "RUNNING") &&
              onCancelRun && (
                <button
                  type="button"
                  onClick={() => void onCancelRun(activeRun.id)}
                  className="rounded border border-zinc-300 px-2 py-1 hover:bg-white"
                >
                  Cancel run
                </button>
              )}
          </div>
          {activeRun.status === "NEEDS_INPUT" && onResumeRun && (
            <form
              className="mt-3 flex gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                if (resumeInput.trim()) {
                  void onResumeRun(activeRun.id, resumeInput.trim());
                  setResumeInput("");
                }
              }}
            >
              <input
                aria-label="Clarification"
                value={resumeInput}
                onChange={(event) => setResumeInput(event.target.value)}
                placeholder="Answer the clarification…"
                className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1.5"
              />
              <button
                type="submit"
                disabled={!resumeInput.trim()}
                className="rounded bg-zinc-900 px-3 py-1.5 text-white disabled:opacity-40"
              >
                Continue
              </button>
            </form>
          )}
          {activeRun.status === "AWAITING_APPROVAL" &&
            pendingAction &&
            onDecideAction && (
              <div className="mt-3 rounded border border-amber-300 bg-amber-50 p-3">
                <p className="font-medium">
                  Review proposed action:{" "}
                  {pendingAction.action_type.replaceAll("_", " ")}
                </p>
                <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-words text-[11px]">
                  {JSON.stringify(pendingAction.arguments, null, 2)}
                </pre>
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    onClick={() => void onDecideAction(pendingAction.id, true)}
                    className="rounded bg-emerald-700 px-3 py-1.5 text-white"
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    onClick={() => void onDecideAction(pendingAction.id, false)}
                    className="rounded border border-zinc-300 px-3 py-1.5"
                  >
                    Reject
                  </button>
                </div>
              </div>
            )}
        </section>
      )}

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-zinc-400 text-sm">
            {activeConversation
              ? "Ask a question over the indexed papers in this project."
              : "Click '+ New Chat' to begin asking questions."}
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === "USER" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === "USER"
                    ? "bg-zinc-900 text-white"
                    : "bg-zinc-100 text-zinc-900 border border-zinc-200"
                }`}
              >
                {msg.role === "ASSISTANT"
                  ? renderContent(msg.content, msg.citations)
                  : msg.content}
              </div>
              <span className="mt-1 text-[10px] text-zinc-400">
                {msg.role === "USER" ? "You" : "mYrA"}
              </span>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex items-start">
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-2.5 text-sm text-zinc-600 animate-pulse">
              {routedIntent
                ? `${intentLabel(routedIntent)} in progress…`
                : "Routing your research request…"}
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="border-t border-zinc-200 p-3">
        {sourceSelection && (
          <div className="mb-2 flex items-start justify-between gap-2 rounded border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
            <span>
              Selected passage from page {sourceSelection.page_number}. Your
              next question will be checked against this exact text.
            </span>
            {onClearSourceSelection && (
              <button
                type="button"
                onClick={onClearSourceSelection}
                className="shrink-0 underline"
              >
                Clear
              </button>
            )}
          </div>
        )}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={disabled || isLoading}
            placeholder={
              disabled
                ? !activeConversation
                  ? "Create a chat to begin asking questions…"
                  : "Upload and index a paper to begin asking questions…"
                : "Ask a grounded research question…"
            }
            className="flex-1 rounded-lg border border-zinc-300 px-3 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:border-zinc-900 focus:outline-hidden disabled:bg-zinc-50 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading || disabled}
            className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

function remarkCitationLinks() {
  return (tree: Root) => {
    visit(tree, "text", (node, index, parent) => {
      if (index === undefined || !parent || !/\[\d+\]/.test(node.value)) return;
      const children: PhrasingContent[] = node.value
        .split(/(\[\d+\])/g)
        .filter(Boolean)
        .map((part) => {
          const match = part.match(/^\[(\d+)\]$/);
          return match
            ? {
                type: "link",
                url: `myra-citation:${match[1]}`,
                children: [{ type: "text", value: part }],
              }
            : { type: "text", value: part };
        });
      parent.children.splice(index, 1, ...children);
      return index + children.length;
    });
  };
}

function intentLabel(intent: AssistantIntent): string {
  const labels: Record<AssistantIntent, string> = {
    help: "Help",
    qa: "Question answering",
    read_paper: "Paper reading",
    compare: "Paper comparison",
    verify_claim: "Claim verification",
    discover: "Paper discovery",
    notes: "Research notes",
    report: "Report drafting",
    research: "Research workflow",
    gap_analysis: "Gap analysis",
    experiment_plan: "Experiment planning",
    translate: "PDF translation",
    vision: "Figure understanding",
    graph: "Graph exploration",
    clarify: "Clarification",
  };
  return labels[intent];
}
