"use client";

import { useState } from "react";
import ConversationNavigator from "@/components/ConversationNavigator";
import type { Citation, Conversation, Message } from "@/types";

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  onSendMessage: (content: string) => Promise<void>;
  onCitationClick: (citation: Citation) => void;
  activeCitation: Citation | null;
  disabled: boolean;
  conversations?: Conversation[];
  activeConversation?: Conversation | null;
  onSelectConversation?: (conv: Conversation) => void;
  onCreateConversation?: (title?: string) => Promise<void>;
  onRenameConversation?: (id: string, newTitle: string) => Promise<void>;
  onDeleteConversation?: (id: string) => Promise<void>;
}

export default function ChatPanel({
  messages,
  isLoading,
  onSendMessage,
  onCitationClick,
  activeCitation,
  disabled,
  conversations,
  activeConversation,
  onSelectConversation,
  onCreateConversation,
  onRenameConversation,
  onDeleteConversation,
}: ChatPanelProps) {
  const [input, setInput] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || disabled) return;
    const content = input.trim();
    setInput("");
    await onSendMessage(content);
  };

  const renderContentWithCitations = (
    content: string,
    citations: Citation[],
  ) => {
    const parts = content.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citeIndex = parseInt(match[1], 10);
        const citation = citations.find((c) => c.citation_index === citeIndex);
        if (citation) {
          const isSelected = activeCitation?.citation_index === citeIndex;
          return (
            <button
              key={`cite-${index}`}
              type="button"
              onClick={() => onCitationClick(citation)}
              className={`inline-flex items-center justify-center rounded px-1.5 py-0.5 text-xs font-semibold transition-colors mx-0.5 cursor-pointer ${
                isSelected
                  ? "bg-amber-400 text-amber-950 ring-2 ring-amber-500"
                  : "bg-blue-100 text-blue-800 hover:bg-blue-200"
              }`}
              title={`Click to view Page ${citation.page_number} evidence`}
            >
              [{citeIndex}]
            </button>
          );
        }
      }
      return <span key={`text-${index}`}>{part}</span>;
    });
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xs">
      <div className="border-b border-zinc-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">
          Research QA Chat
        </h2>
        <p className="text-xs text-zinc-500">
          Grounded answers citing uploaded papers. Click any [n] chip to jump to
          evidence.
        </p>
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
            onDeleteConversation={onDeleteConversation}
            disabled={disabled}
          />
        )}

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-zinc-400 text-sm">
            Ask a question over the indexed papers in this project.
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
                  ? renderContentWithCitations(msg.content, msg.citations)
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
              Retrieving evidence & generating grounded answer…
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="border-t border-zinc-200 p-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={disabled || isLoading}
            placeholder={
              disabled
                ? "Upload and index a paper to begin asking questions…"
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
