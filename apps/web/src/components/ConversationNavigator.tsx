"use client";

import { useState } from "react";
import type { Conversation } from "@/types";

interface ConversationNavigatorProps {
  conversations: Conversation[];
  activeConversation: Conversation | null;
  onSelectConversation: (conv: Conversation) => void;
  onCreateConversation: (title?: string) => Promise<void>;
  onRenameConversation: (id: string, newTitle: string) => Promise<void>;
  onArchiveConversation?: (id: string, isArchived: boolean) => Promise<void>;
  onDeleteConversation: (id: string) => Promise<void>;
  disabled?: boolean;
}

export default function ConversationNavigator({
  conversations,
  activeConversation,
  onSelectConversation,
  onCreateConversation,
  onRenameConversation,
  onArchiveConversation,
  onDeleteConversation,
  disabled = false,
}: ConversationNavigatorProps) {
  const [isCreating, setIsCreating] = useState(false);
  const [newTitle, setNewTitle] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);
  const [editTitle, setEditTitle] = useState("");
  const [loading, setLoading] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    try {
      await onCreateConversation(newTitle.trim() || undefined);
      setNewTitle("");
      setIsCreating(false);
    } finally {
      setLoading(false);
    }
  };

  const handleRename = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeConversation || !editTitle.trim() || loading) return;
    setLoading(true);
    try {
      await onRenameConversation(activeConversation.id, editTitle.trim());
      setIsRenaming(false);
    } finally {
      setLoading(false);
    }
  };

  const handleArchive = async () => {
    if (!activeConversation || !onArchiveConversation || loading) return;
    setLoading(true);
    try {
      await onArchiveConversation(
        activeConversation.id,
        !activeConversation.is_archived,
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!activeConversation || loading) return;
    if (window.confirm("Are you sure you want to delete this conversation?")) {
      setLoading(true);
      try {
        await onDeleteConversation(activeConversation.id);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 bg-zinc-50/50 px-4 py-2">
      {/* Left: Active Chat selector */}
      <div className="flex items-center gap-2">
        <label
          htmlFor="conversation-select"
          className="text-xs font-medium text-zinc-500"
        >
          Chat:
        </label>
        <select
          id="conversation-select"
          value={activeConversation?.id || ""}
          disabled={disabled || conversations.length === 0}
          onChange={(e) => {
            const selected = conversations.find((c) => c.id === e.target.value);
            if (selected) onSelectConversation(selected);
          }}
          className="max-w-[200px] truncate rounded border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-900 focus:border-zinc-900 focus:outline-hidden disabled:opacity-50"
        >
          {conversations.length === 0 ? (
            <option key="empty" value="">
              No conversations
            </option>
          ) : (
            conversations.map((c, idx) => (
              <option key={c.id || `conv-${idx}`} value={c.id || ""}>
                {c.title || "Untitled Chat"}
                {c.is_archived ? " [Archived]" : ""}
                {c.message_count !== undefined ? ` (${c.message_count})` : ""}
              </option>
            ))
          )}
        </select>

        {activeConversation && !isRenaming && (
          <button
            type="button"
            onClick={() => {
              setEditTitle(activeConversation.title || "");
              setIsRenaming(true);
            }}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-200 hover:text-zinc-800 cursor-pointer text-xs"
            title="Rename conversation"
            aria-label="Rename conversation"
          >
            ✎
          </button>
        )}

        {activeConversation && onArchiveConversation && !isRenaming && (
          <button
            type="button"
            onClick={handleArchive}
            disabled={disabled || loading}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-200 hover:text-zinc-800 cursor-pointer text-xs disabled:opacity-40"
            title={
              activeConversation.is_archived
                ? "Unarchive conversation"
                : "Archive conversation"
            }
            aria-label={
              activeConversation.is_archived
                ? "Unarchive conversation"
                : "Archive conversation"
            }
          >
            {activeConversation.is_archived ? "📦" : "📁"}
          </button>
        )}

        {activeConversation && (
          <button
            type="button"
            onClick={handleDelete}
            disabled={disabled || loading}
            className="rounded p-1 text-zinc-400 hover:bg-red-50 hover:text-red-600 cursor-pointer text-xs disabled:opacity-40"
            title="Delete conversation"
            aria-label="Delete conversation"
          >
            ✕
          </button>
        )}
      </div>

      {/* Right: Actions (New Chat, Rename input) */}
      <div className="flex items-center gap-2">
        {isRenaming ? (
          <form onSubmit={handleRename} className="flex items-center gap-1">
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setIsRenaming(false);
                }
              }}
              placeholder="Chat title"
              required
              autoFocus
              className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs text-zinc-900 focus:border-zinc-900 focus:outline-hidden"
            />
            <button
              type="submit"
              disabled={loading || !editTitle.trim()}
              className="rounded bg-zinc-900 px-2 py-0.5 text-xs text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
            >
              Save
            </button>
            <button
              type="button"
              onClick={() => setIsRenaming(false)}
              className="rounded border border-zinc-300 px-2 py-0.5 text-xs text-zinc-600 hover:bg-zinc-100 cursor-pointer"
            >
              Cancel
            </button>
          </form>
        ) : isCreating ? (
          <form onSubmit={handleCreate} className="flex items-center gap-1">
            <input
              type="text"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setIsCreating(false);
                }
              }}
              placeholder="New chat title (optional)"
              autoFocus
              className="rounded border border-zinc-300 bg-white px-2 py-0.5 text-xs text-zinc-900 focus:border-zinc-900 focus:outline-hidden"
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded bg-zinc-900 px-2 py-0.5 text-xs text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
            >
              Start
            </button>
            <button
              type="button"
              onClick={() => setIsCreating(false)}
              className="rounded border border-zinc-300 px-2 py-0.5 text-xs text-zinc-600 hover:bg-zinc-100 cursor-pointer"
            >
              Cancel
            </button>
          </form>
        ) : (
          <button
            type="button"
            onClick={() => setIsCreating(true)}
            disabled={disabled}
            className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40 cursor-pointer"
          >
            + New Chat
          </button>
        )}
      </div>
    </div>
  );
}
