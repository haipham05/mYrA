"use client";

import { useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import MemoryInspector from "@/components/MemoryInspector";
import PaperUploader from "@/components/PaperUploader";
import PdfViewer from "@/components/PdfViewer";
import ProjectSelector from "@/components/ProjectSelector";
import type {
  Citation,
  Conversation,
  MemorySource,
  Message,
  Paper,
  Project,
} from "@/types";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"workspace" | "memory">(
    "workspace",
  );
  const [apiStatus, setApiStatus] = useState("Checking API…");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [deepseekStatus, setDeepseekStatus] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);

  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  const apiUrl = rawApiUrl.replace("localhost", "127.0.0.1");

  // 1. Health check & system status
  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then(() => setApiStatus("Connected"))
      .catch(() => setApiStatus("Unavailable"));

    fetch(`${apiUrl}/api/v1/system/status`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => {
        setDeepseekStatus(
          data.deepseek_configured
            ? "DeepSeek configured"
            : "Test provider mode",
        );
      })
      .catch(() => {});
  }, [apiUrl]);

  // 2. Fetch projects
  useEffect(() => {
    let ignore = false;
    async function loadProjects() {
      try {
        const res = await fetch(`${apiUrl}/api/v1/projects`);
        if (res.ok && !ignore) {
          const data = await res.json();
          const items: Project[] = data.items || [];
          setProjects(items);
          if (items.length > 0) {
            setSelectedProject((current) => current || items[0]);
          }
        }
      } catch {
        // Ignore network errors on initial mount
      }
    }
    loadProjects();
    return () => {
      ignore = true;
    };
  }, [apiUrl]);

  // 3. Fetch papers & conversations when project changes
  useEffect(() => {
    if (!selectedProject) return;
    let ignore = false;

    async function loadProjectDetails() {
      try {
        if (!ignore) {
          setActiveCitation(null);
          setChatError(null);
        }

        const papersRes = await fetch(
          `${apiUrl}/api/v1/projects/${selectedProject?.id}/papers`,
        );

        if (!ignore && papersRes.ok) {
          const pData = await papersRes.json();
          const items: Paper[] = pData.items || [];
          setPapers(items);
          if (items.length > 0) {
            setSelectedPaper(items[0]);
          } else {
            setSelectedPaper(null);
          }
        } else if (!ignore) {
          setPapers([]);
          setSelectedPaper(null);
        }

        // Restore conversations for project
        const listConvRes = await fetch(
          `${apiUrl}/api/v1/projects/${selectedProject?.id}/conversations?include_archived=true`,
        );
        let convList: Conversation[] = [];
        if (listConvRes.ok) {
          const data = await listConvRes.json();
          convList = Array.isArray(data) ? data : data.items || [];
        }

        let activeConv: Conversation | null = null;
        if (convList.length > 0) {
          activeConv = convList[0];
        } else {
          const createConvRes = await fetch(
            `${apiUrl}/api/v1/projects/${selectedProject?.id}/conversations`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ title: "Workspace Chat" }),
            },
          );
          if (createConvRes.ok) {
            activeConv = await createConvRes.json();
            if (activeConv) {
              convList = [activeConv];
            }
          }
        }

        if (!ignore) {
          setConversations(convList);
          if (activeConv) {
            setConversation(activeConv);
            const msgsRes = await fetch(
              `${apiUrl}/api/v1/conversations/${activeConv.id}/messages`,
            );
            if (!ignore && msgsRes.ok) {
              const msgs = await msgsRes.json();
              setMessages(msgs);
            } else if (!ignore) {
              setMessages([]);
            }
          } else {
            setConversation(null);
            setMessages([]);
          }
        }
      } catch {
        // Ignore network errors
      }
    }

    loadProjectDetails();
    return () => {
      ignore = true;
    };
  }, [apiUrl, selectedProject]);

  // Manual refresh of papers
  const refreshPapers = async () => {
    if (!selectedProject) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${selectedProject.id}/papers`,
      );
      if (res.ok) {
        const data = await res.json();
        const items: Paper[] = data.items || [];
        setPapers(items);
        if (items.length > 0 && !selectedPaper) {
          setSelectedPaper(items[0]);
        }
      }
    } catch {
      // Ignore
    }
  };

  // Project creation
  const handleCreateProject = async (name: string, description?: string) => {
    const res = await fetch(`${apiUrl}/api/v1/projects`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description }),
    });
    if (res.ok) {
      const newProj = await res.json();
      setProjects((prev) => [newProj, ...prev]);
      setSelectedProject(newProj);
    }
  };

  // Switch conversation
  const handleSelectConversation = async (conv: Conversation) => {
    setConversation(conv);
    setActiveCitation(null);
    try {
      const msgsRes = await fetch(
        `${apiUrl}/api/v1/conversations/${conv.id}/messages`,
      );
      if (msgsRes.ok) {
        const msgs = await msgsRes.json();
        setMessages(msgs);
      } else {
        setMessages([]);
      }
    } catch {
      setMessages([]);
    }
  };

  // Jump from memory source to paper or conversation in workspace
  const handleSelectMemorySource = (source: MemorySource) => {
    setActiveTab("workspace");
    if (source.source_type === "PAPER_CHUNK" && source.paper_id) {
      const targetPaper = papers.find((p) => p.id === source.paper_id);
      if (targetPaper) {
        setSelectedPaper(targetPaper);
      }
      if (source.page_number && source.quote_text) {
        setActiveCitation({
          citation_index: 1,
          evidence_id: `mem-src-${source.id}`,
          paper_id: source.paper_id,
          page_number: source.page_number,
          bounding_boxes: source.bounding_boxes || [],
          quote: source.quote_text,
          document_sha256: source.document_sha256,
          parser_version: source.parser_version,
          anchor_status: source.anchor_status || "unresolved",
          anchors: source.anchors || [],
        });
      }
    } else if (source.source_type === "MESSAGE") {
      if (source.conversation_id) {
        const targetConv = conversations.find(
          (c) => c.id === source.conversation_id,
        );
        if (targetConv) {
          handleSelectConversation(targetConv);
        }
      }
    }
  };

  // Create conversation
  const handleCreateConversation = async (title?: string) => {
    if (!selectedProject) return;
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/projects/${selectedProject.id}/conversations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: title || "New Research Chat" }),
        },
      );
      if (res.ok) {
        const newConv: Conversation = await res.json();
        setConversations((prev) => [newConv, ...prev]);
        setConversation(newConv);
        setMessages([]);
        setActiveCitation(null);
      } else {
        setChatError("Failed to create conversation.");
      }
    } catch {
      setChatError("Network error: Failed to create conversation.");
    }
  };

  // Rename conversation
  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      const res = await fetch(`${apiUrl}/api/v1/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
      if (res.ok) {
        const updated: Conversation = await res.json();
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, title: updated.title } : c)),
        );
        if (conversation?.id === id) {
          setConversation((prev) =>
            prev ? { ...prev, title: updated.title } : prev,
          );
        }
      } else {
        setChatError("Failed to rename conversation.");
      }
    } catch {
      setChatError("Network error: Failed to rename conversation.");
    }
  };

  // Archive / unarchive conversation
  const handleArchiveConversation = async (id: string, isArchived: boolean) => {
    try {
      const res = await fetch(`${apiUrl}/api/v1/conversations/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_archived: isArchived }),
      });
      if (res.ok) {
        const updated: Conversation = await res.json();
        setConversations((prev) =>
          prev.map((c) =>
            c.id === id ? { ...c, is_archived: updated.is_archived } : c,
          ),
        );
        if (conversation?.id === id) {
          setConversation((prev) =>
            prev ? { ...prev, is_archived: updated.is_archived } : prev,
          );
        }
      } else {
        setChatError("Failed to update archive status.");
      }
    } catch {
      setChatError("Network error: Failed to update archive status.");
    }
  };

  // Delete conversation
  const handleDeleteConversation = async (id: string) => {
    try {
      const res = await fetch(`${apiUrl}/api/v1/conversations/${id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        const remaining = conversations.filter((c) => c.id !== id);
        setConversations(remaining);
        if (conversation?.id === id) {
          if (remaining.length > 0) {
            handleSelectConversation(remaining[0]);
          } else {
            setConversation(null);
            setMessages([]);
            setActiveCitation(null);
          }
        }
      } else {
        setChatError("Failed to delete conversation.");
      }
    } catch {
      setChatError("Network error: Failed to delete conversation.");
    }
  };

  // Send QA Question
  const handleSendMessage = async (content: string) => {
    if (!conversation || isAsking) return;
    setIsAsking(true);
    setChatError(null);
    try {
      const res = await fetch(
        `${apiUrl}/api/v1/conversations/${conversation.id}/messages`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        },
      );
      if (res.ok) {
        const assistantMsg: Message = await res.json();
        const userMsg: Message = {
          id: `usr-${assistantMsg.id}`,
          conversation_id: conversation.id,
          role: "USER",
          content,
          citations: [],
          evidence: [],
          created_at: assistantMsg.created_at,
        };
        setMessages((prev) => [...prev, userMsg, assistantMsg]);

        // Auto-select first citation if available
        if (assistantMsg.citations && assistantMsg.citations.length > 0) {
          handleCitationClick(assistantMsg.citations[0]);
        }
      } else {
        const errData = await res.json().catch(() => ({}));
        setChatError(
          errData.detail || "Failed to generate answer. Please try again.",
        );
      }
    } catch (err: unknown) {
      setChatError(
        err instanceof Error ? err.message : "Network error. Please try again.",
      );
    } finally {
      setIsAsking(false);
    }
  };

  // Handle Citation click
  const handleCitationClick = (citation: Citation) => {
    setActiveCitation(citation);
    const citedPaper = papers.find((p) => p.id === citation.paper_id);
    if (citedPaper) {
      setSelectedPaper(citedPaper);
    }
  };

  // Handle project selection with immediate state clearing
  const handleSelectProject = (project: Project | null) => {
    if (selectedProject?.id === project?.id) return;
    setSelectedProject(project);
    setSelectedPaper(null);
    setActiveCitation(null);
    setPapers([]);
    setConversations([]);
    setConversation(null);
    setMessages([]);
  };

  const hasReadyPaper = papers.some((p) => p.status === "READY");

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans text-zinc-900">
      {/* Top Header Bar */}
      <header className="border-b border-zinc-200 bg-white px-6 py-4 shadow-2xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div>
            <p className="text-xs font-medium text-zinc-500">
              My Research Assistant
            </p>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-950">
              mYrA
            </h1>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-1 bg-zinc-100 p-1 rounded-lg">
              <button
                type="button"
                onClick={() => setActiveTab("workspace")}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition ${
                  activeTab === "workspace"
                    ? "bg-white text-zinc-900 shadow-2xs"
                    : "text-zinc-600 hover:text-zinc-900"
                }`}
              >
                Workspace
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("memory")}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition ${
                  activeTab === "memory"
                    ? "bg-white text-zinc-900 shadow-2xs"
                    : "text-zinc-600 hover:text-zinc-900"
                }`}
              >
                Project Memory
              </button>
            </div>

            <ProjectSelector
              projects={projects}
              selectedProject={selectedProject}
              onSelectProject={handleSelectProject}
              onCreateProject={handleCreateProject}
            />

            <div className="text-xs text-zinc-600">
              API:{" "}
              <span className="font-medium text-zinc-900">{apiStatus}</span>
              {deepseekStatus && (
                <span className="ml-2 font-normal text-zinc-500">
                  · {deepseekStatus}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 p-6">
        {activeTab === "memory" ? (
          <section className="flex-1">
            <MemoryInspector
              projectId={selectedProject?.id || null}
              apiUrl={apiUrl}
              onSelectSource={handleSelectMemorySource}
            />
          </section>
        ) : (
          <>
            {/* Paper Ingestion Section */}
            <section>
              <PaperUploader
                projectId={selectedProject?.id || null}
                apiUrl={apiUrl}
                papers={papers}
                selectedPaper={selectedPaper}
                onPaperSelect={setSelectedPaper}
                onUploadSuccess={refreshPapers}
              />
            </section>

            {/* Split Screen QA and PDF Citation Viewer */}
            <section className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-2 min-h-[600px]">
              {/* Left: Chat QA */}
              <div className="flex flex-col h-[650px]">
                <ChatPanel
                  messages={messages}
                  isLoading={isAsking}
                  error={chatError}
                  onDismissError={() => setChatError(null)}
                  onSendMessage={handleSendMessage}
                  onCitationClick={handleCitationClick}
                  activeCitation={activeCitation}
                  disabled={!hasReadyPaper || !conversation}
                  conversations={conversations}
                  activeConversation={conversation}
                  onSelectConversation={handleSelectConversation}
                  onCreateConversation={handleCreateConversation}
                  onRenameConversation={handleRenameConversation}
                  onArchiveConversation={handleArchiveConversation}
                  onDeleteConversation={handleDeleteConversation}
                />
              </div>

              {/* Right: PDF Evidence Viewer */}
              <div className="flex flex-col h-[650px]">
                <PdfViewer
                  paper={selectedPaper}
                  activeCitation={activeCitation}
                  apiUrl={apiUrl}
                />
              </div>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
