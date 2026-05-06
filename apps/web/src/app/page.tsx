"use client";

import { useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import PaperUploader from "@/components/PaperUploader";
import PdfViewer from "@/components/PdfViewer";
import ProjectSelector from "@/components/ProjectSelector";
import type { Citation, Conversation, Message, Paper, Project } from "@/types";

export default function Home() {
  const [apiStatus, setApiStatus] = useState("Checking API…");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isAsking, setIsAsking] = useState(false);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  // 1. Health check
  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then(() => setApiStatus("Connected"))
      .catch(() => setApiStatus("Unavailable"));
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

  // 3. Fetch papers & active conversation when project changes
  useEffect(() => {
    if (!selectedProject) return;
    let ignore = false;

    async function loadProjectDetails() {
      try {
        const papersRes = await fetch(
          `${apiUrl}/api/v1/projects/${selectedProject?.id}/papers`,
        );

        if (!ignore && papersRes.ok) {
          const pData = await papersRes.json();
          const items: Paper[] = pData.items || [];
          setPapers(items);
          if (items.length > 0) {
            setSelectedPaper(items[0]);
          }
        }

        // Restore existing conversation if available, or create new
        const listConvRes = await fetch(
          `${apiUrl}/api/v1/projects/${selectedProject?.id}/conversations`,
        );
        let activeConv: Conversation | null = null;
        if (listConvRes.ok) {
          const convList = await listConvRes.json();
          if (Array.isArray(convList) && convList.length > 0) {
            activeConv = convList[0];
          }
        }
        if (!activeConv) {
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
          }
        }

        if (!ignore && activeConv) {
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

  // Send QA Question
  const handleSendMessage = async (content: string) => {
    if (!conversation) return;
    setIsAsking(true);
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
          id: `usr-${Date.now()}`,
          conversation_id: conversation.id,
          role: "USER",
          content,
          citations: [],
          evidence: [],
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, userMsg, assistantMsg]);

        // Auto-select first citation if available
        if (assistantMsg.citations && assistantMsg.citations.length > 0) {
          handleCitationClick(assistantMsg.citations[0]);
        }
      }
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
            <ProjectSelector
              projects={projects}
              selectedProject={selectedProject}
              onSelectProject={setSelectedProject}
              onCreateProject={handleCreateProject}
            />

            <div className="text-xs text-zinc-600">
              API:{" "}
              <span className="font-medium text-zinc-900">{apiStatus}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 p-6">
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
              onSendMessage={handleSendMessage}
              onCitationClick={handleCitationClick}
              activeCitation={activeCitation}
              disabled={!hasReadyPaper}
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
      </main>
    </div>
  );
}
