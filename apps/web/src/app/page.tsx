"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import ChatPanel from "@/components/ChatPanel";
import MemoryInspector from "@/components/MemoryInspector";
import PaperMetadataEditor from "@/components/PaperMetadataEditor";
import PaperUploader from "@/components/PaperUploader";
import PdfViewer from "@/components/PdfViewer";
import ProjectSelector from "@/components/ProjectSelector";
import TranslationPanel from "@/components/TranslationPanel";
import type {
  Citation,
  Conversation,
  AssistantIntent,
  AssistantApprovalResponse,
  AssistantRunResponse,
  MemorySource,
  Message,
  Paper,
  ProviderBudgetUsage,
  Project,
  SourceSelection,
} from "@/types";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"workspace" | "memory">(
    "workspace",
  );
  const [apiStatus, setApiStatus] = useState("Checking API…");
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [paperTotal, setPaperTotal] = useState(0);
  const [paperOffset, setPaperOffset] = useState(0);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [sourceSelection, setSourceSelection] =
    useState<SourceSelection | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [paperScope, setPaperScope] = useState<
    "paper" | "selection" | "project"
  >("project");
  const [selectedPaperIds, setSelectedPaperIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [routedIntent, setRoutedIntent] = useState<AssistantIntent | null>(
    null,
  );
  const [activeRun, setActiveRun] = useState<AssistantRunResponse | null>(null);
  const [pendingAction, setPendingAction] =
    useState<AssistantApprovalResponse | null>(null);
  const [deepseekStatus, setDeepseekStatus] = useState<string | null>(null);
  const [budgetUsage, setBudgetUsage] = useState<ProviderBudgetUsage | null>(
    null,
  );
  const [chatError, setChatError] = useState<string | null>(null);

  const activeSourceSelection =
    sourceSelection &&
    selectedPaper &&
    sourceSelection.paper_id === selectedPaper.id &&
    sourceSelection.document_sha256 === selectedPaper.document_sha256
      ? sourceSelection
      : null;

  const rawApiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
  const apiUrl = rawApiUrl.replace("localhost", "127.0.0.1");

  const handleCitationClick = useCallback(
    (citation: Citation) => {
      setActiveCitation(citation);
      const citedPaper = papers.find((paper) => paper.id === citation.paper_id);
      if (citedPaper) {
        setSelectedPaper(citedPaper);
      } else {
        fetch(`${apiUrl}/api/v1/papers/${citation.paper_id}`)
          .then((response) => (response.ok ? response.json() : null))
          .then((paper: Paper | null) => {
            if (paper) setSelectedPaper(paper);
          })
          .catch(() => {});
      }
    },
    [apiUrl, papers],
  );

  const pollAssistantRun = useCallback(
    async (initialRun: AssistantRunResponse) => {
      let run = initialRun;
      const runKey = `myra.activeRun.${run.conversation_id}`;
      const isCurrentConversation = () =>
        conversation?.id === run.conversation_id;
      const updateRun = (next: AssistantRunResponse) => {
        run = next;
        if (isCurrentConversation()) {
          setActiveRun(next);
          setRoutedIntent(next.intent);
        }
      };
      const loadPendingAction = async () => {
        const response = await fetch(
          `${apiUrl}/api/v1/runs/${run.id}/actions`,
          {
            cache: "no-store",
          },
        );
        if (!response.ok)
          throw new Error("Could not load the proposed action.");
        const actions: AssistantApprovalResponse[] = await response.json();
        if (isCurrentConversation()) {
          setPendingAction(
            actions.find((action) => action.status === "PENDING") ?? null,
          );
        }
      };

      updateRun(run);
      for (let poll = 0; poll < 240; poll += 1) {
        if (run.status !== "QUEUED" && run.status !== "RUNNING") break;
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        const response = await fetch(`${apiUrl}/api/v1/runs/${run.id}`, {
          cache: "no-store",
        });
        if (!response.ok)
          throw new Error("Could not read research run status.");
        updateRun(await response.json());
      }

      if (run.status === "AWAITING_APPROVAL") await loadPendingAction();
      else if (isCurrentConversation()) setPendingAction(null);

      if (run.status === "QUEUED" || run.status === "RUNNING") {
        if (isCurrentConversation()) {
          setChatError(`This request is still processing (run ${run.id}).`);
        }
        return;
      }
      if (run.status === "NEEDS_INPUT" || run.status === "AWAITING_APPROVAL")
        return;

      window.localStorage.removeItem(runKey);
      if (isCurrentConversation()) {
        setActiveRun(null);
        if (run.result) {
          const payloadMessageId = run.result.structured_payload.message_id;
          const assistantMessage: Message = {
            id:
              typeof payloadMessageId === "string" ? payloadMessageId : run.id,
            conversation_id: run.conversation_id,
            role: "ASSISTANT",
            content: run.result.display_text,
            citations: run.result.citations ?? [],
            evidence: [],
            model_name:
              typeof run.result.usage.model_name === "string"
                ? run.result.usage.model_name
                : null,
            assistantResult: run.result,
            created_at: run.updated_at,
          };
          setMessages((previous) =>
            previous.some((message) => message.id === assistantMessage.id)
              ? previous
              : [...previous, assistantMessage],
          );
          if (assistantMessage.citations.length > 0) {
            handleCitationClick(assistantMessage.citations[0]);
          }
        } else if (run.status === "CANCELLED") {
          setChatError("The research request was cancelled.");
        } else if (run.status === "FAILED") {
          setChatError(
            `The research request could not be completed${run.safe_error ? ` (${run.safe_error})` : ""}.`,
          );
        }
      }
    },
    [apiUrl, conversation?.id, handleCitationClick],
  );

  useEffect(() => {
    if (!conversation) return;
    const runId = window.localStorage.getItem(
      `myra.activeRun.${conversation.id}`,
    );
    if (!runId) {
      return;
    }
    let ignore = false;
    fetch(`${apiUrl}/api/v1/runs/${runId}`, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Saved run is unavailable.");
        const run: AssistantRunResponse = await response.json();
        if (!ignore) {
          setIsAsking(true);
          await pollAssistantRun(run);
        }
      })
      .catch(() => {
        if (!ignore) {
          window.localStorage.removeItem(`myra.activeRun.${conversation.id}`);
          setActiveRun(null);
          setPendingAction(null);
        }
      })
      .finally(() => {
        if (!ignore) setIsAsking(false);
      });
    return () => {
      ignore = true;
    };
  }, [apiUrl, conversation, pollAssistantRun]);

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

    fetch(`${apiUrl}/api/v1/budget/usage`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data: ProviderBudgetUsage) => setBudgetUsage(data))
      .catch(() => setBudgetUsage(null));
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
            let matchedProj: Project | undefined;
            if (typeof window !== "undefined") {
              const params = new URLSearchParams(window.location.search);
              const targetProjId = params.get("project");
              if (targetProjId) {
                matchedProj = items.find((p) => p.id === targetProjId);
              }
            }
            setSelectedProject((current) => current || matchedProj || items[0]);
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
          setPaperTotal(pData.total || 0);
          setPaperOffset(pData.offset || 0);
          if (items.length > 0) {
            let matchedPaper: Paper | undefined;
            if (typeof window !== "undefined") {
              const params = new URLSearchParams(window.location.search);
              const targetPaperId = params.get("paper");
              if (targetPaperId) {
                matchedPaper = items.find((p) => p.id === targetPaperId);
              }
            }
            setSelectedPaper(matchedPaper || items[0]);
          } else {
            setSelectedPaper(null);
          }
        } else if (!ignore) {
          setPapers([]);
          setPaperTotal(0);
          setPaperOffset(0);
          setSelectedPaper(null);
        }

        // Check for fact deep link
        if (typeof window !== "undefined" && selectedProject?.id) {
          const params = new URLSearchParams(window.location.search);
          const targetFactId = params.get("fact");
          const targetPageNum = params.get("page");
          if (targetFactId) {
            fetch(
              `${apiUrl}/api/v1/projects/${selectedProject.id}/graph/facts/${targetFactId}`,
            )
              .then((res) => (res.ok ? res.json() : null))
              .then((fact) => {
                if (!ignore && fact) {
                  if (fact.citation) {
                    setActiveCitation(fact.citation);
                  } else if (fact.exact_quote) {
                    setActiveCitation({
                      citation_index: 1,
                      evidence_id: `graph-fact-${fact.id}`,
                      paper_id: fact.paper_id,
                      page_number:
                        fact.page_number ||
                        (targetPageNum ? parseInt(targetPageNum, 10) : 1),
                      bounding_boxes: [],
                      quote: fact.exact_quote,
                      document_sha256: fact.document_sha256,
                      anchor_status: fact.anchor_status || "unresolved",
                    });
                  }
                }
              })
              .catch(() => {});
          }
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
            const created: Conversation = await createConvRes.json();
            if (created && created.id) {
              activeConv = created;
              convList = [created];
            }
          }
        }

        if (!ignore) {
          setConversations(convList);
          if (activeConv) {
            setConversation(activeConv);
            setPaperScope(activeConv.paper_scope ?? "project");
            setSelectedPaperIds(activeConv.selected_paper_ids ?? []);
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
        `${apiUrl}/api/v1/projects/${selectedProject.id}/papers?limit=50&offset=0`,
      );
      if (res.ok) {
        const data = await res.json();
        const items: Paper[] = data.items || [];
        setPapers(items);
        setPaperTotal(data.total || 0);
        setPaperOffset(data.offset || 0);
        if (items.length > 0 && !selectedPaper) {
          setSelectedPaper(items[0]);
        }
      }
    } catch {
      // Ignore
    }
  };

  const searchProjectPapers = async (filters: {
    q?: string;
    status?: string;
    year?: string;
    offset: number;
  }) => {
    if (!selectedProject) return;
    const params = new URLSearchParams({
      limit: "50",
      offset: String(filters.offset),
    });
    if (filters.q) params.set("q", filters.q);
    if (filters.status) params.set("status", filters.status);
    if (filters.year) params.set("year", filters.year);
    const response = await fetch(
      `${apiUrl}/api/v1/projects/${selectedProject.id}/papers?${params.toString()}`,
    );
    if (!response.ok) throw new Error("Could not load the paper list.");
    const data = await response.json();
    const items: Paper[] = data.items || [];
    setPapers(items);
    setPaperTotal(data.total || 0);
    setPaperOffset(data.offset || 0);
    if (
      items.length > 0 &&
      !items.some((paper) => paper.id === selectedPaper?.id)
    ) {
      setSelectedPaper(items[0]);
      setActiveCitation(null);
    } else if (items.length === 0) {
      setSelectedPaper(null);
      setActiveCitation(null);
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
    setPaperScope(conv.paper_scope ?? "project");
    setSelectedPaperIds(conv.selected_paper_ids ?? []);
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
        setPaperScope(newConv.paper_scope ?? "project");
        setSelectedPaperIds(newConv.selected_paper_ids ?? []);
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
    if (!conversation || !selectedProject || isAsking) return;
    setIsAsking(true);
    setChatError(null);
    setRoutedIntent(null);
    try {
      const submitted = await fetch(
        `${apiUrl}/api/v1/conversations/${conversation.id}/runs`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: content,
            conversation_id: conversation.id,
            project_id: selectedProject.id,
            scope: activeSourceSelection ? "paper" : paperScope,
            selected_paper_ids: activeSourceSelection
              ? [activeSourceSelection.paper_id]
              : paperScope === "project"
                ? []
                : selectedPaperIds,
            source_selection: activeSourceSelection ?? undefined,
            idempotency_key: `web-${crypto.randomUUID()}`,
          }),
        },
      );
      if (!submitted.ok) {
        const errData = await submitted.json().catch(() => ({}));
        throw new Error(
          errData.detail || "Could not submit the research request.",
        );
      }

      const run: AssistantRunResponse = await submitted.json();
      setSourceSelection(null);
      window.localStorage.setItem(`myra.activeRun.${conversation.id}`, run.id);
      setActiveRun(run);
      setRoutedIntent(run.intent);
      const userMessage: Message = {
        id: `usr-${run.id}`,
        conversation_id: conversation.id,
        role: "USER",
        content,
        citations: [],
        evidence: [],
        created_at: run.created_at,
      };
      setMessages((previous) => [...previous, userMessage]);
      await pollAssistantRun(run);
    } catch (err: unknown) {
      setChatError(
        err instanceof Error ? err.message : "Network error. Please try again.",
      );
    } finally {
      setIsAsking(false);
    }
  };

  const handleCancelRun = async (runId: string) => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/runs/${runId}/cancel`, {
        method: "POST",
      });
      if (!response.ok) throw new Error("Could not cancel the research run.");
      await pollAssistantRun(await response.json());
    } catch (err) {
      setChatError(
        err instanceof Error ? err.message : "Could not cancel the run.",
      );
    }
  };

  const handleResumeRun = async (runId: string, additionalInput: string) => {
    if (!conversation) return;
    setIsAsking(true);
    try {
      const response = await fetch(`${apiUrl}/api/v1/runs/${runId}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ additional_input: additionalInput }),
      });
      if (!response.ok)
        throw new Error("Could not continue this research run.");
      const run: AssistantRunResponse = await response.json();
      window.localStorage.setItem(`myra.activeRun.${conversation.id}`, run.id);
      await pollAssistantRun(run);
    } catch (err) {
      setChatError(
        err instanceof Error ? err.message : "Could not continue the run.",
      );
    } finally {
      setIsAsking(false);
    }
  };

  const handleDecideAction = async (actionId: string, approve: boolean) => {
    if (!activeRun) return;
    setIsAsking(true);
    try {
      const response = await fetch(
        `${apiUrl}/api/v1/actions/${actionId}/${approve ? "approve" : "reject"}`,
        { method: "POST" },
      );
      if (!response.ok)
        throw new Error("The proposed action could not be updated.");
      setPendingAction(null);
      const runResponse = await fetch(`${apiUrl}/api/v1/runs/${activeRun.id}`, {
        cache: "no-store",
      });
      if (!runResponse.ok) throw new Error("Could not read the updated run.");
      await pollAssistantRun(await runResponse.json());
    } catch (err) {
      setChatError(
        err instanceof Error ? err.message : "Could not update the action.",
      );
    } finally {
      setIsAsking(false);
    }
  };

  // Handle project selection with immediate state clearing
  const handleSelectProject = (project: Project | null) => {
    if (selectedProject?.id === project?.id) return;
    setSelectedProject(project);
    setSelectedPaper(null);
    setActiveCitation(null);
    setPapers([]);
    setPaperTotal(0);
    setPaperOffset(0);
    setConversations([]);
    setConversation(null);
    setPaperScope("project");
    setSelectedPaperIds([]);
    setMessages([]);
  };

  const saveConversationScope = async (
    nextScope: "paper" | "selection" | "project",
    nextPaperIds: string[],
  ) => {
    if (!conversation) throw new Error("Choose a conversation first.");
    const response = await fetch(
      `${apiUrl}/api/v1/conversations/${conversation.id}?project_id=${conversation.project_id}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_scope: nextScope,
          selected_paper_ids: nextPaperIds,
        }),
      },
    );
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Could not save the conversation scope.");
    }
    const updated: Conversation = await response.json();
    setPaperScope(updated.paper_scope ?? "project");
    setSelectedPaperIds(updated.selected_paper_ids ?? []);
    setConversation(updated);
    setConversations((current) =>
      current.map((item) => (item.id === updated.id ? updated : item)),
    );
  };

  const handlePaperScopeChange = async (
    nextScope: "paper" | "selection" | "project",
  ) => {
    let nextIds = selectedPaperIds;
    if (nextScope === "project") nextIds = [];
    if (nextScope === "paper") {
      nextIds = selectedPaper
        ? [selectedPaper.id]
        : selectedPaperIds.slice(0, 1);
    }
    await saveConversationScope(nextScope, nextIds);
  };

  const handleSelectedPaperIdsChange = async (
    paperId: string,
    checked: boolean,
  ) => {
    const nextIds =
      paperScope === "paper"
        ? checked
          ? [paperId]
          : []
        : checked
          ? [...new Set([...selectedPaperIds, paperId])]
          : selectedPaperIds.filter((id) => id !== paperId);
    await saveConversationScope(paperScope, nextIds);
  };

  const handlePaperMetadataUpdated = (updatedPaper: Paper) => {
    setPapers((current) =>
      current.map((paper) =>
        paper.id === updatedPaper.id ? updatedPaper : paper,
      ),
    );
    setSelectedPaper(updatedPaper);
  };

  const hasReadyPaper =
    selectedPaper?.status === "READY" ||
    papers.some((p) => p.status === "READY");

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

            {selectedProject && (
              <Link
                href={`/projects/${selectedProject.id}/graph`}
                className="px-3 py-1.5 text-xs font-medium rounded-md transition text-zinc-600 hover:text-zinc-900 border border-zinc-200 bg-white hover:bg-zinc-50 shadow-2xs"
              >
                Graph Explorer
              </Link>
            )}

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

            <details className="relative text-xs text-zinc-700">
              <summary className="cursor-pointer rounded-md border border-zinc-200 px-3 py-2 font-medium hover:bg-zinc-50">
                Provider usage
              </summary>
              <div className="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-zinc-200 bg-white p-4 shadow-lg">
                {budgetUsage?.status === "available" ? (
                  <>
                    <p className="font-semibold text-zinc-900">
                      Estimated daily allowance
                    </p>
                    <p className="mt-1 text-sm text-zinc-700">
                      $
                      {Number(
                        budgetUsage.daily_remaining_estimate_usd ?? 0,
                      ).toFixed(4)}{" "}
                      remaining of $
                      {Number(budgetUsage.daily_limit_estimate_usd).toFixed(2)}
                    </p>
                    <dl className="mt-3 space-y-1 text-zinc-600">
                      <div className="flex justify-between gap-3">
                        <dt>Estimated committed</dt>
                        <dd>
                          $
                          {Number(
                            budgetUsage.daily.committed_estimate_usd,
                          ).toFixed(4)}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt>Reported prompt tokens</dt>
                        <dd>
                          {budgetUsage.daily.reported_prompt_tokens ??
                            "Unknown"}
                        </dd>
                      </div>
                      <div className="flex justify-between gap-3">
                        <dt>Reported completion tokens</dt>
                        <dd>
                          {budgetUsage.daily.reported_completion_tokens ??
                            "Unknown"}
                        </dd>
                      </div>
                    </dl>
                    <p className="mt-3 text-[11px] leading-4 text-zinc-500">
                      {budgetUsage.usage_note}
                    </p>
                  </>
                ) : (
                  <p className="text-zinc-600">
                    Usage tracking is unavailable for this runtime profile.
                  </p>
                )}
              </div>
            </details>
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
                key={selectedProject?.id ?? "no-project"}
                projectId={selectedProject?.id || null}
                apiUrl={apiUrl}
                papers={papers}
                total={paperTotal}
                offset={paperOffset}
                selectedPaper={selectedPaper}
                onPaperSelect={setSelectedPaper}
                onUploadSuccess={refreshPapers}
                onSearch={searchProjectPapers}
                paperScope={paperScope}
                selectedPaperIds={selectedPaperIds}
                onScopeChange={handlePaperScopeChange}
                onSelectedPaperIdsChange={handleSelectedPaperIdsChange}
              />
              {selectedPaper && (
                <PaperMetadataEditor
                  key={selectedPaper.id}
                  paper={selectedPaper}
                  apiUrl={apiUrl}
                  onUpdated={handlePaperMetadataUpdated}
                />
              )}
            </section>

            {/* Split Screen QA and PDF Citation Viewer */}
            <section className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-2 min-h-[600px]">
              {/* Left: Chat QA */}
              <div className="flex flex-col h-[650px]">
                <ChatPanel
                  messages={messages}
                  isLoading={isAsking}
                  routedIntent={routedIntent}
                  error={chatError}
                  onDismissError={() => setChatError(null)}
                  onSendMessage={handleSendMessage}
                  onCitationClick={handleCitationClick}
                  activeCitation={activeCitation}
                  disabled={
                    !hasReadyPaper ||
                    !conversation ||
                    (paperScope !== "project" && selectedPaperIds.length === 0)
                  }
                  conversations={conversations}
                  activeConversation={conversation}
                  onSelectConversation={handleSelectConversation}
                  onCreateConversation={handleCreateConversation}
                  onRenameConversation={handleRenameConversation}
                  onArchiveConversation={handleArchiveConversation}
                  onDeleteConversation={handleDeleteConversation}
                  activeRun={
                    activeRun?.conversation_id === conversation?.id
                      ? activeRun
                      : null
                  }
                  pendingAction={
                    pendingAction?.run_id === activeRun?.id
                      ? pendingAction
                      : null
                  }
                  onCancelRun={handleCancelRun}
                  onResumeRun={handleResumeRun}
                  onDecideAction={handleDecideAction}
                  sourceSelection={activeSourceSelection}
                  onClearSourceSelection={() => setSourceSelection(null)}
                />
              </div>

              {/* Right: PDF Evidence Viewer */}
              <div className="flex flex-col h-[650px]">
                <PdfViewer
                  paper={selectedPaper}
                  activeCitation={activeCitation}
                  apiUrl={apiUrl}
                  onExplainSelection={setSourceSelection}
                />
              </div>
            </section>

            {selectedProject && selectedPaper && (
              <TranslationPanel
                apiUrl={apiUrl}
                projectId={selectedProject.id}
                paperId={selectedPaper.id}
                paperStatus={selectedPaper.status}
                onOpenSource={(source) => {
                  const verified =
                    source.anchor_status === "verified" &&
                    source.source_char_start !== null &&
                    source.source_char_end !== null &&
                    source.document_sha256 === selectedPaper.document_sha256 &&
                    source.parser_version === "translation-page-rawtext-v1";
                  setActiveCitation({
                    citation_index: 1,
                    evidence_id: `translation-source-${selectedPaper.id}-${source.ordinal}`,
                    paper_id: selectedPaper.id,
                    page_number: source.source_page_number,
                    bounding_boxes: [],
                    quote: source.source_quote,
                    document_sha256: source.document_sha256,
                    parser_version: verified ? source.parser_version : null,
                    anchor_status: verified ? "verified" : "unresolved",
                    anchors: verified
                      ? [
                          {
                            id: `translation-anchor-${selectedPaper.id}-${source.ordinal}`,
                            page_number: source.source_page_number,
                            source_element_id: source.source_element_id,
                            exact_quote: source.source_quote,
                            source_char_start: source.source_char_start,
                            source_char_end: source.source_char_end,
                            document_sha256: source.document_sha256,
                            parser_version: source.parser_version,
                            anchor_status: "verified",
                            bounding_boxes: [],
                          },
                        ]
                      : [],
                  });
                }}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
