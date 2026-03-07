"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Menu, Settings } from "lucide-react";

import Sidebar from "./components/Sidebar";
import InputArea, { AttachedFile } from "./components/InputArea";
import ChatArea, { Message } from "./components/ChatArea";
import ModelSelectionModal from "@/app/components/ModelSelectionModal";
import ModelBar, { SlotName, SlotStatus } from "./components/ModelBar";
import IngestionProgressModal, { IngestFileEntry } from "./components/IngestionProgressModal";
import { ChatSession, DBMessage } from "./types/chat";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

const EMPTY_SLOTS: SlotStatus = { llm: null, embed: null, ocr: null, asr: null };

function dbMsgToUiMsg(m: DBMessage): Message {
  return {
    id: `db-${m.id}`,
    role: m.role,
    content: m.content,
    isTranscript: m.is_transcript,
  };
}

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isModelSidebarOpen, setIsModelSidebarOpen] = useState(false);
  const [message, setMessage] = useState("");

  const [slots, setSlots] = useState<SlotStatus>(EMPTY_SLOTS);
  const [busySlot, setBusySlot] = useState<SlotName | null>(null);
  const [categorizedModels, setCategorizedModels] = useState<Record<string, string[]>>({});
  const [asrModels, setAsrModels] = useState<string[]>([]);

  const activeLlm   = slots.llm;
  const activeEmbed = slots.embed;
  const activeAsr   = slots.asr;

  // ---- session state ------------------------------------------------------
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null);

  // ---- chat state ---------------------------------------------------------
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // ---- folder ingestion state ---------------------------------------------
  const [ingestModalOpen, setIngestModalOpen]     = useState(false);
  const [ingestFiles, setIngestFiles]             = useState<IngestFileEntry[]>([]);
  const [ingestProgress, setIngestProgress]       = useState(0);
  const [ingestDone, setIngestDone]               = useState(false);


  const fetchModels = useCallback(async () => {
    try {
      const [modelsRes, asrRes, statusRes] = await Promise.all([
        fetch(`${API_BASE}/api/models`),
        fetch(`${API_BASE}/api/whisper/models`),
        fetch(`${API_BASE}/api/models/status`),
      ]);
      const modelsData  = await modelsRes.json();
      const asrData     = await asrRes.json();
      const statusData  = await statusRes.json();

      setCategorizedModels(modelsData.categorized_models || {});
      setAsrModels(asrData.models || []);
      setSlots(statusData);
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    const id = setInterval(fetchModels, 5000);
    return () => clearInterval(id);
  }, [fetchModels]);


  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`);
      if (!res.ok) return;
      const data = await res.json();
      setSessions(data.sessions ?? []);
    } catch (err) {
      console.error("Failed to fetch sessions:", err);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  /** Create a new session and switch to it (clears current messages). */
  const handleNewChat = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Chat" }),
      });
      if (!res.ok) throw new Error("Failed to create session");
      const session: ChatSession = await res.json();
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
    } catch (err) {
      console.error("New chat error:", err);
    }
  }, []);

  const handleSelectSession = useCallback(async (session: ChatSession) => {
    if (session.id === activeSessionId) return;
    setActiveSessionId(session.id);
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${session.id}/messages`);
      if (!res.ok) throw new Error("Failed to load messages");
      const data = await res.json();
      const uiMessages: Message[] = (data.messages as DBMessage[]).map(dbMsgToUiMsg);
      setMessages(uiMessages);
    } catch (err) {
      console.error("Load session messages error:", err);
      setMessages([]);
    }
  }, [activeSessionId]);

  /** Delete a session; if it was active, clear the chat. */
  const handleDeleteSession = useCallback(async (sessionId: number) => {
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Delete failed");
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error("Delete session error:", err);
    }
  }, [activeSessionId]);

  /** Rename a session locally and on the server. */
  const handleRenameSession = useCallback(async (sessionId: number, newTitle: string) => {
    // Optimistic update
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, title: newTitle } : s))
    );
    try {
      await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
    } catch (err) {
      console.error("Rename session error:", err);
      // Rollback on error
      fetchSessions();
    }
  }, [fetchSessions]);

  const ensureActiveSession = useCallback(async (): Promise<number | null> => {
    if (activeSessionId !== null) return activeSessionId;
    try {
      const res = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Chat" }),
      });
      if (!res.ok) return null;
      const session: ChatSession = await res.json();
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      return session.id;
    } catch {
      return null;
    }
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSessionId) return;
    const id = setTimeout(() => {
      fetchSessions();
    }, 2000);
    return () => clearTimeout(id);
  }, [messages, activeSessionId, fetchSessions]);

  const handleMount = useCallback(async (slot: SlotName, modelName: string) => {
    setBusySlot(slot);
    try {
      const res = await fetch(`${API_BASE}/api/models/mount`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slot, model_name: modelName }),
      });

      const contentType = res.headers.get("content-type") ?? "";
      if (contentType.includes("application/x-ndjson")) {
        const reader = res.body!.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const d = JSON.parse(line);
              if (d.status === "success") {
                setSlots((prev) => ({ ...prev, [slot]: modelName }));
              }
            } catch { /* partial line */ }
          }
        }
      } else {
        const d = await res.json();
        if (d.status === "success") {
          setSlots((prev) => ({ ...prev, [slot]: modelName }));
        }
      }
    } catch (err) {
      console.error("Mount failed:", err);
    } finally {
      setBusySlot(null);
    }
  }, []);

  const handleUnmount = useCallback(async (slot: SlotName) => {
    setBusySlot(slot);
    try {
      await fetch(`${API_BASE}/api/models/unmount`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slot }),
      });
      setSlots((prev) => ({ ...prev, [slot]: null }));
    } catch (err) {
      console.error("Unmount failed:", err);
    } finally {
      setBusySlot(null);
    }
  }, []);

  const SUPPORTED_EXTS = new Set([
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
    ".txt", ".md",
    ".pdf",
  ]);

  function fileCategory(name: string): IngestFileEntry["category"] {
    const ext = name.slice(name.lastIndexOf(".")).toLowerCase();
    if ([".mp3",".wav",".m4a",".ogg",".flac"].includes(ext))              return "audio";
    if ([".mp4",".mov",".avi",".mkv",".webm",".m4v"].includes(ext))       return "video";
    if ([".png",".jpg",".jpeg",".gif",".webp",".bmp",".tiff"].includes(ext)) return "image";
    if ([".txt",".md"].includes(ext))                                     return "text";
    if (ext === ".pdf")                                                   return "pdf";
    return "unknown";
  }

  const handleFolderIngest = async (pickedFiles: File[]) => {
    const sid = await ensureActiveSession();

    const entries: IngestFileEntry[] = pickedFiles.map((f) => {
      const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
      const supported = SUPPORTED_EXTS.has(ext);
      return {
        id: Math.random().toString(36).slice(2) + Date.now().toString(36),
        name: f.name,
        category: fileCategory(f.name),
        status: supported ? "pending" : "skipped",
      };
    });

    setIngestFiles(entries);
    setIngestProgress(0);
    setIngestDone(false);
    setIngestModalOpen(true);

    const supportedEntries = entries.filter((e) => e.status !== "skipped");
    const total = supportedEntries.length;

    const allContextMessages: Message[] = [];
    const allInfoMessages:    Message[] = [];
    const allTranscriptMessages: Message[] = [];

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (entry.status === "skipped") continue;

      setIngestFiles((prev) =>
        prev.map((e) => (e.id === entry.id ? { ...e, status: "processing" } : e))
      );

      const file = pickedFiles[i];
      const formData = new FormData();
      formData.append("files", file);
      if (activeEmbed) formData.append("embedding_model", activeEmbed);
      if (activeAsr)   formData.append("asr_model", activeAsr);
      if (sid !== null) formData.append("session_id", String(sid));

      try {
        const res  = await fetch(`${API_BASE}/api/files/ingest`, { method: "POST", body: formData });
        const data = await res.json();
        const r    = (data.results ?? [])[0] as {
          filename: string; category: string; status: string;
          chunks: number; detail: string; text?: string;
        } | undefined;

        if (!r) throw new Error("Empty response");

        const newStatus: IngestFileEntry["status"] =
          r.status === "success"   ? "done"
          : r.status === "duplicate" ? "duplicate"
          : r.status === "error"     ? "error"
          : "done";

        setIngestFiles((prev) =>
          prev.map((e) =>
            e.id === entry.id
              ? { ...e, status: newStatus, chunks: r.chunks, detail: r.detail }
              : e
          )
        );

        const hasText = r.text && r.text.trim().length > 0;
        const infoBody =
          r.status === "error"
            ? `**${r.filename}** — error: ${r.detail}`
            : r.status === "duplicate"
            ? `**${r.filename}** — already in knowledge base (${r.chunks} chunk${r.chunks !== 1 ? "s" : ""}), skipped re-embedding`
            : r.chunks > 0
            ? `**${r.filename}** — ${r.chunks} chunk${r.chunks !== 1 ? "s" : ""} embedded`
            : hasText
            ? `**${r.filename}** — text extracted, added to context`
            : `**${r.filename}** — no text could be extracted`;

        allInfoMessages.push({ id: uid(), role: "system" as const, content: infoBody });

        if ((r.status === "success" || r.status === "duplicate") && hasText) {
          allContextMessages.push({
            id: uid(),
            role: "context" as const,
            content: `[File: ${r.filename}]\n${r.text}`,
          });
        }
        if (r.category === "audio" && (r.status === "success" || r.status === "duplicate") && hasText) {
          allTranscriptMessages.push({
            id: uid(),
            role: "assistant" as const,
            content: `Transcription of ${r.filename}:\n\n${r.text!.trim()}`,
            isTranscript: true,
          });
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : "failed";
        setIngestFiles((prev) =>
          prev.map((e) => (e.id === entry.id ? { ...e, status: "error", detail } : e))
        );
        allInfoMessages.push({
          id: uid(),
          role: "system" as const,
          content: `**${entry.name}** — error: ${detail}`,
        });
      }

      // Update overall progress (only count supported files)
      const doneNow = i + 1 - entries.slice(0, i + 1).filter((e) => e.status === "skipped").length;
      setIngestProgress(Math.round((doneNow / total) * 100));
    }

    // Mark complete and inject messages into chat
    setIngestProgress(100);
    setIngestDone(true);

    // Add a summary user message so the session has context
    const fileNames = pickedFiles.map((f) => f.name).join(", ");
    const summaryUserMsg: Message = {
      id: uid(),
      role: "user",
      content: `Attached folder: ${fileNames}`,
    };

    setMessages((prev) => [
      ...prev,
      summaryUserMsg,
      ...allContextMessages,
      ...allInfoMessages,
      ...allTranscriptMessages,
    ]);
  };

  const handleSendMessage = async (text: string, files: AttachedFile[] = []) => {
    setIsStreaming(true);

    const sid = await ensureActiveSession();

    let workingMessages = [...messages];

    if (files.length > 0) {
      const fileNames = files.map((f) => f.name).join(", ");
      const userFileMsg: Message = {
        id: uid(),
        role: "user",
        content: text ? `${text}\n\nAttached: ${fileNames}` : `Attached: ${fileNames}`,
      };
      workingMessages = [...workingMessages, userFileMsg];
      setMessages(workingMessages);

      const formData = new FormData();
      for (const f of files) formData.append("files", f.file);
      if (activeEmbed) formData.append("embedding_model", activeEmbed);
      if (activeAsr)   formData.append("asr_model", activeAsr);
      if (sid !== null) formData.append("session_id", String(sid));

      try {
        const ingestRes = await fetch(`${API_BASE}/api/files/ingest`, {
          method: "POST",
          body: formData,
        });
        const ingestData = await ingestRes.json();

        const infoMessages: Message[] = (ingestData.results ?? []).map(
          (r: { filename: string; category: string; status: string; chunks: number; detail: string; text?: string }) => {
            const hasText = r.text && r.text.trim().length > 0;
            const body =
              r.status === "error"
                ? `**${r.filename}** — error: ${r.detail}`
                : r.status === "duplicate"
                ? `**${r.filename}** — already in knowledge base (${r.chunks} chunk${r.chunks !== 1 ? "s" : ""}), skipped re-embedding`
                : r.chunks > 0
                ? `**${r.filename}** — ${r.chunks} chunk${r.chunks !== 1 ? "s" : ""} embedded`
                : hasText
                ? `**${r.filename}** — text extracted, added to context`
                : `**${r.filename}** — no text could be extracted`;
            return { id: uid(), role: "system" as const, content: body };
          }
        );

        const contextMessages: Message[] = (ingestData.results ?? [])
          .filter((r: { status: string; text?: string }) => (r.status === "success" || r.status === "duplicate") && r.text && r.text.trim())
          .map((r: { filename: string; text: string }) => ({
            id: uid(),
            role: "context" as const,
            content: `[File: ${r.filename}]\n${r.text}`,
          }));

        const transcriptMessages: Message[] = (ingestData.results ?? [])
          .filter(
            (r: { category: string; status: string; text?: string }) =>
              r.category === "audio" && (r.status === "success" || r.status === "duplicate") && r.text && r.text.trim()
          )
          .map((r: { filename: string; text: string }) => ({
            id: uid(),
            role: "assistant" as const,
            content: `Transcription of ${r.filename}:\n\n${r.text.trim()}`,
            isTranscript: true,
          }));

        workingMessages = [...workingMessages, ...contextMessages];

        if (infoMessages.length > 0) {
          workingMessages = [...workingMessages, ...infoMessages, ...transcriptMessages];
          setMessages(workingMessages);
        }
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : "Upload failed";
        const errInfoMsg: Message = { id: uid(), role: "system", content: `⚠ File upload error: ${errMsg}` };
        workingMessages = [...workingMessages, errInfoMsg];
        setMessages(workingMessages);
      }

      if (!text.trim()) {
        const hasAudioContext = workingMessages.some(
          (m) => m.role === "context" && m.content.match(/\[File:.*\.(mp3|wav|m4a|ogg|flac)\]/i)
        );
        if (hasAudioContext) {
          const autoPrompt: Message = {
            id: uid(),
            role: "user",
            content: "Please summarise the transcribed audio.",
          };
          workingMessages = [...workingMessages, autoPrompt];
          setMessages(workingMessages);
        } else {
          setIsStreaming(false);
          return;
        }
      }
    } else {
      const userMsg: Message = { id: uid(), role: "user", content: text };
      workingMessages = [...workingMessages, userMsg];
      setMessages(workingMessages);
    }

    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const llmMessages = workingMessages
      .filter((m) => (m.role === "user" || m.role === "assistant" || m.role === "context") && !m.isTranscript)
      .map((m) => ({ role: m.role === "context" ? "user" : m.role, content: m.content }));

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          messages: llmMessages,
          session_id: sid,
          embedding_model: activeEmbed,
          max_tokens: 1024,
          temperature: 0.7,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ message: "Unknown error" }));
        throw new Error(err.message || `HTTP ${res.status}`);
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const chunk = JSON.parse(line);
            if (chunk.error) throw new Error(chunk.error);
            if (chunk.token) {
              accumulated += chunk.token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: accumulated } : m
                )
              );
            }
          } catch {
          }
        }
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, streaming: false } : m
        )
      );
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m
          )
        );
      } else {
        const msg = err instanceof Error ? err.message : "Failed to reach the LLM.";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `Error: ${msg}`, streaming: false }
              : m
          )
        );
      }
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  };

  const handleStopStreaming = () => {
    abortControllerRef.current?.abort();
  };


  return (
    <div className="flex h-screen w-full bg-background bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-accent/30 via-background to-background text-foreground overflow-hidden font-sans">
      <Sidebar
        isOpen={isSidebarOpen}
        onOpenSettings={() => setIsSettingsOpen(true)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onRenameSession={handleRenameSession}
      />

      <div className="absolute top-4 left-4 z-30" style={{ left: isSidebarOpen ? "calc(260px + 1.75rem)" : "1rem", transition: "left 300ms cubic-bezier(0.16,1,0.3,1)" }}>
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="p-2 rounded-xl hover:bg-panel/80 backdrop-blur-xl border border-transparent hover:border-panel-border text-muted hover:text-foreground transition-all shadow-sm group"
        >
          <Menu size={20} className="group-hover:scale-105 transition-transform" />
        </button>
      </div>

      <div
        className="absolute top-4 right-4 z-30"
        style={{
          right: isModelSidebarOpen ? "calc(300px + 1.75rem)" : "1rem",
          transition: "right 300ms cubic-bezier(0.16,1,0.3,1)",
        }}
      >
        <button
          onClick={() => setIsModelSidebarOpen(!isModelSidebarOpen)}
          className="p-2 rounded-xl hover:bg-panel/80 backdrop-blur-xl border border-transparent hover:border-panel-border text-muted hover:text-foreground transition-all shadow-sm group"
          title="Manage Models"
        >
          <Settings size={20} className="group-hover:scale-105 transition-transform" />
        </button>
      </div>

      <div
        className={`flex-1 flex flex-col h-full relative min-w-0 transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          isSidebarOpen || isModelSidebarOpen
            ? "scale-[0.94] rounded-2xl brightness-75 cursor-pointer"
            : "scale-100 brightness-100 cursor-auto"
        } ${
          isSidebarOpen && !isModelSidebarOpen
            ? "origin-right"
            : isModelSidebarOpen && !isSidebarOpen
            ? "origin-left"
            : "origin-center"
        }`}
        onClick={
          isSidebarOpen
            ? () => setIsSidebarOpen(false)
            : isModelSidebarOpen
            ? () => setIsModelSidebarOpen(false)
            : undefined
        }
      >

        <header className="absolute top-0 w-full flex items-center justify-between px-4 py-4 z-20 bg-gradient-to-b from-background via-background/80 to-transparent pointer-events-none">

          <div className="w-9 h-9" />

          <div className="pointer-events-auto">
            <ModelBar
              slots={slots}
              categorizedModels={categorizedModels}
              asrModels={asrModels}
              onMount={handleMount}
              onUnmount={handleUnmount}
              busySlot={busySlot}
            />
          </div>

          <div className="w-9 h-9" />
        </header>

        <ChatArea messages={messages} isStreaming={isStreaming} />

        <InputArea
          message={message}
          setMessage={setMessage}
          onSendMessage={handleSendMessage}
          isStreaming={isStreaming}
          onStopStreaming={handleStopStreaming}
          onFolderIngest={handleFolderIngest}
        />
      </div>

      <ModelSelectionModal
        isOpen={isModelSidebarOpen}
        onClose={() => {
          setIsModelSidebarOpen(false);
          fetchModels();
        }}
        slots={slots}
        onMount={handleMount}
        onUnmount={handleUnmount}
        busySlot={busySlot}
      />

      <IngestionProgressModal
        isOpen={ingestModalOpen}
        onClose={() => setIngestModalOpen(false)}
        files={ingestFiles}
        overallProgress={ingestProgress}
        isDone={ingestDone}
      />
    </div>
  );
}
