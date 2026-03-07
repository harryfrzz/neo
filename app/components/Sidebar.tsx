"use client";

import { Plus, Trash2, MessageSquare, Pencil, Check, X } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { ChatSession } from "../types/chat";

interface SidebarProps {
  isOpen: boolean;
  onOpenSettings: () => void;
  sessions: ChatSession[];
  activeSessionId: number | null;
  onNewChat: () => void;
  onSelectSession: (session: ChatSession) => void;
  onDeleteSession: (sessionId: number) => void;
  onRenameSession: (sessionId: number, newTitle: string) => void;
}

export default function Sidebar({
  isOpen,
  onOpenSettings,
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId !== null && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const startEdit = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const commitEdit = () => {
    if (editingId !== null && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
  };

  const cancelEdit = () => {
    setEditingId(null);
  };

  const confirmDelete = (sessionId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(sessionId);
  };

  const executeDelete = (sessionId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    onDeleteSession(sessionId);
    setDeletingId(null);
  };

  const cancelDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeletingId(null);
  };

  // Format relative time for session list
  const formatTime = (isoDate: string) => {
    const date = new Date(isoDate);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  return (
    <>
      <div
        className={`flex flex-col bg-panel/40 backdrop-blur-2xl border border-panel-border rounded-2xl transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] my-3 ml-3 relative z-20 ${
          isOpen
            ? "w-[260px] opacity-100 translate-x-0"
            : "w-0 opacity-0 -translate-x-10 !ml-0 !border-transparent"
        } overflow-hidden shrink-0 shadow-2xl`}
      >

        <div className="flex flex-col h-full p-3 gap-2 w-[260px] overflow-hidden">
          {/* New Chat Button */}
          <button
            onClick={onNewChat}
            className="flex items-center gap-2 w-full p-2.5 rounded-xl hover:bg-accent/80 transition-all text-sm font-medium border border-transparent hover:border-panel-border text-foreground hover:shadow-sm"
          >
            <Plus size={16} />
            <span>New Chat</span>
          </button>

          {/* Session list */}
          {sessions.length > 0 && (
            <div className="flex flex-col gap-0.5 overflow-y-auto flex-1 no-scrollbar mt-1">
              {sessions.map((session) => {
                const isActive = session.id === activeSessionId;
                const isEditing = editingId === session.id;
                const isDeleting = deletingId === session.id;

                return (
                  <div
                    key={session.id}
                    onClick={() => !isEditing && onSelectSession(session)}
                    className={`group relative flex items-center gap-2 w-full px-2.5 py-2 rounded-xl cursor-pointer transition-all text-sm border ${
                      isActive
                        ? "bg-accent/60 border-panel-border text-foreground"
                        : "border-transparent hover:bg-accent/40 hover:border-panel-border/50 text-muted hover:text-foreground"
                    }`}
                  >
                    <MessageSquare size={13} className="shrink-0 opacity-60" />

                    {isEditing ? (
                      <div className="flex items-center gap-1 flex-1 min-w-0">
                        <input
                          ref={editInputRef}
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") commitEdit();
                            if (e.key === "Escape") cancelEdit();
                          }}
                          onClick={(e) => e.stopPropagation()}
                          className="flex-1 min-w-0 bg-background/60 border border-panel-border rounded-md px-2 py-0.5 text-xs text-foreground outline-none focus:border-primary/50"
                        />
                        <button
                          onClick={(e) => { e.stopPropagation(); commitEdit(); }}
                          className="text-emerald-400 hover:text-emerald-300 transition-colors"
                        >
                          <Check size={13} />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); cancelEdit(); }}
                          className="text-muted hover:text-foreground transition-colors"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    ) : isDeleting ? (
                      <div className="flex items-center gap-1 flex-1 min-w-0">
                        <span className="text-xs text-red-400 truncate flex-1">Delete?</span>
                        <button
                          onClick={(e) => executeDelete(session.id, e)}
                          className="text-red-400 hover:text-red-300 transition-colors text-xs font-medium"
                        >
                          Yes
                        </button>
                        <button
                          onClick={cancelDelete}
                          className="text-muted hover:text-foreground transition-colors"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    ) : (
                      <>
                        <div className="flex-1 min-w-0">
                          <p className="truncate text-xs leading-tight">{session.title}</p>
                          <p className="text-[10px] text-muted/50 mt-0.5">
                            {formatTime(session.updated_at)}
                          </p>
                        </div>
                        {/* Action buttons — show on hover or when active */}
                        <div
                          className={`flex items-center gap-0.5 shrink-0 transition-opacity ${
                            isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          }`}
                        >
                          <button
                            onClick={(e) => startEdit(session, e)}
                            className="p-1 rounded-md hover:bg-panel-border/60 text-muted hover:text-foreground transition-colors"
                            title="Rename"
                          >
                            <Pencil size={11} />
                          </button>
                          <button
                            onClick={(e) => confirmDelete(session.id, e)}
                            className="p-1 rounded-md hover:bg-red-500/20 text-muted hover:text-red-400 transition-colors"
                            title="Delete"
                          >
                            <Trash2 size={11} />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {sessions.length === 0 && (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-xs text-muted/40 text-center px-4">
                No chats yet. Start a conversation!
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
