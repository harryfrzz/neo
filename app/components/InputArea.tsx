"use client";

import React, { useRef, useEffect } from "react";
import { Paperclip, FolderOpen, ArrowUp, X, StopCircle, Music, Image, FileText, File, Video, AlertCircle } from "lucide-react";


export type FileCategory = "audio" | "image" | "text" | "pdf" | "video" | "unknown";
export type FileStatus = "idle" | "uploading" | "done" | "error";

export interface AttachedFile {
  id: string;
  name: string;
  category: FileCategory;
  status: FileStatus;
  file: File;
  detail?: string;
  chunks?: number;
}

interface InputAreaProps {
  message: string;
  setMessage: (message: string) => void;
  onSendMessage: (text: string, files: AttachedFile[]) => void;
  isStreaming: boolean;
  onStopStreaming: () => void;
  onFolderIngest?: (files: File[]) => void;
  hasModel: boolean;
  onOpenModels: () => void;
}

const AUDIO_EXTS = new Set([".mp3", ".wav", ".m4a", ".ogg", ".flac"]);
const VIDEO_EXTS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]);
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"]);
const TEXT_EXTS  = new Set([".txt", ".md"]);
const PDF_EXTS   = new Set([".pdf"]);

function categorise(filename: string): FileCategory {
  const ext = filename.slice(filename.lastIndexOf(".")).toLowerCase();
  if (AUDIO_EXTS.has(ext)) return "audio";
  if (VIDEO_EXTS.has(ext)) return "video";
  if (IMAGE_EXTS.has(ext)) return "image";
  if (TEXT_EXTS.has(ext))  return "text";
  if (PDF_EXTS.has(ext))   return "pdf";
  return "unknown";
}

function CategoryIcon({ category }: { category: FileCategory }) {
  const cls = "shrink-0";
  switch (category) {
    case "audio":   return <Music     size={14} className={`${cls} text-violet-400`} />;
    case "video":   return <Video     size={14} className={`${cls} text-pink-400`} />;
    case "image":   return <Image     size={14} className={`${cls} text-sky-400`} />;
    case "text":    return <FileText  size={14} className={`${cls} text-emerald-400`} />;
    case "pdf":     return <FileText  size={14} className={`${cls} text-orange-400`} />;
    default:        return <File      size={14} className={`${cls} text-muted`} />;
  }
}

const STATUS_RING: Record<FileStatus, string> = {
  idle:      "border-panel-border/60",
  uploading: "border-primary/60 animate-pulse",
  done:      "border-emerald-500/60",
  error:     "border-red-500/60",
};

const STATUS_LABEL: Record<FileStatus, string | null> = {
  idle:      null,
  uploading: "uploading…",
  done:      null,
  error:     "error",
};

export default function InputArea({
  message,
  setMessage,
  onSendMessage,
  isStreaming,
  onStopStreaming,
  onFolderIngest,
  hasModel,
  onOpenModels,
}: InputAreaProps) {
  const textareaRef    = useRef<HTMLTextAreaElement>(null);
  const fileInputRef   = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const [attachedFiles, setAttachedFiles] = React.useState<AttachedFile[]>([]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 192)}px`;
    }
  }, [message]);

  const handleSend = () => {
    const text = message.trim();
    if ((!text && attachedFiles.length === 0) || isStreaming) return;
    setMessage("");
    const toSend = attachedFiles;
    setAttachedFiles([]);
    onSendMessage(text, toSend);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newItems: AttachedFile[] = Array.from(e.target.files).map((file) => ({
        id: Math.random().toString(36).slice(2) + Date.now().toString(36),
        name: file.name,
        category: categorise(file.name),
        status: "idle",
        file,
      }));
      setAttachedFiles((prev) => [...prev, ...newItems]);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const picked = Array.from(e.target.files);
      if (onFolderIngest) {
        onFolderIngest(picked);
      } else {
        const newItems: AttachedFile[] = picked.map((file) => ({
          id: Math.random().toString(36).slice(2) + Date.now().toString(36),
          name: file.name,
          category: categorise(file.name),
          status: "idle",
          file,
        }));
        setAttachedFiles((prev) => [...prev, ...newItems]);
      }
    }
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const removeFile = (id: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const canSend = (message.trim().length > 0 || attachedFiles.length > 0) && !isStreaming && hasModel;

  return (
    <div className="w-full pb-8 pt-2 px-4 sm:px-0 z-10">
      <div className="max-w-3xl mx-auto relative flex flex-col gap-3">

        {!hasModel && !isStreaming && (
          <div className="flex justify-center animate-in fade-in duration-300">
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs border border-amber-500/20 bg-amber-500/5 backdrop-blur-xl text-amber-400/70">
              <AlertCircle size={11} className="shrink-0" />
              <span>No model connected —</span>
              <button
                onClick={onOpenModels}
                className="underline underline-offset-2 hover:text-amber-300 transition-colors"
              >
                Connect a model
              </button>
            </div>
          </div>
        )}

        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 px-2">
            {attachedFiles.map((item) => (
              <div
                key={item.id}
                className={`flex items-center gap-1.5 bg-panel/60 backdrop-blur-md border ${STATUS_RING[item.status]} rounded-xl px-3 py-1.5 text-sm shadow-sm animate-in zoom-in-95 duration-200`}
              >
                <CategoryIcon category={item.category} />
                <span className="text-foreground/90 max-w-[150px] truncate font-medium">
                  {item.name}
                </span>
                {STATUS_LABEL[item.status] && (
                  <span className={`text-[10px] font-medium ${item.status === "error" ? "text-red-400" : "text-primary/70"}`}>
                    {STATUS_LABEL[item.status]}
                  </span>
                )}
                {item.status === "done" && item.chunks !== undefined && (
                  <span className="text-[10px] text-emerald-400 font-medium">
                    {item.chunks} chunk{item.chunks !== 1 ? "s" : ""}
                  </span>
                )}
                {item.status !== "uploading" && (
                  <button
                    onClick={() => removeFile(item.id)}
                    className="ml-0.5 p-0.5 text-muted hover:text-foreground hover:bg-foreground/10 rounded-full transition-colors"
                  >
                    <X size={13} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Hidden inputs */}
        <input
          type="file"
          multiple
          className="hidden"
          ref={fileInputRef}
          onChange={handleFileSelect}
        />
        <input
          type="file"
          // @ts-expect-error — non-standard but widely supported
          webkitdirectory="true"
          directory="true"
          className="hidden"
          ref={folderInputRef}
          onChange={handleFolderSelect}
        />

        {/* Input bar */}
        <div className="relative flex items-end w-full border border-panel-border/80 bg-panel/50 backdrop-blur-2xl shadow-2xl rounded-[32px] p-1.5 focus-within:border-muted/60 focus-within:ring-4 focus-within:ring-accent/20 transition-all duration-300">

          <div className="flex items-center gap-1 mb-0.5 ml-1">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming}
              className="p-2.5 text-muted hover:text-foreground hover:bg-foreground/5 rounded-full transition-all duration-200 disabled:opacity-40"
              title="Attach files"
            >
              <Paperclip size={20} />
            </button>
            <button
              onClick={() => folderInputRef.current?.click()}
              disabled={isStreaming}
              className="p-2.5 text-muted hover:text-foreground hover:bg-foreground/5 rounded-full transition-all duration-200 disabled:opacity-40"
              title="Open folder"
            >
              <FolderOpen size={20} />
            </button>
          </div>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isStreaming || !hasModel}
            placeholder={isStreaming ? "Generating..." : !hasModel ? "Connect a model to start chatting..." : "Type your message..."}
            className="flex-1 max-h-48 min-h-[44px] bg-transparent border-none text-foreground text-[15px] px-3 py-3 mx-1 resize-none focus:outline-none focus:ring-0 leading-relaxed disabled:opacity-60 disabled:cursor-not-allowed"
            rows={1}
          />

          <div className="mb-0.5 mr-1">
            {isStreaming ? (
              <button
                onClick={onStopStreaming}
                className="p-2.5 rounded-full bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-all duration-200 flex items-center justify-center"
                title="Stop generation"
              >
                <StopCircle size={20} />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!canSend}
                className={`p-2.5 rounded-full transition-all duration-300 flex items-center justify-center ${
                  canSend
                    ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg shadow-primary/30 scale-100"
                    : "bg-foreground/5 text-muted/50 cursor-not-allowed scale-95"
                }`}
              >
                <ArrowUp size={20} strokeWidth={2.5} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
