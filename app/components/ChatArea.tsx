"use client";

import { useEffect, useRef } from "react";
import { marked } from "marked";
import ScrambledText from "./ScrambledText";
import { Mic, Image as ImageIcon, Video, FileText, Cpu } from "lucide-react";

marked.use({ gfm: true, breaks: true });

export interface Message {
  id: string;
  role: "user" | "assistant" | "system" | "context";
  content: string;
  streaming?: boolean;
  isTranscript?: boolean;
}

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
}

const options = [
  { name: "Audio", icon: Mic },
  { name: "Image", icon: ImageIcon },
  { name: "Video", icon: Video },
  { name: "Docs", icon: FileText },
];

export default function ChatArea({ messages, isStreaming }: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Landing screen
  if (messages.length === 0) {
    return (
      <div className="flex-1 overflow-y-auto flex flex-col pt-20 pb-4 sm:px-12 no-scrollbar relative z-10">
        <div className="flex flex-col md:flex-row justify-center items-center h-full w-full max-w-5xl mx-auto gap-12 md:gap-20">
          <div className="flex flex-col items-center md:items-start text-center md:text-left flex-shrink-0">
            <ScrambledText
              className="!m-0 text-7xl sm:text-7xl tracking-[0.2em] uppercase leading-[1.1]"
              radius={60}
              duration={1.2}
              speed={0.3}
              scrambleChars="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%"
            >
              NEO
              <br />
              Chat
            </ScrambledText>
          </div>

          <div className="hidden md:block w-[1px] h-50 bg-panel-border/60" />
          <div className="md:hidden h-[1px] w-48 bg-panel-border/60" />

          <div className="flex flex-wrap justify-center md:justify-start gap-3 w-full max-w-md">
            {options.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.name}
                  className="group bg-panel border border-panel-border/60 hover:border-white/40 hover:bg-white/10 rounded-full px-6 py-3 flex flex-row items-center justify-center gap-2 transition-all duration-300 ease-out cursor-pointer shadow-sm hover:shadow-emerald-500/10 relative"
                >
                  <Icon className="w-5 h-5 text-muted group-hover:text-white transition-colors" />
                  <span className="text-sm font-medium text-foreground group-hover:text-white transition-colors">
                    {item.name}
                  </span>
                  {item.name === "Video" && (
                    <span className="absolute -top-2 -right-2 text-[9px] font-semibold uppercase tracking-wide bg-amber-500/20 text-amber-400 border border-amber-500/30 rounded-full px-1.5 py-0.5 leading-none">
                      Beta
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto flex flex-col pt-20 pb-4 no-scrollbar relative z-10">
      <div className="w-full max-w-3xl mx-auto px-4 flex flex-col gap-6 py-6">
        {messages.map((msg) => {
          if (msg.role === "context") return null;

          // System / info messages — centred pill
          if (msg.role === "system") {
            return (
              <div key={msg.id} className="flex justify-center">
                <div className="text-xs text-muted/80 bg-panel/40 border border-panel-border/40 rounded-full px-4 py-1.5 max-w-[90%] text-center">
                  {msg.content}
                </div>
              </div>
            );
          }

          return (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >

            {msg.role === "user" ? (
              <div className="relative max-w-[80%] px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed whitespace-pre-wrap break-words bg-primary text-primary-foreground">
                {msg.content}
              </div>
            ) : (
              <div className="relative max-w-[80%] px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-relaxed bg-panel border border-panel-border/60 text-foreground">
                <div
                  className="prose-msg"
                  dangerouslySetInnerHTML={{ __html: marked.parse(msg.content) as string }}
                />
                {/* Blinking cursor while streaming */}
                {msg.streaming && (
                  <span className="inline-block w-[2px] h-[1em] bg-current ml-0.5 align-middle animate-pulse" />
                )}
              </div>
            )}
          </div>
          );
        })}

        {/* Typing indicator when we're waiting for the first token */}
        {isStreaming && messages.filter(m => m.role !== "system").at(-1)?.role !== "assistant" && (
          <div className="flex gap-3 justify-start">
            <div className="shrink-0 w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center mt-0.5">
              <Cpu size={14} className="text-primary" />
            </div>
            <div className="bg-panel border border-panel-border/60 rounded-2xl rounded-bl-sm px-4 py-3 flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-muted/60 animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-muted/60 animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-muted/60 animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
