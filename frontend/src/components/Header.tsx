import { CircleDot, Sparkles } from 'lucide-react';
import { Button } from './ui/button';
import { CHAT_SHELL_CLASS } from '../config/layout';

interface HeaderProps {
  onChatClear: () => void;
}

export function Header({ onChatClear }: HeaderProps) {
  return (
    <header className="shrink-0 border-b border-slate-200/80 bg-white/88 backdrop-blur-sm">
      <div className={`${CHAT_SHELL_CLASS} flex h-14 items-center justify-between gap-3`}>
        <div className="flex min-w-0 items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[12px] border border-slate-200 bg-slate-50 text-slate-700">
            <Sparkles className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-[15px] font-semibold tracking-tight text-slate-900">Data Agent</div>
            <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
              <CircleDot className="h-3 w-3 fill-emerald-500 text-emerald-500" />
              <span>Single-column chat</span>
            </div>
          </div>
        </div>

        <Button
          onClick={onChatClear}
          variant="outline"
          size="sm"
          className="h-8 rounded-[12px] border-slate-200 bg-white px-3 text-[11px] font-semibold text-slate-700 shadow-none hover:bg-slate-50"
        >
          清除对话
        </Button>
      </div>
    </header>
  );
}
